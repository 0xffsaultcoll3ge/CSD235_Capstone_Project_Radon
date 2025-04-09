from flask import *
from flask_cors import CORS
from flask_sso import SSO
from sqlalchemy import create_engine, MetaData
from sqlalchemy.orm import sessionmaker
import pandas as pd
import sys


sys.path.insert(1, 'backend/model')
sys.path.insert(2, 'backend/db')
sys.path.insert(3, './backend/api')
from nhl_model import NHLModel, best_model_path
import os
from dotenv import load_dotenv
from nhl_train import NHLModelTrainer
from db import NHLPipeline, create_table_map
from flask_login import LoginManager
from models import User, db
from auth import auth_bp

from flask import request, jsonify
from subscriptions import create_subscription




load_dotenv()

app = Flask(__name__)
CORS(app, supports_credentials=True, origins=["http://localhost:3000"], allow_headers=["Content-Type"], methods=["GET", "POST", "OPTIONS"])  # FIXED CORS

app.config['SECRET_KEY'] = 'raghav-sharma-key'
BASE_DIR = os.path.abspath(os.path.dirname(__file__))  
app.config['SQLALCHEMY_DATABASE_URI'] = f'sqlite:///{os.path.join(BASE_DIR, "../users.db")}' 
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db.init_app(app)
login_manager = LoginManager(app)
login_manager.login_view = 'auth.login'

CORS(auth_bp, supports_credentials=True)  # fixed cors for auth routes
app.register_blueprint(auth_bp)

MODEL_DIR = os.getenv("MODEL_DIR")

nhl_trainer = NHLModelTrainer()
nhl_pipeline = NHLPipeline()
nhl_ml_model = NHLModel("ml", model_path=f"./backend/model/models/ML/XGBoot_61.2%_ML.json")
nhl_ou_model = lambda ou: NHLModel("ou", model_path = best_model_path("OU", MODEL_DIR, ou))
nhl_spread_model = lambda spread: NHLModel("spread", model_path = best_model_path("spread", MODEL_DIR, spread))

DATABASE_URL = "sqlite:///nhl.db"
engine = create_engine(DATABASE_URL)

Session = sessionmaker(bind=engine)
session = Session()

# User loader
@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

@app.route('/api/nhl/ml/predict', methods=['GET'])
def get_predictions_ml():
    try: 
        home = request.args.get('home')
        away = request.args.get('away')

        predictions = nhl_ml_model.get_team_prediction(home, away)
        return jsonify({"predictions":predictions.tolist()}), 200
    except Exception as e:
        return jsonify({"error":str(e)}), 500


@app.route('/api/nhl/ou/predict', methods=['GET'])
def get_predictions_ou():
    try:
        ou = float(request.args.get('ou'))
        home = request.args.get('home')
        away = request.args.get('away')

        predictions = nhl_ou_model(ou).get_team_prediction(home, away)
        return jsonify({"predictions":predictions.tolist()}), 200
    except Exception as e:
        return jsonify({"error":str(e)}), 500

    return jsonify({"predictions":predictions.tolist()}), 200
@app.route('/api/nhl/spread/predict', methods=['GET'])
def get_predictions_spread():
    try:
        spread = float(request.args.get('spread'))
        home = request.args.get('home')
        away = request.args.get('away')

        predictions = nhl_spread_model(spread).get_team_prediction(home, away)
        return jsonify({"predictions":predictions.tolist()}), 200
    except Exception as e:
        return jsonify({"error":str(e)}), 500

    return jsonify({"predictions":predictions.tolist()}), 200
@app.route('/api/nhl/team')
def get_team_data():
    try:
        team = request.args.get('team')
        q = f"SELECT * FROM {team} LIMIT 1"
        df = pd.read_sql(q, engine)

        return df.to_json()
    except Exception as e:
        return jsonify({"error": e})
@app.route('/api/nhl/odds/{gameId}')
def get_game_odds():
    try:
        q = "SELECT * FROM odds WHERE gameId = '{gameId}'"
        df = pd.read_sql(q, engine)

        return df.to_json()
    except Exception as e:
        return jsonify({"error": e})
    

#@app.route('/api/nhl/odds/<int:gameId>', methods=['GET'])
#def get_game_odds(gameId):  # Add 'gameId' as a parameter
#    try:
#        odds_format = request.args.get('odds_format', default='american', type=str)
#
#        q = "SELECT * FROM odds WHERE gameId = :gameId"
#        df = pd.read_sql(q, engine, params={"gameId": gameId})
#       
#        if odds_format == 'decimal':
#            df['odds'] = df['odds'].apply(lambda x: american_to_decimal(x))
#        elif odds_format == 'american':
#            pass 
#        else:
#            return jsonify({"error": "Invalid odds format. Use 'american' or 'decimal'."}), 400

#        return df.to_json(orient='records'), 200
#    except Exception as e:
#        return jsonify({"error": str(e)}), 500  
 
    
"""@app.route('/api/nhl/odds/historical/<int:gameId>', methods=['GET'])
def get_historical_odds(gameId): 
    try:
        q = "SELECT * FROM historicalOdds WHERE gameId = :gameId"
        df = pd.read_sql(q, engine, params={"gameId": gameId})

        return df.to_json(orient='records'), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500  """
    
def american_to_decimal(american_odds):
  
    if american_odds > 0:
        return (american_odds / 100) + 1
    else:
        return (100 / abs(american_odds)) + 1

@app.route('/api/nhl/train/ou')
def train_nhl_ou_model():
    try:
        ou = float(request.args.get('ou'))
        X, y = nhl_trainer.preprocess('ou', nhl_trainer.load_data(), value=ou)
        model, acc = nhl_trainer.train_ou(X, y, ou)
        return jsonify({'success': True, 'max_accuracy': acc })
    except Exception as e:
        print(e)
        return jsonify({'success': False})


@app.route('/api/nhl/train/spread')
def train_nhl_spread_model():
    try:
        spread = float(request.args.get('spread'))
        X, y = nhl_trainer.preprocess('spread', nhl_trainer.load_data(), value=spread)
        model, acc = nhl_trainer.train_spread(X, y, spread)
        return jsonify({'success': True, 'max_accuracy': acc })
    except Exception as e:
        print(e)
        return jsonify({'success': False})
@app.route('/api/nhl/train/ml')
def train_nhl_ml_model():
   try:
       ou = request.args.get('ou')
       X, y = nhl_trainer.preprocess('ml', nhl_trainer.load_data())
       model, acc = nhl_trainer.train_ml(X, y)
       return jsonify({'success': True, 'max_accuracy': acc })
   except Exception as e:
       print(e)
       return jsonify({'success': False})

@app.route('/api/nhl/data/update')
def train_nhl_update():
    mp = create_table_map()
    try:
        nhl_pipeline.update_team_db(mp)
        nhl_pipeline.preprocess_team_data("games_preproc")
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'success': False})
    

# STRIPE
    
@app.route('/api/stripe/subscription', methods=['POST'])
def create_embedded_subscription():
    data = request.json
    email = data.get('email')
    price_id = data.get('price_id')
    
    if not email or not price_id:
        return jsonify({"error": "Missing email or price_id"}), 400

    try:
        result = create_subscription(email, price_id)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    

with app.app_context():
    db.create_all()

if __name__ == "__main__":
    app.run(debug=True)  
