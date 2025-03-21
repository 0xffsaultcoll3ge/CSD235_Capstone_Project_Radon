from flask import *
from flask_sso import SSO
from sqlalchemy import create_engine, MetaData
from sqlalchemy.orm import sessionmaker
import pandas as pd
import sys
sys.path.insert(1, 'backend/model')
sys.path.insert(2, 'backend/db')
from nhl_model import NHLModel, best_model_path
import os
from dotenv import load_dotenv
from nhl_train import NHLModelTrainer
from db import NHLPipeline, create_table_map

load_dotenv()
app = Flask(__name__)


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
if __name__ == "__main__":
    app.run(debug=True)

        
