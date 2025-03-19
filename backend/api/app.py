from flask import *
from sqlalchemy import create_engine, MetaData
from sqlalchemy.orm import sessionmaker
import pandas as pd
import sys
sys.path.insert(1, './backend/model')
from nhl import NHLModel, best_model_path

app = Flask(__name__)
nhl_ml_model = NHLModel("ml", model_path="./backend/model/models/ML/XGBoot_60.1%_ML.json")
nhl_ou_model = lambda ou: return NHLModel("ou", model_path = best_model_path("ou", "./backend/model/models/ou", ou) )
nhl_spread_model = lambda spread: return NHLModel("spread", model_path = best_model_path("spread", "./backend/model/models/spread", spread))
# DATABASE_URL = "sqlite:///nhl.db"
# engine = create_engine(DATABASE_URL)

# Session = sessionmaker(bind=engine)
# session = Session()

@app.route('/api/nhl/ml/predict', methods=['GET'])
def get_predictions_ml():
    try: 
        home = request.args.get('home')
        away = request.args.get('away')

        predictions = nhl_ml_model.get_team_prediction(home, away)
        return jsonify({"predictions":predictions.tolist()}), 200
    except Exception as e:
        return jsonify({"error":str(e)}), 500


@app.route('/api/nhl/ou/<float:ou>/predict', methods=['GET'])
def get_predictions_ou():
    try: 
        home = request.args.get('home')
        away = request.args.get('away')

        predictions = nhl_ou_model(ou).get_team_prediction(home, away)
        return jsonify({"predictions":predictions.tolist()}), 200
    except Exception as e:
        return jsonify({"error":str(e)}), 500

    return jsonify({"predictions":predictions.tolist()}), 200
@app.route('/api/nhl/<float:spread/predict', methods=['GET'])
def get_predictions_spread():
    try:
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
        q = "SELECT * FROM games WHERE (team = '{team}' OR opposingTeam = '{team}')"
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
def train_nhl_model():
    try:
        ou = request.args.get('ou')
    except Exception as e:
        print(e)
@app.route('/api/nhl/train/ml')
def train_nhl_model():
    try:
        ou = request.args.get('ou')
    except Exception as e:
        print(e)

if __name__ == "__main__":
    app.run(debug=True)

        
