from flask import *
from sqlalchemy import create_engine, MetaData
from sqlalchemy.orm import sessionmaker
import pandas as pd
import sys
sys.path.insert(1, './backend/model')
from nhl import NHLModel

app = Flask(__name__)
nhl_ml_model = NHLModel("ml", model_path="./backend/model/models/XGBoot_60.2%_ML.json")
nhl_ou_model = NHLModel("ou", model_path="./backend/model/models/OU/XGBoot_64.3%_OU_6.0.json")
nhl_spread_model = NHLModel("spread", model_path="./backend/model/models/spread/XGBoot_76.5%_Spread_-1.5.json")
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


@app.route('/api/nhl/ou/predict', methods=['GET'])
def get_predictions_ou():
    try: 
        home = request.args.get('home')
        away = request.args.get('away')

        predictions = nhl_ou_model.get_team_prediction(home, away)
        return jsonify({"predictions":predictions.tolist()}), 200
    except Exception as e:
        return jsonify({"error":str(e)}), 500

    return jsonify({"predictions":predictions.tolist()}), 200
@app.route('/api/nhl/spread/predict', methods=['GET'])
def get_predictions_spread():
    try:
        home = request.args.get('home')
        away = request.args.get('away')

        predictions = nhl_spread_model.get_team_prediction(home, away)
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

if __name__ == "__main__":
    app.run(debug=True)

        
