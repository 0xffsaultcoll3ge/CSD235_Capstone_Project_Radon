from flask import *
from sqlalchemy import create_engine, MetaData
from sqlalchemy.orm import sessionmaker
import pandas as pd
import sys
sys.path.insert(1, './backend/model')
from nhl import NHLModel

app = Flask(__name__)
nhl_ml_model = NHLModel("ml", model_path="./backend/model/models/XGBoot_58.0%_ML.json")

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
        team1 = request.args.get('home')
        team2 = request.args.get('away')

        q1 = f"""
        SELECT * FROM games_preproc WHERE (team = '{home}' OR opposingTeam = '{away}') ORDER BY gameId DESC LIMIT 1
        """
        q2 = f"""
        SELECT * FROM games_preproc WHERE (team = '{home}' OR opposingTeam = '{away}') ORDER BY gameId DESC LIMIT 1
        """

        df1 = pd.read_sql(q1, engine)
        df2 = pd.read_sql(q2, engine)
        _match = create_match(df1, df2) #asssume this method exists

        model = xgb.Booster()
        model.load_model(ou_model_path)

        dmatrix = xgb.DMatrix(_match)

        predictions = model.predict(dmatrix)
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

        