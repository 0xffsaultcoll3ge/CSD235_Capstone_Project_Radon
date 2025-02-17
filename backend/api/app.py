from flask import *
from sqlalchemy import create_engine, MetaData
from sqlalchemy.orm import sessionmaker
import pandas as pd

app = Flask(__name__)

# DATABASE_URL = "sqlite://nhl.db"
# engine = create_engine(DATABASE_URL)

# Session = sessionmaker(bind=engine)
# session = Session()

@app.route('/api/nhl/ml/predict', methods=['GET'])
def get_predictions_ml():
    try: 
        team1 = request.args.get('team1')
        team2 = request.args.get('team2')

        q1 = f"""
        SELECT * FROM game_preproc WHERE (team = '{team1}' OR opposingTeam = '{team1}') ORDER BY 'gameId' DESC LIMIT 1
        """
        q2 = f"""
        SELECT * FROM game_preproc WHERE (team = '{team2}' OR opposingTeam = '{team2}') ORDER BY 'gameId' DESC LIMIT 1
        """

        df1 = pd.read_sql(q1, engine)
        df2 = pd.read_sql(q2, engine)
        _match = create_match(df1, df2) #asssume this method exists

        model = xgb.Booster()
        model.load_model(ml_model_path)

        dmatrix = xgb.DMatrix(_match)

        predictions = model.predict(dmatrix)
    except Exception as e:
        return jsonify({"error":e}), 500

    return jsonify({"predictions":predictions.tolist()}), 200


@app.route('/api/nhl/ml/predict', methods=['GET'])
def get_predictions_ou():
    try: 
        team1 = request.args.get('team1')
        team2 = request.args.get('team2')

        q1 = f"""
        SELECT * FROM games_preproc WHERE (team = '{team1}' OR opposingTeam = '{team1}') ORDER BY gameId DESC LIMIT 1
        """
        q2 = f"""
        SELECT * FROM games_preproc WHERE (team = '{team2}' OR opposingTeam = '{team2}') ORDER BY gameId DESC LIMIT 1
        """

        df1 = pd.read_sql(q1, engine)
        df2 = pd.read_sql(q2, engine)
        _match = create_match(df1, df2) #asssume this method exists

        model = xgb.Booster()
        model.load_model(ou_model_path)

        dmatrix = xgb.DMatrix(_match)

        predictions = model.predict(dmatrix)
    except Exception as e:
        return jsonify({"error":e}), 500

    return jsonify({"predictions":predictions.tolist()}), 200

@app.route('/api/nhl/team/{id}')
def get_team_data():
    team = dbLayer.get_team_by_id(id)
    try:
        q = "SELECT * FROM games WHERE team = '{team}'"
        df = pd.read_sql(q1, engine)

        return df.to_json()
    except Exception as e:
        return jsonify({"error": e})

if __name__ == "__main__":
    app.run(debug=True)

        