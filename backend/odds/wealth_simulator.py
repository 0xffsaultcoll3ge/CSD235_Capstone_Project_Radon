import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
import json
import os
import sys
import xgboost as xgb
sys.path.insert(1,'backend/db')
sys.path.insert(2, 'backend/model')
from db import NHLPipeline
from nhl_model import *
from datetime import datetime

nhl_pipeline = NHLPipeline()
odds_df = pd.read_csv("./backend/data/NHL/odds/nhl_game_scores_1g_20240930.csv")

print(odds_df)
odds_df["season"] = odds_df["season"].str[:4]
odds_df["date"] = odds_df["date"].str.replace('-','')

print(odds_df)
MODEL_DIR = "./backend/model/models/"

spread_vals = [-1.5, 1.5]
ou_vals = [5.0, 5.5, 6.0, 6.5]


def load_model(path: str):
    try:
        model = xgb.Booster()
        model.load_model(path)
        return model
    except Exception as e:
        print(e)

nhl_ou_model = lambda ou: NHLModel("ou", model_path = best_model_path("ou", MODEL_DIR, ou))
nhl_spread_model = lambda spread: NHLModel("spread", model_path = best_model_path("spread", MODEL_DIR, spread))

ml_model = NHLModel("ml", model_path = "./backend/model/models/ml/XGBoost_tuned_58.46.json")
ou_models = {k: (lambda k: nhl_ou_model(k))(k) for k in ou_vals}
spread_models = {k: (lambda k: nhl_spread_model(k))(k) for k in spread_vals}

# nhl_pipeline.write_to_table(odds_df, "odds")

# query = f'''
# SELECT * FROM games_preproc gp
# JOIN odds o
# ON gp.season = o.season
#     AND gp.gameDate = o.date 
# '''
# merged = nhl_pipeline.fetch_query(query)

# print(merged)


def simulate(game_df: pd.DataFrame, odds_df: pd.DataFrame, frac = 1/8, event="ml", value=None):
    bankroll = 1000
    dates = []
    totals = []
    if event == "ml":
        for i in range(len(odds_df)):
            row = odds_df.iloc[i]
            game_row = game_df[
                (game_df['season'].astype(str) == row['season']) &
                (game_df['gameDate'].astype(str) == row['date']) &
                (game_df['team'] == row['home_team']) &
                (game_df['opposingTeam'] == row['away_team'])
            ]

            if game_row.empty:
                continue
            # ou = row["over_under"]
            # ou_model = ou_models[float(ou)]

            match_df = ml_model.create_match_by_date(game_df, row["home_team"], row["away_team"], int(row["date"]))
            # ou_dmat = ou_model.convert_to_dmatrix(match_df)

            # ou_pred = ou_model.predict(ou_dmat) 

            ml_dmat = ml_model.convert_to_dmatrix(match_df)

            ml_pred = ml_model.predict(ml_dmat)

            home_ml_odds = round(american_to_decimal(row["home_money_line"]), 2)
            away_ml_odds = round(american_to_decimal(row["away_money_line"]), 2)

            total =  bankroll if len(totals) == 0 else totals[-1]

            kelly_home_ml = frac * kelly_criterion_result(total, ml_pred[: , 1], home_ml_odds - 1)
            kelly_away_ml = frac * kelly_criterion_result(total, 1 - ml_pred[: , 0], away_ml_odds - 1)

            if kelly_home_ml > 0:
                add = kelly_home_ml * (home_ml_odds - 1) if (game_row["winner"] == 1.0).any() else -kelly_home_ml
                totals.append(total + add)
                dates.append(game_row["gameDate"].iloc[0])
            if kelly_away_ml > 0:
                add = kelly_away_ml * (away_ml_odds - 1) if (game_row["winner"] == 0.0).any() else -kelly_away_ml
                totals.append(total + add)
                dates.append(game_row["gameDate"].iloc[0])
            # print(dates[-1], totals[-1], type(totals[-1]))
            # print(len(dates))
            # print(len(totals))
            # print(totals)
        dates = [datetime.strptime(str(date), "%Y%m%d") for date in dates]
        return dates,[item.item() if isinstance(item, np.ndarray) else item for item in totals]

    elif event == "ou":
        # ou_model = ou_models[value]
        return None


game_df = pd.read_csv("all_games_preproc.csv")

dates, totals = simulate(game_df, odds_df[odds_df["season"].astype(int) == 2019])
print(totals)
plt.figure(figsize=(10, 5))
plt.plot(dates, totals, marker='o', linestyle='-', color='blue')
plt.xlabel('Time')
plt.ylabel('Wealth')

plt.show()
