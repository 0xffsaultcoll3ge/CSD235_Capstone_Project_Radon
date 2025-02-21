import xgboost as xgb
import pandas as pd
import sys
sys.path.insert(1, './backend/preprocess')
import elo


class NHLModel:
    def __init__(self, event: str, model_name="xgboost", model_path=None):
        self.event = event.lower()
        self.model_name = model_name
        self.model_path = model_path
        self.model = None
        self.data = None  # Initialize data attribute
        if model_path is not None:
            self.model = self.load_xgb_model(model_path)

    def load_xgb_model(self, model_path: str) -> xgb.Booster:
        try:
            model = xgb.Booster()
            model.load_model(model_path)
            return model
        except Exception as e:
            print(f"Error loading model: {e}")
            return None

    def get_data(self):
        try:
            self.data = pd.read_csv("all_games_preproc.csv")
            return self.data
        except Exception as e:
            print(f"Error loading data: {e}")
            return None

    def preprocess(self, target=None):
        if self.event == "ml":
            assert self.data is not None, "Data not loaded"
            df = self.data

            # Select relevant features
            feature_cols = df.columns[df.columns.str.contains("ema") |
                            df.columns.str.contains('eloExpected') |
                            df.columns.str.contains('daysRest') |
                            df.columns.str.contains('winPercentage')]

            X = df[feature_cols]
            X = (X - X.mean()) / X.std()  # Standardize features
            y = df["winner"]

            return X, y

    def create_match(self, df: pd.DataFrame, home: str, away: str):
        home_df = df[(df["team"] == home) | (df["opposingTeam"] == home)].tail(1)
        away_df = df[(df["team"] == away) | (df["opposingTeam"] == away)].tail(1)

        if home_df.empty or away_df.empty:
            raise ValueError("No data found for the specified teams")

        home_status = "home" if home_df.iloc[0]["team"] == home else "away"
        away_status = "home" if away_df.iloc[0]["team"] == away else "away"

        # Update EMAs
        for col in home_df.columns:
            if 'ema' in col:
                base_col = col.replace("_seasonal_ema", "")
                home_df[col] = home_df[col] * (1 / 3) + (2 / 3) * home_df[base_col]
                away_df[col] = away_df[col] * (1 / 3) + (2 / 3) * away_df[base_col]

        # Update Elo ratings
        self.update_elo_ratings(home_df, away_df, home_status, away_status)

        # Rename columns
        home_df = self.rename_columns(home_df, home_status, is_home=True)
        away_df = self.rename_columns(away_df, away_status, is_home=False)

        # Select relevant columns
        home_df = home_df[home_df.columns[home_df.columns.str.contains("For")]]
        away_df = away_df[away_df.columns[away_df.columns.str.contains("Against")]]

        print(home_df)
        print(away_df)

        ret = pd.concat([home_df.iloc[0], away_df.iloc[0]], axis=0)

        return pd.DataFrame(ret).T

    def update_elo_ratings(self, home_df, away_df, home_status, away_status):
        elo_scorer = elo.Elo(100, 0.01)

        # Set initial Elo ratings
        teams = {
            home_df.iloc[0]["team"]: home_df.iloc[0]["eloFor"],
            home_df.iloc[0]["opposingTeam"]: home_df.iloc[0]["eloAgainst"],
            away_df.iloc[0]["team"]: away_df.iloc[0]["eloFor"],
            away_df.iloc[0]["opposingTeam"]: away_df.iloc[0]["eloAgainst"],
        }
        for team, rating in teams.items():
            elo_scorer[team] = rating

        # Update Elo ratings based on match outcomes
        self.update_elo_for_match(home_df, elo_scorer, home_status)
        self.update_elo_for_match(away_df, elo_scorer, away_status)

        # Update DataFrame with new Elo ratings
        if home_status == "home":
            home_df.at[home_df.index[0], "eloFor"] = elo_scorer[home_df.iloc[0]["team"]]
        else:
            home_df.at[home_df.index[0], "eloAgainst"] = elo_scorer[home_df.iloc[0]["team"]]

        if away_status == "home":
            away_df.at[away_df.index[0], "eloFor"] = elo_scorer[away_df.iloc[0]["team"]]
        else:
            away_df.at[away_df.index[0], "eloAgainst"] = elo_scorer[away_df.iloc[0]["team"]]

    def update_elo_for_match(self, df, elo_scorer, status):
        margin = elo_scorer.get_margin_factor(df.iloc[0]["goalsFor"] - df.iloc[0]["goalsAgainst"])

        if df.iloc[0]["winner"] == 1.0:
            winner = df.iloc[0]["team"]
            loser = df.iloc[0]["opposingTeam"]
        else:
            winner = df.iloc[0]["opposingTeam"]
            loser = df.iloc[0]["team"]

        inflation = elo_scorer.get_inflation_factor(elo_scorer[winner], elo_scorer[loser])
        elo_scorer.update_ratings(winner, loser, 0.01, margin, inflation)

    def rename_columns(self, df, status, is_home=True):
        if is_home:
            if status == "home":
                return df
            else:
                return df.rename(columns=lambda col: col.replace('Against', 'For') if 'Against' in col \
                else col.replace('For', 'Against') if 'For' in col else  col)
        else:
            if status == "away":
                return df
            else:
                return df.rename(columns=lambda col: col.replace('For', 'Against') if 'For' in col \
                else col.replace('Against', 'For') if 'Against' in col else  col)
    def predict(self, X):
        return self.model.predict(X)
    def get_feature_names(self):
        return self.model.feature_names
    def to_dmatrix(self, match_df: pd.DataFrame):
        try:
            features = self.get_feature_names()

            match_df["eloExpectedFor"] = 1 / (1 + 10 ** ((match_df['eloFor'] - match_df['eloAgainst']) / 400))
            match_df['eloExpectedAgainst'] = 1 - match_df['eloExpectedFor']

            return xgb.DMatrix(match_df[features])
        except Exception as e:
            print(e)

    def get_team_prediction(self, home: str, away: str):
        #replace with DB
        try:
            df = pd.read_csv("all_games_preproc.csv")
            match_df = self.create_match(df, home, away)

            dmat = self.to_dmatrix(match_df)

            return self.predict(dmat)
        except Exception as e:
            print(e)
    

def get_expect_result(p1: float, p2: float) -> float:
    exp = (p2 - p1) / 400.0
    return 1 / ((10.0 ** (exp)) + 1)


if __name__ == "__main__":
    nhl_model = NHLModel("ml", model_path="./backend/model/models/XGBoot_57.8%_ML.json")
    df = nhl_model.get_data()

    print(nhl_model.get_team_prediction("PIT", "FLA"))
    print(type(nhl_model.get_team_prediction("PIT", "BOS")))
    print(nhl_model.get_team_prediction("OTT", "NYR"))

    
    

    



                
            









        




        


  