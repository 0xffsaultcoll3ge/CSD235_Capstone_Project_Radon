import sys
import sqlite3
import math
import pandas as pd
import numpy as np
import os
import requests
import threading
import math
from tqdm import tqdm
import warnings
import elo as elo
from skopt import gp_minimize
from skopt.space import Real
from skopt.utils import use_named_args

warnings.filterwarnings("ignore")

def calculate_seasonal_ema(df, col, shift=1):
    ema_column = [np.nan] * len(df)
    if(df[col].dtype != np.float64):
        raise Exception(f"Error occurred calculating seasonal EMA for {col}")
    ema_vals = df[col].ewm(span=5.5, adjust=False).mean()
    shifted_vals = ema_vals.shift()
    for i in range(0, len(df)):
        ema_val = shifted_vals.iloc[i]
        ema_column[i] = ema_val
    return ema_column


def get_float_features(df):
    feats = []
    for feat in df.columns.tolist():
        if(df[feat].dtype == np.float64):
            feats.append(feat)
    return feats
def nhl_team_names():
    team_names = []
    try:
        with open('./team_files', 'r') as f:
            team_names = [l.strip('\n') for l in f.readlines()]
    except:
        return Error("Failed to read NHL team file")
    return team_names
def download_file(url, sport, subject, gametype, path=None):
    if path == None:
        _dir = "./data/{0}/{1}/{2}".format(sport, subject, gametype)
        path = "./data/{0}/{1}/{2}/{3}".format(sport, subject, gametype, url.split("/")[-1])
    try:
        r = requests.get(url)
        if r.status_code == 404:
            return
        content = requests.get(url, stream = True)
        if not os.path.exists(_dir):
            os.makedirs(_dir)
        with open(path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024):
                if chunk:
                    f.write(chunk)
    except:
        print("Downloading {0} failed...".foramt(url))

class Scraper:
    def __init__(self, sport):
        self.sport = sport
        self.sports = ["NHL"]
        if self.sport not in self.sports:
            raise Exception("Error during instantiation, invalid sport: {0}".format(self.sport))

    def download_nhl_team_data(regular=True, playoff=True):
        try:
            nhl_teams = nhl_team_names()
            regular_base_url = "https://moneypuck.com/moneypuck/playerData/careers/gameByGame/regular/teams/"
            playoff_base_url = "https://moneypuck.com/moneypuck/playerData/careers/gameByGame/playoffs/teams/"

            for team in tqdm(nhl_teams):
                t1, t2 = (None, None)
                team = team.strip()
                url1 = regular_base_url + "{0}.csv".format(team)
                url2 = playoff_base_url + "{0}.csv".format(team)

                if regular:
                    t1 = threading.Thread(target=download_file, args=((url1, "NHL", "teams", "regular")))
                    t1.start()
                if playoff:
                    t2 = threading.Thread(target=download_file, args=((url2, "NHL", "teams", "playoff")))
                    t2.start()
                if t1 != None:
                    t1.join()
                if t2 != None:
                    t2.join()
        except:
            raise Exception("Error occurred downloading team data ...")

            return True
class Preprocessor:
    def __init__(self, sport):
        self.sport = sport
        self.sports = ["NHL"]
        if self.sport not in self.sports:
            raise Exception("Error during instantiation, invalid sport: {0}".format(sport))
        self.data = None
        self.data_list = []
        self.ema_data = None
        self.preproc_data = None
        self.data_path = "./data"
        self.subject = "teams"
        # self.scraper = Scraper(sport)
    def normalize(self, df):
        assert(~df.isnumeric())
        return (df -df.mean())/df.std()
    def dataframe_list(self, subject):
        path = self.data_path
        df_list = []
        if self.sport == "NHL":
            sport_dir = os.path.join(path, self.sport, subject)
            for d in os.listdir(sport_dir):
                files = os.path.join(sport_dir, d)
                for file in os.listdir(files):
                    file_path = os.path.join(files, file)
                    df = pd.read_csv(file_path)
                    df_list.append(df)
        return df_list
    def clean_dataframe(self, df):
        if self.sport == "NHL" and self.subject == "teams":
            df["winner"] = np.where(df["goalsFor"] > df["goalsAgainst"], 1.0, 0.0)
            df = df[df["situation"] == "all"]
            df = df.loc[:, ~df.columns.str.contains("Against")]
            return df
        else:
            return None

    def ema_df(self, df):
        if self.sport == "NHL" and self.subject == "teams":
            for col in get_float_features(df):
                df[f"{col}_seasonal_ema"] = calculate_seasonal_ema(df, col)
            return df
        else:
            return None
    def apply_win_percentage(self, df, groupby_col='season'):
        df1 = df.groupby(groupby_col, group_keys=False) \
        .apply(lambda x: x[x["winner"] == 1.0] / (x[x["winner"] == 0.0] + x[x["winner"] == 1.0])) \
        .sort_values(by="gameId")

        return pd.concat([df, df1.drop(df.columns, axis=1)], axis=1).sort_values(by="gameId")
    def apply_seasonal_ema(self, df, groupby_col='season', value_col='score', id_col='gameId'):
        df1 = df.groupby(groupby_col, group_keys=False).apply(lambda x: self.ema_df(x.sort_values(by="gameId"))).sort_values(by="gameId")

        return pd.concat([df, df1.drop(df.columns, axis=1)], axis=1).sort_values(by='gameId')

    def get_prev_elo(self, df, i, team):
        df = df.iloc[:i]
        filtered = df[df["opposingTeam"] == team or df["awayTeam"] == team]
        if filtered.empty:
            return -1
        else:
            return filtered.index[-1]
        
        
    def apply_elo_rating(self, df, K, decay, optimized=False):
        errors = []
        elo_scorer = elo.Elo(K, decay)
        team_set = set(df["opposingTeam"].unique().tolist())

        for team in team_set:
            elo_scorer.add_team(team)
        elo_home = np.zeros((len(df,)))
        elo_away = np.zeros((len(df,)))
        elo_expected_home = np.zeros((len(df,)))
        for rowidx, row in enumerate(df.itertuples()):
            if(rowidx == 0):
                elo_scorer.set_season(row.season, team_set)
            home = row.playerTeam
            away = row.opposingTeam
            elo_home[rowidx] = elo_scorer[home]
            elo_away[rowidx] = elo_scorer[away]
            season = elo_scorer.get_season()
            if season < row.season:
                elo_scorer.set_season(row.season, team_set)
            margin = elo_scorer.get_margin_factor(row.goalsFor - row.goalsAgainst)
            elo_expected_home[rowidx] = elo_scorer.get_expect_result(elo_home[rowidx], elo_away[rowidx])
            if row.winner == 1.0:
                inflation = elo_scorer.get_inflation_factor(elo_home[rowidx], elo_away[rowidx])
                elo_scorer.update_ratings(home, away, decay, margin, inflation)
            else:
                inflation = elo_scorer.get_inflation_factor(elo_away[rowidx], elo_home[rowidx])
                elo_scorer.update_ratings(away, home, decay, margin, inflation)
                
        df.loc[:, 'eloFor'] = elo_home
        df.loc[:, 'eloAgainst'] = elo_away
        df.loc[:, 'eloExpectedFor'] = elo_expected_home

        if not optimized:

            space = [Real(10, 100, name="K"), Real(0, 400, name="decay")]

            @use_named_args(space)
            def objective(K, decay):
                return self.calculate_elo_error(df.copy(), K, decay)

            result = gp_minimize(objective, space, n_calls=30, random_state=42)

            best_K, best_decay = result.x

            print(result.x)

            return self.apply_elo_rating(df, best_K, best_decay, optimized=True)

        else:
            return df
    def calculate_elo_error(self, df, K, decay):
        errors = []
        elo_scorer = elo.Elo(K, decay)
        team_set = set(df["opposingTeam"].unique().tolist())

        for team in team_set:
            elo_scorer.add_team(team)
        elo_home = np.zeros((len(df,)))
        elo_away = np.zeros((len(df,)))
        elo_expected_home = np.zeros((len(df,)))

        for rowidx, row in enumerate(df.itertuples()):
            if(rowidx == 0):
                elo_scorer.set_season(row.season, team_set)

            home = row.playerTeam
            away = row.opposingTeam

            elo_home[rowidx] = elo_scorer[home]
            elo_away[rowidx] = elo_scorer[away]
            season = elo_scorer.get_season()
            if season < row.season:
                elo_scorer.set_season(row.season, team_set)
            margin = elo_scorer.get_margin_factor(row.goalsFor - row.goalsAgainst)
            elo_expected_home[rowidx] = elo_scorer.get_expect_result(elo_home[rowidx], elo_away[rowidx])
            if row.winner == 1.0:
                inflation = elo_scorer.get_inflation_factor(elo_home[rowidx], elo_away[rowidx])
                elo_scorer.update_ratings(home, away, decay, margin, inflation)
            else:
                inflation = elo_scorer.get_inflation_factor(elo_away[rowidx], elo_home[rowidx])
                elo_scorer.update_ratings(away, home, decay, margin, inflation)
            expected = elo_expected_home[rowidx]
            actual = row.winner
            err = -(actual*np.log(expected) + (1 - actual)*np.log(1 - expected))
            errors.append(err)
        return np.mean(err)

    def create_team_matches(self, df, team_data_map):
        if self.sport == "NHL" and self.subject == "teams":
            rows = []
            for i, row in df.iterrows():
                gameId = row["gameId"]
                if row["home_or_away"] == "AWAY":
                    continue
                away_team = row["opposingTeam"]
                away_df = team_data_map[away_team]
                away_row = away_df[away_df["gameId"] == gameId]

                if not away_row.empty:
                    away_row = away_row.rename(columns=lambda col: col.replace('For', 'Against') if 'For' in col else col) \
                   .rename(columns=lambda col: col.replace('Percentage', 'Percentage_Against') if 'Percentage' in col else col) \
                   .rename(columns=lambda col: col.replace('iceTime', 'iceTime_Against') if 'iceTime' in col else col)
                    # print(away_row.columns)
                    away_cols = away_row.columns[away_row.columns.str.contains('Against') | (away_row.columns == 'gameId')]
                    merged_row = pd.merge(row.to_frame().T, away_row[away_cols], on='gameId', how='outer')
                    # print(merged_row)
                    rows.append(merged_row)

            ret = pd.concat(rows, ignore_index=False)
            return ret
        else:
            return None

    def merge_team_dataframes(self, data_map):
        ret_df = pd.DataFrame()
        if self.sport == "NHL" and self.subject == "teams":
            df_list = list(data_map.values())
            ret_df = pd.concat(df_list, ignore_index=True).sort_values(by="gameId")
            # ret_df["winner"] = np.where(ret_df["goalsFor"] > ret_df["goalsAgainst"], 1.0, 0.0)
            # ret_df = self.apply_elo_rating(ret_df, K=32, decay=0.01)
        return ret_df.sort_values(by="gameId").set_index("gameId")

if __name__ == "__main__":  
    preproc = Preprocessor("NHL")
    scraper = Scraper("NHL")
    # scraper.download_nhl_team_data()
    preproc.data_list = preproc.dataframe_list("teams")
    # print(preproc.data_list)
    preproc.data_list = [preproc.clean_dataframe(df) for df in preproc.data_list]
    # print(preproc.data_list)
    # preproc.data_list = [preproc.apply_seasonal_ema(df) for df in preproc.data_list]
    # print(preproc.data_list)

    data_map = {}
    for df in preproc.data_list:
        team = df["team"].iloc[1]
        # print(team, type(team))
        # print(df["team"], type(df["team"]))
        if team in data_map:
            print("Merging")
            data_map[team] = pd.concat([data_map[team], df], ignore_index=True).sort_values(by="gameId")
        else:
            print("Not merging")
            data_map[team] = df.copy()
    new_data_map = {}
    for k,v in data_map.items():
        new_data_map[k] = preproc.apply_seasonal_ema(v)
        new_data_map[k] = preproc.apply_win_percentage(new_data_map[k])
    data_map = new_data_map.copy()
    for k, v in new_data_map.items():
        new_data_map[k] = preproc.create_team_matches(new_data_map[k], data_map)
        # print(new_data_map[k])
    # print(merge_team_dataframes(new_data_map))
    team_df = preproc.merge_team_dataframes(new_data_map)
    print(team_df)
    team_df.to_csv("all_games_preproc.csv")
    team_df = pd.read_csv("all_games_preproc.csv")
    team_df = team_df.loc[:, ~team_df.columns.str.contains('^Unnamed')]
    team_df = preproc.apply_elo_rating(team_df, K=32, decay=0.01)
    team_df.to_csv("all_games_preproc.csv")
    team_df = pd.read_csv("all_games_preproc.csv")
    # def func(team_df):
    #     for i in range(len(team_df)):
    #         print(f"XGP: {team_df.iloc[i]["xGoalsPercentage"]} \t XGP_SA: {team_df.iloc[i]["xGoalsPercentage_seasonal_ema"]}")
    # team_df.groupby(["team", "season"], group_keys=False).apply(func)
    print(team_df.loc[:, ["winner", "goalsFor_seasonal_ema"]].corr())
    print(team_df.loc[:, ["winner", "eloExpectedFor"]].corr())
    print(team_df.loc[:, "eloFor"].max())
    # print(team_df.loc[:, ["winner", "eloExpectedAgainst"]].corr())



# scraper.download_nhl_team_data()


            



            
            
            





    


    



            



            





    
    
            



    
