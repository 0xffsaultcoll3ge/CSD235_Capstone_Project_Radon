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

warnings.filterwarnings("ignore")

def calculate_seasonal_ema(df, col):
    ema_column = [np.nan] * len(df)
    if(df[col].dtype != np.float64):
        raise Exception(f"Error occurred calculating seasonal EMA for {col}")
    for i in range(0, len(df)):
        if i == 0:
            ema_column[i] = df[col][:i+1].ewm(span=i + 1, adjust=False).mean().iloc[-1]
            continue
        ema_column[i] = df[col][:i+1].ewm(span=i + 1, adjust=False).mean().iloc[-2]
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
    def apply_seasonal_ema(self, df, groupby_col='season', value_col='score', id_col='gameId'):
        df1 = df.groupby(groupby_col, group_keys=False).apply(self.ema_df).sort_values(by="gameId")
        return pd.concat([df, df1.drop(df.columns, axis=1)], axis=1)

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
                    print(away_row.columns)
                    away_cols = away_row.columns[away_row.columns.str.contains('Against') | (away_row.columns == 'gameId')]
                    merged_row = pd.merge(row.to_frame().T, away_row[away_cols], on='gameId', how='outer')
                    print(merged_row)
                    rows.append(merged_row)

            return pd.concat(rows, ignore_index=False)
        else:
            return None

    def merge_team_dataframes(data_map):
        ret_df = pd.DataFrame()
        if self.sport == "NHL" and self.subject == "teams":
            df_list = list(data_map.values())
            ret_df = pd.concat(df_list, ignore_index=True).sort_values(by="gameId")
            ret_df["winner"] = np.where(merged_df["goalsFor"] > merged_df["goalsAgainst"], 1.0, 0.0)
        return ret_df


if __name__ == "__main__":  
    preproc = Preprocessor("NHL")
    preproc.data_list = preproc.dataframe_list("teams")
    # print(preproc.data_list)
    preproc.data_list = [preproc.clean_dataframe(df) for df in preproc.data_list]
    # print(preproc.data_list)
    preproc.data_list = [preproc.apply_seasonal_ema(df) for df in preproc.data_list]
    print(preproc.data_list)

    data_map = {}
    for df in preproc.data_list:
        team = df["team"].iloc[1]
        # print(team, type(team))
        # print(df["team"], type(df["team"]))
        if "team" in data_map:
            print("DEBUG")
            data_map["team"] = pd.concat([data_map["team"], df], ignore_index=True).sort_values(by="gameId")
        else:
            data_map[team] = df.copy()
    new_data_map = {}
    for k,v in data_map.items():
        new_data_map[k] = preproc.create_team_matches(v, data_map)
        print(new_data_map[k])
    print(merge_team_dataframes(new_data_map))
        
    



# scraper.download_nhl_team_data()


            



            
            
            





    


    



            



            





    
    
            



    
