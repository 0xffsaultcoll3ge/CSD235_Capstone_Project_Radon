mport sys
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

def apply_seasonal_ema_to_col(df, col):
    ema_column = [np.nan] * len(df)
    for i in range(0, len(df)):
        ema_val = df[col][:i+1].ewm(span=i + 1, adjust=False).mean().iloc[-1]
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
def download_nhl_team_data(regular=True, playoff=True):
    try:
        nhl_teams = nhl_team_names()
        regular_base_url = "https://moneypuck.com/moneypuck/playerData/careers/gameByGame/regular/teams/"
        playoff_base_url = "https://moneypuck.com/moneypuck/playerData/careers/gameByGame/playoffs/teams/"

        for team in tqdm(team_names):
            t1, t2 = (None, None)
            team = team.strip()
            url1 = regular_base_url + "{0}.csv".format(team)
            url2 = playoff_base_url + "{0}.csv".format(team)

            if regular:
                t1 = threading.Thread(target=download_file, args=((url1, "regular")))
                t1.start()
            if playoff:
                t2 = threading.Thread(target=download_file, args=((url2, "playoff")))
                t2.start()
            if(t1 != None) t1.join()
            if(t2 != None) t2.join()
        except:
            return Error("Error occurred downloading team data")

        return True
class Preprocessor:
    def __init__(self, sport):
        self.sport = sport
        self.sports = ["NHL"]
        if self.sport not in self.sports:
            raise Exception("Error during instantiation, invalid sport: {0}".format(self.sport))
        self.data = None
        self.ema_data = None
        self.preproc_data = None
        self.data_path = "./data"
    def download_file(url, gametype, path=None):
        if path == None:
            path = "./data/{0}/teams/{1}/{2}".format(self.sport, gametype, url.split("/")[-1])
        try:
            r = requests.get(url)
            content = requests.get(url, stream = True)

            with open(path, "wb") as f:
                for chunk in r.iter_content(chunk_size=1024):
                    if chunk:
                        f.write(chunk)
        except:
            print("Downloading {0} failed...".foramt(url))
    def download_team_data():
        if sport_in_list(self.sport):
            return False
        else if self.sport == "NHL":
            ret = download_nhl_team_data()
            if(ret):
                return True
            else:
                return ret
    def dataframe_list(path=self.data_path, sport=self.sport, subject):
        df_list = []
        if sport == "NHL":
            sport_dir = os.path.join(path, sport, subject)
            for d in os.listdir(sport_dir):
                files = os.path.join(path, d)
                for file in os.listdir(files):
                    file_path = os.path.join(files, file)
                    df = pd.read_csv(file_path)
                    df_list.append(df)
        return df_list   



    
    
            



    