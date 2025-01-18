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

#df = pd.read_csv('all_teams.csv')
#print(df.columns)

def get_float_features(df):
    feats = []
    for feat in df.columns.tolist():
        if(df[feat].dtype == np.float64):
            feats.append(feat)
    return feats
# print(df['season'])
# print(get_float_features(df))
# print(len(get_float_features(df)))

# print(df['playContinuedOutsideZoneAgainst'].ewm(span=4, adjust=False).mean())

def get_team_names():
    team_names = []
    try:
        with open('./team_files', 'r') as f:
            team_names = [_.strip('\n') for _ in f.readlines()]
    except:
        pass
    return team_names
def download_file(url, gametype, path=None):
    if path == None:
        path = "./data/teams/{0}/{1}".format(gametype, url.split("/")[-1])
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
    team_names = get_team_names()
##    print(team_names)
    regular_base_url = "https://moneypuck.com/moneypuck/playerData/careers/gameByGame/regular/teams/"
    playoff_base_url = "https://moneypuck.com/moneypuck/playerData/careers/gameByGame/playoffs/teams/"
    for team in tqdm(team_names):
        team = team.strip()
        url1 = regular_base_url + "{0}.csv".format(team)
        url2 = playoff_base_url + "{0}.csv".format(team)
        t1 = threading.Thread(target=download_file, args=((url1, "regular")))
        t2 = threading.Thread(target=download_file, args=((url2, "playoff")))
        t1.start()
        t2.start()
        t1.join()
        t2.join()
def get_team_data_map(path='./data/teams'):
    df_map = {}
    # try:
    for d in os.listdir(path):
        files = os.path.join(path, d)
        for file in os.listdir(files):
            file_path = os.path.join(files, file)
            df = pd.read_csv(file_path)
            team = str(file).strip('.csv')
            if team in df_map:
                df_map[team] = pd.concat([df, df_map[team]], ignore_index=True).sort_values(by="gameId")
            else:
                df_map[team] = df
    # except:
        # print("Error retrieving team data")
    return df_map
def get_team_data_map_ema(path='./data/teams'):
    df_map = get_team_data_map()
    if df_map == {}:
        print("Error: An error occurred while trying to create team dataframe map")
    for k,v in df_map.items():
        v["winner"] = np.where(v["goalsFor"] > v["goalsAgainst"], 1.0, 0.0)
        v = v[v["situation"] == "all"]
        df = v.loc[:, ~v.columns.str.contains("Against")]
        df_map[k] = df.groupby("season", group_keys=False).apply(ewm_float64_features).sort_values("gameId")
    return df_map

def combine_games():
    df_map=get_team_data_map_ema()
    if df_map == {}:
        print("Error: An error occurred while trying to create team dataframe map with EMAs")
    for k,v in df_map.items():
        for i in range(len(v)):
            currGameId = v.iloc[i]["gameId"]
            oppTeam = v.iloc[i]["opposingTeam"]
            oppDF = df_map[oppTeam]
            oppDF = oppDF[oppDF["gameId"] == currGameId]
        
    

def merge_game_data(path='./data/teams'):
    df_list = []
    for d in os.listdir(path):
        files = os.path.join(path, d)
        for file in os.listdir(files):
            file_path = os.path.join(files, file)
            df = pd.read_csv(file_path)
            df_list.append(df)
    df_list = [df[df["home_or_away"] == "HOME"] for df in df_list]
    merged_df = pd.concat(df_list, ignore_index=True).sort_values(by="gameId")
    merged_df = merged_df[merged_df["situation"] == "all"]
    merged_df["winner"] = np.where(merged_df["goalsFor"] > merged_df["goalsAgainst"], 1.0, 0.0)
    merged_df['gameDate'] = pd.to_datetime(df['gameDate'])

    return merged_df

def calculate_ema(df, col, window_size):
    ema_column = [np.nan] * len(df)

    for i in range(window_size - 1, len(df)):
        ema_val = df[col][:i+1].ewm(span=window_size, adjust=False).mean().iloc[-1]
        ema_column[i] = ema_val
    return ema_column

def calculate_ema_seasonsal(df, col):
    ema_column = [np.nan] * len(df)
    for i in range(0, len(df)):
        ema_val = df[col][:i+1].ewm(span=i + 1, adjust=False).mean().iloc[-1]
        ema_column[i] = ema_val
    return ema_column



def ewm_float64_features(df):
    ewm_df = df
    float_features = get_float_features(df)
    for feat in float_features:
        if(feat == "winner"):
            continue
        three = f"{feat}_three_game_ema"
        ten = f"{feat}_ten_game_ema"
        season = f"{feat}_seasonal_ema"

        # ewm_df[three] = calculate_ema(df, feat, 3)
        # ewm_df[ten] = calculate_ema(df, feat, 10)
        ewm_df[season] = calculate_ema_seasonsal(df, feat)
        
    return ewm_df

for k,v in combine_team_data().items():
    print(f"{k}:{v}")
# data = merge_game_data()
# features = list(filter(lambda x: x != "winner", get_float_features(data)))

# df1 = data.sort_values(by="gameId")
# df2 = data.groupby(['team', 'season'], group_keys=False).apply(ewm_float64_features).sort_values(by="gameId") 
# ewm_data = pd.concat([df1, df2.drop(columns=df1.columns)], axis=1).drop_duplicates(ignore_index=True)
# ewm_data.to_csv('all_teams_1.csv', index=True)

# df = pd.read_csv("all_teams_1.csv")

# print(df.shape)
# float_columns = df.select_dtypes(include=['float64'])
# corr_map = {}
# for i,v in float_columns.corr()["winner"].items():
#     corr_map[i] = v

# sort_map = dict(sorted(corr_map.items(), key=lambda item: math.fabs(item[1]), reverse=True))

# for i,v in sort_map.items():
#     print(f"{i}:\t{v}")
# print(corr)



