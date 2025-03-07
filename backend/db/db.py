import sqlite3
import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
import os
# from dotenv import load_dotenv

def get_table_csv(table_name, csv_map, update=False) -> pd.DataFrame:
    fpath = csv_map[table_name]
    if table_name == "games_preproc" and update:
        scraper = Scraper("NHL")
        preproc = Preprocesser("NHL")
        scraper.download_nhl_team_data()
        preproc.update_csv(fpath)
    return pd.read_csv(fpath)
def download_file(url, subject, gametype, path=None):
    if path == None:
        _dir = "./backend/data/NHL/{1}/{2}".format(subject, gametype)
        path = "./backend/data/NHL/{1}/{2}/{3}".format(subject, gametype, url.split("/")[-1])
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
        print("Downloading {0} failed...".format(url))
class NHLPipeline:
    def __init__(self, db_uri="sqlite:///nhl.db"):
        """Initialize the database connection."""
        self.engine = create_engine(db_uri)
        Session = sessionmaker(bind=self.engine)
        self.session = Session()
        # self.preprocessor = Preprocessor("NHL")
        # self.scraper = Scraper("NHL")
    def download_team_csv(self, team_name: str, game_type="regular", path=None):
        regular_base_url = f"https://moneypuck.com/moneypuck/playerData/careers/gameByGame/{game_type}/teams/{team_name}.csv"
        playoff_base_url = f"https://moneypuck.com/moneypuck/playerData/careers/gameByGame/{game_type}/teams/{team_name}.csv"

        try:
            if game_type == "regular":
                download_file(regular_base_url,"regular", path=path)
            else:
                download_file(regular_base_url,"playoff", path=path)
        except Exception as e:
            print(e)

    def update_team_table(self, team_name: str, table_name: str):
        regular_fpath = self.download_team_csv(team_name, "regular")
        playoff_fpath = self.download_team_csv(team_name, "playoff")

        reg_df = pd.read_csv(regular_fpath)
        playoff_df = pd.read_csv(playoff_fpath)
        
        try:
            df = pd.concat([reg_df, playoff_df]).sort_values(by="gameId").set_index("gameId")
            df.to_sql(table_name, con=self.engine, if_exists="append", index=False)
        except Exception as e:
            print(e)
    def update_team_db(self, team_table_map):
        for team,table in team_table_map.items():
            try:
                self.update_team_table(team, table)
            except Exception as e:
                print(f"An error occurred while trying to update table {table}: {ee}")

        
    def write_to_table(self, df: pd.DataFrame, table_name: str):
        try:
            df.to_sql(table_name, con=self.engine, if_exists="append", index=False)
        except Exception as e:
            print(e)
    def fetch_all(self, table_name):
        try:
            query = f'''
            SELECT * FROM {table_name}
            '''
            df = pd.read_sql_query(query, con=self.engine)
            return df
        except Exception as e:
            print(e)
    def fetch_all_team_matches(self, team_name):
        query = f'''
        SELECT * FROM games_preproc WHERE
        (team = '{team_name}' OR opposingTeam = '{team_name}')
        '''
        df = pd.read_sql_query(query, con=self.engine)
        return df
    def fetch_all_team_games(self, team_name):
        query = f'''
        SELECT * FROM games WHERE
        (team = '{team_name}' OR opposingTeam = '{team_name}')
        '''
        df = pd.read_sql_query(query, con=self.engine)
        return df

    def fetch_last_game_by_team(self, team_name):
        """Fetch the last game data for a given team."""
        query = ""
        return self.cursor.fetchone()
    # def fetch_games_played_in_season(self, team_name):

    # def fetch_odds(self, team1, team2):
    #     """Fetch odds for a specific game matchup."""
    #     query = """
    #         SELECT team1_odds, team2_odds FROM games 
    #         WHERE team1 = ? AND team2 = ? 
    #         ORDER BY game_date DESC 
    #         LIMIT 1
    #     """
    #     self.cursor.execute(query, (team1, team2))
    #     return self.cursor.fetchone()


    # def close(self):
    #     self.conn.close()

if __name__ == "__main__":
    db = NHLPipeline()
    df = pd.read_csv("all_games_preproc.csv")
    df = db.write_to_table(df, "games_preproc")
    print(df)
