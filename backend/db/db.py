import sqlite3
import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session

table_to_csv = {
    "games_preproc": "all_games_preproc.csv"
}

def get_table_csv(table_name, csv_map, update=False) -> pd.DataFrame:
    fpath = csv_map[table_name]
    if table_name == "games_preproc" and update:
        scraper = Scraper("NHL")
        preproc = Preprocesser("NHL")
        scraper.download_nhl_team_data()
        preproc.update_csv(fpath)
    return pd.read_csv(fpath)
    


class Database:
    def __init__(self, db_uri="sqlite:///nhl.db"):
        """Initialize the database connection."""
        self.engine = create_engine(db_uri)
        Session = sessionmaker(bind=self.engine)
        self.session = Session()

    def write_to_table(self, table_name):
        try:
            df = update_csv(table_name, table_to_csv)
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
    db = Database()
    df = db.fetch_all_team_matches("OTT")
    print(df)