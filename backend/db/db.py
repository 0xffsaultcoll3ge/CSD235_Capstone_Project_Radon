import sqlite3
import pandas as pd
class Database:
    def __init__(self, db_name="nhl_predictions.db"):
        """Initialize the database connection."""
        self.conn = sqlite3.connect(db_name)
        self.cursor = self.conn.cursor()
        self.create_tables()

    def create_tables(self):
        #implement logic
        self.conn.commit()

    def fetch_historical_team_data(self, team_name):
        query = #to-do
        df = pd.read_sql_query(query, self.conn, params=(team_name, team_name))
        return df

    def fetch_last_game_by_team(self, team_name):
        """Fetch the last game data for a given team."""
        query = #to-do
        return self.cursor.fetchone()
    def fetch_games_played_in_season(self, team_name):

    def fetch_odds(self, team1, team2):
        """Fetch odds for a specific game matchup."""
        query = """
            SELECT team1_odds, team2_odds FROM games 
            WHERE team1 = ? AND team2 = ? 
            ORDER BY game_date DESC 
            LIMIT 1
        """
        self.cursor.execute(query, (team1, team2))
        return self.cursor.fetchone()

    def update_team_database(self, df):
        #implement logic
        self.conn.commit()

    def close(self):
        self.conn.close()
