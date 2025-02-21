import pandas as pd

df = pd.read_csv("all_games_preproc.csv")

pd.to_sql("games", self.conn, if_exist='replace', index_label="gameId")

scraper = Scraper("NHL")
preprocessor = Preprocessor("NHL")
scraper.download_nhl_team_data()
preprocessor.update_csv("all_games_preproc.csv")

query = "SELECT MAX(gameId) FROM games_preproc"

df[df["gameId"].to_numeric() > int(_gameId)]