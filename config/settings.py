import os
from dotenv import load_dotenv

class Settings:
    #DB stuff
    NHL_DB_URI = os.getenv("NHL_DB_URI")
    if not NHL_DB_URI:
        raise ValueError("NHL_DB_URI not found in .env file")
    #Table names
    NHL_GAMES_PREPROC = os.getenv("NHL_GAMES_PREPROC")
    NHL_GAMES = os.getenv("NHL_GAMES")
    NHL_ODDS = os.getenv("NHL_ODDS")
    
    #PATHS
    DATA_DIR = os.getenv("DATA_DIR")
