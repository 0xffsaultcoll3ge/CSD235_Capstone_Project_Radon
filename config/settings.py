# import os
# from dotenv import load_dotenv

# load_dotenv()

# class Settings:
#     #DB stuff
#     NHL_DB_URI = os.getenv("NHL_DB_URI")
#     if not NHL_DB_URI:
#         raise ValueError("NHL_DB_URI not found in .env file")
#     #Table names
#     NHL_GAMES_PREPROC = os.getenv("NHL_GAMES_PREPROC")
#     NHL_GAMES = os.getenv("NHL_GAMES")
#     NHL_ODDS = os.getenv("NHL_ODDS")
    
#     #PATHS
#     DATA_DIR = os.getenv("DATA_DIR")


#     STRIPE_SECRET_KEY = os.getenv("STRIPE_SECRET_KEY")
#     STRIPE_PUBLISHABLE_KEY = os.getenv("STRIPE_PUBLISHABLE_KEY")

#     # STRIPE_PRICE_ID = os.getenv("price_1R6EshICYd8zgaTDu9zHa1a6")