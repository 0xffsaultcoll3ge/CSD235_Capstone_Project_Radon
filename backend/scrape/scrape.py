import csv
import re
from bs4 import BeautifulSoup
import requests
from datetime import datetime
import pandas as pd
import numpy as np

# NHL team abbreviations (including historical names)
team_abbreviations = {
    "Anaheim Ducks": "ANA", "Arizona Coyotes": "ARI", "Phoenix Coyotes": "PHX", 
    "Boston Bruins": "BOS", "Buffalo Sabres": "BUF", "Calgary Flames": "CGY", 
    "Carolina Hurricanes": "CAR", "Chicago Blackhawks": "CHI", "Colorado Avalanche": "COL", 
    "Columbus Blue Jackets": "CBJ", "Dallas Stars": "DAL", "Detroit Red Wings": "DET", 
    "Edmonton Oilers": "EDM", "Florida Panthers": "FLA", "Los Angeles Kings": "LAK", 
    "Minnesota Wild": "MIN", "Montreal Canadiens": "MTL", "Nashville Predators": "NSH", 
    "New Jersey Devils": "NJD", "New York Islanders": "NYI", "New York Rangers": "NYR", 
    "Ottawa Senators": "OTT", "Philadelphia Flyers": "PHI", "Pittsburgh Penguins": "PIT", 
    "San Jose Sharks": "SJS", "Seattle Kraken": "SEA", "St. Louis Blues": "STL", 
    "Tampa Bay Lightning": "TBL", "Toronto Maple Leafs": "TOR", "Vancouver Canucks": "VAN", 
    "Vegas Golden Knights": "VGK", "Washington Capitals": "WSH", "Winnipeg Jets": "WPG",
    # Additional mappings for historical or alternate names
    "Phoenix": "PHX",  # Phoenix Coyotes (used before 2014)
    "Montreal": "MTL",  # Short form
    "Toronto": "TOR",  # Short form
    "Chicago": "CHI",  # Short form
    "Washington": "WSH",  # Short form
    "Edmonton": "EDM",  # Short form
    "Winnipeg": "WPG",  # Short form
    "Philadelphia": "PHI",  # Short form
    "Detroit": "DET",  # Short form
    "Buffalo": "BUF",  # Short form
    "Colorado": "COL",  # Short form
    "Anaheim": "ANA",  # Short form
    "Pittsburgh": "PIT",  # Short form
    "New Jersey": "NJD",  # Short form
    "Boston": "BOS",  # Short form
    "Tampa Bay": "TBL",  # Short form
    "Calgary": "CGY",  # Short form
    "Los Angeles": "LAK",  # Short form
    "St. Louis": "STL",  # Short form
    "Nashville": "NSH",  # Short form
    "Dallas": "DAL",  # Short form
    "Florida": "FLA",  # Short form
    "Phoenix Coyotes": "PHX",  # Historical name
    "San Jose": "SJS",  # Short form
    "Vancouver": "VAN",  # Short form
    "Columbus": "CBJ",  # Short form
    "NY Rangers": "NYR",  # Short form
    "NY Islanders": "NYI",  # Short form
    "Carolina": "CAR",  # Short form
    "Minnesota": "MIN",  # Short form
    "Ottawa": "OTT",  # Short form
}

# Define all seasons to scrape
seasons = ["2024-2025", "2023-2024", "2022-2023", "2021-2022", "2020-2021", 
           "2019-2020", "2018-2019", "2017-2018", "2016-2017", "2015-2016", 
           "2014-2015", "2013-2014", "2012-2013", "2011-2012", "2010-2011", "2009-2010"]

csv_filename = "./nhl_odds_all_seasons.csv"

with open(csv_filename, mode="w", newline="", encoding="utf-8") as file:
    writer = csv.writer(file)
    writer.writerow(["season", "date", "team1", "team2", "odds1", "oddsX", "odds2"])

    for season in seasons:
        url = f"https://checkbestodds.com/hockey-odds/archive-nhl/{season}" if season != "2024-2025" else "https://checkbestodds.com/hockey-odds/archive-nhl/"

        response = requests.get(url)
        soup = BeautifulSoup(response.text, "html.parser")
        table = soup.find("table", class_="tble")

        if table:
            print(f"Scraping data for {season}...")

            rows = table.find_all("tr")
            current_date = None  # Track the current date

            for row in rows:
                # Check if the row contains a date
                date_tag = row.find("b", class_="dBY")
                if date_tag:
                    raw_date = date_tag.text.strip()  # Extract raw text, e.g., "12 January 2024"
                    try:
                        date_obj = datetime.strptime(raw_date, "%d %B %Y")  # Parse into datetime
                        current_date = date_obj.strftime("%Y%m%d")  # Convert to YYYYMMDD
                    except ValueError:
                        print(f"Skipping row with invalid date format: {raw_date}")
                        continue

                # Check if the row contains a match
                match_tag = row.find("a")
                if match_tag and current_date:
                    match_text = match_tag.text.strip()

                    # Extract teams
                    teams = match_text.split(" - ") if " - " in match_text else []
                    if len(teams) != 2:
                        print(f"Skipping row with unexpected team format: {match_text}")
                        continue

                    team1 = team_abbreviations.get(teams[0].strip(), "Unknown")
                    team2 = team_abbreviations.get(teams[1].strip(), "Unknown")

                    # Extract odds
                    odds_tags = row.find_all("b")
                    if len(odds_tags) < 3:
                        print(f"Skipping row with insufficient odds: {match_text}")
                        continue

                    odds1 = odds_tags[0].text.strip()
                    oddsX = odds_tags[1].text.strip()
                    odds2 = odds_tags[2].text.strip()

                    # Save to CSV
                    writer.writerow([season, current_date, team1, team2, odds1, oddsX, odds2])

print(f"All season data saved to {csv_filename}")

# Load the odds data
odds_df = pd.read_csv("./nhl_odds_all_seasons.csv")

# Load the NHL game data
games_df = pd.read_csv("./all_games_preproc.csv")  # Ensure this dataset contains 'date', 'team', 'opposingTeam', and 'gameId'

# Standardizing column names for consistency
games_df.rename(columns={'playerTeam': 'team1', 'opposingTeam': 'team2', 'gameDate': 'date'}, inplace=True)

# Ensure 'date' is the same format in both dataframes
odds_df['date'] = pd.to_datetime(odds_df['date'], format="%Y%m%d")
games_df['date'] = pd.to_datetime(games_df['date'], format="%Y%m%d")

# Add a 'gameId' column to odds_df with NaN values initially
odds_df['gameId'] = np.nan

# Iterate over the odds dataframe to find matching rows in games_df and set the gameId
for idx, row in odds_df.iterrows():
    match = games_df[
        (games_df['date'] == row['date']) &
        (games_df['team1'] == row['team1']) &
        (games_df['team2'] == row['team2'])
    ]
    
    if not match.empty:
        odds_df.at[idx, 'gameId'] = match.iloc[0]['gameId']

# Save the merged dataset
odds_df.to_csv("./nhl_odds_with_gameId.csv", index=False)

print("Game IDs successfully merged into odds dataset!")