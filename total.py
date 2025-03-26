import pandas as pd

df = pd.read_csv("./backend/data/NHL/teams/regular/ARI.csv")

ties = df[(df["goalsFor"] == df["goalsAgainst"]) & (df["situation"] == "all")]

print(ties["gameId"].head())

gameIds = df[df["gameId"].isin(ties["gameId"])]



#tmp = zip(ties_all["goalsFor"], ties_all["goalsAgainst"])
print(gameIds[["gameId", "situation", "goalsFor", "goalsAgainst"]])
