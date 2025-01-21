import pandas as pd

df = pd.read_csv("all_teams_1.csv")

num = 10

print(type(df[df["gameId"] == 2008020001]))
print(df[df["gameId"] == 2008020001])
for row in df.itertuples():
    if(num > 10):
        break
    print(type(row["gameId"]))
    num += 1
 
