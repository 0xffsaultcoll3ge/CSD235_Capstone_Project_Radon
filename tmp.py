import subprocess
import time
import pandas as pd
import numpy as np

def run_node_server():
    node_process = subprocess.Popen(['node', 'web/scripts/server.js'])

    return node_process
def run_flask_server():
    node_process = subprocess.Popen(['python3', 'backend/api/app.py'])
    return flask_process

if __name__ == "__main__":
    df = pd.read_csv("all_games_preproc.csv")
    # print(df.shape)
    # print(df)
    # corr = []
    # for col in df.columns.tolist():
    #     if df[col].dtype == np.float64:
    #     # if "ryder" in col:
    #         val = df["winner"].corr(df[col])
    #         corr.append([col, val])
    # for x in sorted(corr, key = lambda x: x[1]):
    #     print(f"Column: {x[0]} \t Corr: {x[1]}")
        
    print("Starting Node.js server...")
    node_process = run_node_server()
    time.sleep(3)
    flask_process = run_flask_server()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        node_process.terminate()
        flask_process.terminate()
