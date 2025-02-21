import subprocess
import time

def run_node_server():
    node_process = subprocess.Popen(['node', 'web/server.js'])

    return node_process
def run_flask_server():
    node_process = subprocess.Popen(['python3', 'backend/api/app.py'])
    return flask_process

if __name__ == "__main__":
    print("Starting Node.js server...")
    node_process = run_node_server()
    flask_process = run_flask_server()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        node_process.terminate()
        flask_process.terminate()
