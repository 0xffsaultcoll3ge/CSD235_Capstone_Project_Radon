from flask import *
from queue import Queue, PriorityQueue


class APIService:
    def __init__(self):
        self.task_queue = PriorityQueue()
        self.app = Flask(__name__)
    s
