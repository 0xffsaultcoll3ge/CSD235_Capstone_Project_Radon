from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPRegressor
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.multioutput import MultiOutputRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sqlalchemy import create_engine
import xgboost as xgb
import pandas as pd
import lightgbm as lgb
import numpy as np
import tensorflow as tf
# from tensorflow.keras import Sequential
# from tensorflow.keras.layers import Dense, Dropout
# from tensorflow.keras.optimizers import Adam
from tqdm import tqdm
import warnings

class Model:
    def __init__(self, sport, event, db_uri, table_map, model_name="xgboost", params=None, task="auto" ):
        self.sport = sport
        self.event = event
        self.data = None
        self.target = None
        self.model = None
        self.db_uri = db_uri
        self.engine = create_engine(self.db_uri)
        self.table_map = table_map
        self.model_name = model_name.lower()
        self.params = params if params else {}
        self.task = task
    def __init__model__(self, is_classification):
        if self.model_name == "xgboost":
            return xgb.XGBClassifier(**self.params) if is_classification else xgb.XGBRegressor(**self.params)
        elif self.model_name == "lightgbm":
            return lgb.LGBMClassifier(**self.params) if is_classification else lgb.LGBMRegressor(**self.params)
        elif self.model_name == "random_forest":
            return RandomForestClassifier(**self.params) if is_classification else RandomForestRegressor(**self.params)
        # elif self.model_name == "neural_network":
        #     return self._initialize_neural_network(is_classification)
        elif self.model_name == "gradient_boosting":
            return GradientBoostingClassifier(**self.params) if is_classification else GradientBoostingRegressor(**self.params)
        else:
            raise ValueError(f"Unsupported model: {self.model_name}")
    def load_data_csv(self, filepath):
        try:
            return pd.read_csv(filepath), None
        except:
            return None, e

    def load_data(self, query, chunksize):
        try:
            if(chunksize):
                return pd.concat(pd.read_sql(query, self.engine, chunksize=chunksize)), None
            else:
                return pd.read_sql(query, self.engine), None
        except Exception as e:
            return None, e
    
    def get_important_features(self):
        #to-do
    def train(self):
        #to do
    def predict(self, X, y):
        return self.model.predict(X, y)
    def predict_proba(self, X, y):
        return self.model.predict_proba(X,y)
    def prediction_accuracy(self, y, y_pred, accuracy):
        return accuracy(y, y_pred)
    def get_model(self):
        return self.model
    def save_model(self, path, acc):
        this.model.save_model(f"./models/{self.sport}/{self.event}/{self.model_name}/{self.model_name}_{acc * 100}%_{self.event}")


    
    
    