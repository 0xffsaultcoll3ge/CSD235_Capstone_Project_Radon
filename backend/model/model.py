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

def get_classifier(model_name, params=None):
    if model_name == "xgboost":
        return xgb.XGBClassifier(**params) if is_classification else xgb.XGBRegressor(**params)
    elif model_name == "lightgbm":
        return lgb.LGBMClassifier(**params) if is_classification else lgb.LGBMRegressor(**params)
    elif model_name == "random_forest":
        return RandomForestClassifier(**params) if is_classification else RandomForestRegressor(**params)
    elif model_name == "gradient_boosting":
        return GradientBoostingClassifier(**params) if is_classification else GradientBoostingRegressor(**params)
    else:
        raise ValueError(f"Unsupported model: {model_name}")
def get_regressor(model_name, params=None):
    if model_name == "xgboost":
        return xgb.XGBRegressor(**params)
    elif model_name == "lightgbm":
        return lgb.LGBMRegressor(**params)
    elif model_name == "random_forest":
        return RandomForestRegressor(**params)
    elif model_name == "gradient_boosting":
        return GradientBoostingRegressor(**params)
    else:
        raise ValueError(f"Unsupported model: {model_name}")
class Model:
    # separate db layer
    def __init__(self, sport, event, db_uri, table_map, model_name="xgboost", params=None, task="auto" ):
        self.sport = sport
        self.event = event
        self.data = None
        self.target = None
        self.model = None
        # self.db_uri = db_uri
        # self.engine = create_engine(self.db_uri)
        # self.table_map = table_map
        self.model_name = model_name.lower()
        self.params = params if params else {}
        self.task = task
    # def __init__model__(self, is_classification):
    #     if self.model_name == "xgboost":
    #         return xgb.XGBClassifier(**self.params) if is_classification else xgb.XGBRegressor(**self.params)
    #     elif self.model_name == "lightgbm":
    #         return lgb.LGBMClassifier(**self.params) if is_classification else lgb.LGBMRegressor(**self.params)
    #     elif self.model_name == "random_forest":
    #         return RandomForestClassifier(**self.params) if is_classification else RandomForestRegressor(**self.params)
    #     # elif self.model_name == "neural_network":
    #     #     return self._initialize_neural_network(is_classification)
    #     elif self.model_name == "gradient_boosting":
    #         return GradientBoostingClassifier(**self.params) if is_classification else GradientBoostingRegressor(**self.params)
    #     else:
    #         raise ValueError(f"Unsupported model: {self.model_name}")
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
    
    def get_important_features(self, X=self.data, y=self.target, model_name):
        if model_name == "random_forest":
            rf = RandomForestClassifier(random_state = 42)
            rf.fit(X, y)
            important = pd.DataFrame({'Feature': X.columns, 'Importance':rf.feature_importances_})
            important.sort_values(by='Importance', ascending=False, inplace=True)

            return important
    def cross_validate(self):
        assert(self.model_name in ["xgboost", "lightgbm", "gradient_boosting"])
        if self.sport == "NHL" and self.event == "ML":
            assert(self.data != None and self.target != None)

            important_feats = get_important_features().head(15)
            X = self.data.loc[:, important_feats]
            
            params = {
                'objective': 'multi:softprob',
                'eval_metric': 'mlogloss',
                'booster': 'gbtree',
                'learning_rate': 0.01,
                'max_depth': 5,
                'num_class':2
            }

            cv_results = xgb.cv(
                params=params,
                dtrain=X,
                num_boost_round=1000,
                nfold=3,
                early_stopping_rounds=25,
                metrics={'mlogloss'},
                as_pandas=True
            )

            best_num_boost_round = cv_results['test-mlogloss-mean'].idxmin()

            return cv_results, best_num_boost_round






    def train(self, params):
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


    
    
    