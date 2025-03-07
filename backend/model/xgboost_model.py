import keras
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from tensorflow.keras import layers, regularizers
from tensorflow.keras.callbacks import EarlyStopping

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, RandomizedSearchCV
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPRegressor
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, RandomForestRegressor, StackingClassifier, VotingClassifier
from sklearn.multioutput import MultiOutputRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, mean_squared_error, r2_score, roc_auc_score, accuracy_score, log_loss
from skopt import BayesSearchCV
from sklearn.naive_bayes import GaussianNB
from scipy.stats import randint, uniform, loguniform
import xgboost as xgb
from xgboost import XGBClassifier
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from tqdm import tqdm
import warnings
import sys

warnings.filterwarnings('ignore')

def test_nn_nhl_ml():
    df = pd.read_csv("all_games_preproc.csv", low_memory=False)
    df["gameDate"] = pd.to_datetime(df["gameDate"], format="%Y%m%d")
    df["lastGameFor"] = df.groupby(df["team"])["gameDate"].shift()
    df["lastGameAgainst"] = df.groupby(df["opposingTeam"])["gameDate"].shift()
    df["daysRestFor"] = (df["gameDate"] - df["lastGameFor"]).dt.days
    df["daysRestAgainst"] = (df["gameDate"] - df["lastGameAgainst"]).dt.days
    df = df.dropna(axis=0)
    # goal_cols = []
    # for col in df.columns:
    #     if "goal" in col or "GoalsFor" in col:
    #         goal_cols.append(col)
    # df.drop(goal_cols, axis = 1, inplace=True)
    # df = df.fillna(0)

    print(df)

    print(df.shape)
    
    # Move to preprocess

    # df["total_goals"] = df[df["goalsFor"] + df["goalsAgainst"]]
    df["eloExpectedAgainst"] = 1 - df["eloExpectedFor"]
    X = df.loc[:, df.columns.str.contains("ema") 
    +df.columns.str.contains('elo') 
    + df.columns.str.contains('daysRest')
    + df.columns.str.contains('winPercentage') ] #+ df.columns.str.contains('eloExpectedFor')
    # X = (X -X.mean())/(X-X.std())
    X = X.drop(columns=[col for col in X.columns if "winner" in col])
    y = df.loc[:, "winner"]

    gb = XGBClassifier(objective="multi:softprob", num_class=2, learning_rate=0.013, max_depth=3)
    gb.fit(X, y)
    # feature_importance = np.mean(np.abs(gnb.theta_), axis=0)
    # important = pd.DataFrame({'Feature': X.columns, 'Importance': gb.feature_importances})
    # important.sort_values(by='Importance', ascending=False, inplace=True)
    important = pd.DataFrame({'Feature': X.columns, 'Importance':gb.feature_importances_})
    important.sort_values(by='Importance', ascending=False, inplace=True)
    top_features = important['Feature'].head(50)
    print(top_features)
    X = X.loc[:, top_features]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    model = tf.keras.Sequential([
        Dense(128, activation='relu', kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4)),
        layers.Dropout(0.3),
        Dense(64, activation='relu', kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4)),
        layers.Dropout(0.3),
        layers.Dense(32, activation='relu', kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4)),
        layers.Dropout(0.3),
        layers.Dense(1, activation='sigmoid')
    ])

    model.compile(loss='binary_crossentropy', optimizer='sgd', metrics=["binary_accuracy"])

    model.summary()

    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    history = model.fit(X_train, y_train, epochs=400, batch_size=2, validation_data=(X_test, y_test), callbacks=[early_stop])

    loss,accuracy = model.evaluate(X_test, y_test)
    print(f"Test Loss: {loss}")
    print(f"Test Accuracy: {accuracy}")

def test_xgbosot_nhl_spread(spread):
    df = pd.read_csv("all_games_preproc.csv", low_memory=False)
    df = df[df["goalDiffFor"] != 0]
    X = df.loc[:, df.columns.str.contains("ema") + df.columns.str.contains("elo")]

    df[f"spread_{spread}"] = np.where(df["goalDiffFor"] > spread, 1.0, 0.0)
    y = df.loc[:, f"spread_{spread}"]

    X_temp = X.loc[:, X.columns.str.contains("ema") + X.columns.str.contains("eloExpected")]

    rf = XGBClassifier()
    rf.fit(X_temp, y)
    important = pd.DataFrame({'Feature': X_temp.columns, 'Importance':rf.feature_importances_})
    important.sort_values(by='Importance', ascending=False, inplace=True)
    top_features = important['Feature'].head(50)
    print(top_features)
    X = X.loc[:, top_features]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)

    params = {
        'objective': 'multi:softprob',  # You can change this to multi-class if needed
        'eval_metric': 'mlogloss',  # You can also use 'error', 'auc', etc.
        'num_class': 2,
        'booster': 'gbtree',
        'learning_rate': 0.01,
        'max_depth':4,
        # 'num_class':2,
        # 'lambda': 0.2,  # L2 regularization term
        # 'alpha': 0.1,  # L1 regularization term
        # 'scale_pos_weight': [1., 2, 3]  # for imbalanced classes
    }

    # Perform cross-validation with xgb.cv
    cv_results = xgb.cv(
    params=params,
    dtrain=dtrain,
    num_boost_round=1000,   # Number of boosting rounds (trees)
    nfold=5,               # Number of folds for cross-validation
    early_stopping_rounds=20,  # Stop early if the validation metric does not improve
    metrics={'mlogloss'},  # You can use logloss for classification
    as_pandas=True        # Return results as a pandas DataFrame
    )

    # Display the cross-validation results
    print(cv_results)

    # Train the final model with the best parameters
    best_num_boost_round = cv_results['test-mlogloss-mean'].idxmin()  # Find best boosting round
    params['num_boost_round'] = best_num_boost_round

    # Train model on entire training set using the best boosting rounds
    final_model = xgb.train(params, dtrain, num_boost_round=best_num_boost_round)

    # Make predictions with the final model
    y_pred_proba= final_model.predict(dtest)[:, 1]
    y_train_pred_proba = final_model.predict(dtrain)[:, 1]

    y_pred = (y_pred_proba > 0.5).astype(int)
    y_train_pred = (y_train_pred_proba > 0.5).astype(int)


    # Evaluate accuracy on the test set
    accuracy = accuracy_score(y_test, y_pred)
    train_acc = accuracy_score(y_train, y_train_pred)
    print(f"Test Accuracy: {accuracy * 100:.2f}%")
    print(f"Training Accuracy: {train_acc * 100:.2f}%")

    train_nhl_spread(X, y, spread)



def train_nhl_spread(X, y, spread, params=None):
    acc_results = []
    for x in tqdm(range(300)):
        print(len(X), len(y))
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        
        train = xgb.DMatrix(X_train, label=y_train)
        test = xgb.DMatrix(X_test, label=y_test)

        params = params if params else {
            'max_depth': 3,
            'eta': 0.01,
            'objective':'multi:softprob',
            'num_class':2,
            # 'n_estimators':400
        }

        epochs = 750

        model = xgb.train(params, train, epochs)
        predictions = model.predict(test)
        y_pred = []

        for z in predictions:
            y_pred.append(np.argmax(z))
        acc = round(accuracy_score(y_test, y_pred)*100, 1)
        print(acc)
        acc_results.append(acc)

        if acc == max(acc_results):
            model.save_model(f"./backend/model/models/spread/XGBoot_{acc}%_Spread_{spread}.json")
def test_xgboost_nhl_ou(ou):
    df = pd.read_csv("all_games_preproc.csv", low_memory=False)
    df = df[df["goalDiffFor"] != 0]
    X = df.loc[:, df.columns.str.contains("ema") + df.columns.str.contains("elo")]
    df["total_goals"] = df["goalsFor"] + df["goalsAgainst"]
    df[f"ou_{ou}"] = np.where(df["total_goals"] > ou, 1.0, 0.0)

    y = df.loc[:, f"ou_{ou}"]

    rf = XGBClassifier()

    X_temp = X.loc[:, X.columns.str.contains("ema") + X.columns.str.contains("eloExpected")]

    rf.fit(X_temp, y)
    important = pd.DataFrame({'Feature': X_temp.columns, 'Importance':rf.feature_importances_})
    important.sort_values(by='Importance', ascending=False, inplace=True)
    top_features = important['Feature'].head(20)
    print(top_features)
    X = X.loc[:, top_features]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)

    params = {
        'objective': 'multi:softprob',  # You can change this to multi-class if needed
        'eval_metric': 'mlogloss',  # You can also use 'error', 'auc', etc.
        'num_class': 2,
        'booster': 'gbtree',
        'learning_rate': 0.01,
        'max_depth':4,
        # 'num_class':2,
        # 'lambda': 0.2,  # L2 regularization term
        # 'alpha': 0.1,  # L1 regularization term
        # 'scale_pos_weight': [1., 2, 3]  # for imbalanced classes
    }

    # Perform cross-validation with xgb.cv
    cv_results = xgb.cv(
    params=params,
    dtrain=dtrain,
    num_boost_round=30,   # Number of boosting rounds (trees)
    nfold=5,               # Number of folds for cross-validation
    early_stopping_rounds=20,  # Stop early if the validation metric does not improve
    metrics={'mlogloss'},  # You can use logloss for classification
    as_pandas=True        # Return results as a pandas DataFrame
    )

    # Display the cross-validation results
    print(cv_results)

    # Train the final model with the best parameters
    best_num_boost_round = cv_results['test-mlogloss-mean'].idxmin()  # Find best boosting round
    params['num_boost_round'] = best_num_boost_round

    # Train model on entire training set using the best boosting rounds
    final_model = xgb.train(params, dtrain, num_boost_round=best_num_boost_round)

    # Make predictions with the final model
    y_pred_proba= final_model.predict(dtest)[:, 1]
    y_train_pred_proba = final_model.predict(dtrain)[:, 1]

    y_pred = (y_pred_proba > 0.5).astype(int)
    y_train_pred = (y_train_pred_proba > 0.5).astype(int)


    # Evaluate accuracy on the test set
    accuracy = accuracy_score(y_test, y_pred)
    train_acc = accuracy_score(y_train, y_train_pred)
    print(f"Test Accuracy: {accuracy * 100:.2f}%")
    print(f"Training Accuracy: {train_acc * 100:.2f}%")

    train_nhl_ou(X, y, ou, params)
def train_nhl_ou(X, y, ou, params=None):
    acc_results = []
    for x in tqdm(range(300)):
        print(len(X), len(y))
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        
        train = xgb.DMatrix(X_train, label=y_train)
        test = xgb.DMatrix(X_test, label=y_test)

        params = params if params else {
            'max_depth': 3,
            'eta': 0.01,
            'objective':'multi:softprob',
            'num_class':2,
            # 'n_estimators':400
        }

        epochs = 750

        model = xgb.train(params, train, epochs)
        predictions = model.predict(test)
        y_pred = []

        for z in predictions:
            y_pred.append(np.argmax(z))
        acc = round(accuracy_score(y_test, y_pred)*100, 1)
        print(acc)
        acc_results.append(acc)

        if acc == max(acc_results):
            model.save_model(f"./backend/model/models/OU/XGBoot_{acc}%_OU_{ou}.json")
def tune_xgboost_nhl_ou(ou):
    df = pd.read_csv("all_games_preproc.csv", low_memory=False)
    X = df.loc[:, df.columns.str.contains("ema") + df.columns.str.contains("elo")]
    df["total_goals"] = df["goalsFor"] + df["goalsAgainst"]
    df[f"ou_{ou}"] = np.where(df["total_goals"] > ou, 1.0, 0.0)
    df = df[df["goalDiffFor"] != 0]

    y = df.loc[:, f"ou_{ou}"]

    rf = XGBClassifier()

    X_temp = X.loc[:, X.columns.str.contains("ema") + X.columns.str.contains("eloExpected")]

    rf.fit(X_temp, y)
    important = pd.DataFrame({'Feature': X_temp.columns, 'Importance':rf.feature_importances_})
    important.sort_values(by='Importance', ascending=False, inplace=True)
    top_features = important['Feature'].head(20)
    print(top_features)
    X = X.loc[:, top_features]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    param_dist = {
        # Core Parameters
        'n_estimators': randint(180, 350),  # Number of trees
        'max_depth': randint(3, 5),  # Maximum depth of trees
        'learning_rate': uniform(0.01, 0.05),  # Learning rate (eta)
        'reg_alpha': uniform(0, 1),  # L1 regularization (alpha)
        'reg_lambda': uniform(0, 1),  # L2 regularization (lambda)

        # Learning Task Parameters
        'objective': ['multi:softprob'],  # Objective function
        'eval_metric': ['mlogloss'],
        'num_class': [2],
        'seed': [42],  # Random seed
    }

    # Initialize the XGBClassifier with GPU support
    est = XGBClassifier(tree_method='gpu_hist', random_state=42)

    # Initialize RandomizedSearchCV
    random_search = RandomizedSearchCV(
        estimator=est,
        param_distributions=param_dist,
        n_iter=50,
        scoring='neg_log_loss',
        cv=3,
        verbose=2,
        n_jobs=-1,
        random_state=42
    )

    # Fit the model
    random_search.fit(X_train, y_train)

    print("Best parameters found: ", random_search.best_params_)
    print("Best score: ", random_search.best_score_)

    best_model = random_search.best_estimator_
    y_pred_proba = best_model.predict_proba(X_test)[:, 1]  # Probability of home team winning

    # Calculate ROC AUC, accuracy, and log loss
    y_pred = (y_pred_proba > 0.5).astype(int)  # Convert probabilities to class labels
    test_accuracy = accuracy_score(y_test, y_pred)
    test_log_loss = log_loss(y_test, y_pred_proba)

    # print(f"Test set ROC AUC: {test_roc_auc:.4f}")
    print(f"Test set accuracy: {test_accuracy:.4f}")
    print(f"Test set log loss: {test_log_loss:.4f}")

    train_nhl_ou(X, y, ou, params=random_search.best_params_)
def train_nhl_ml(X, y, params=None):
    acc_results = []
    for x in tqdm(range(300)):
        print(len(X), len(y))
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        
        train = xgb.DMatrix(X_train, label=y_train)
        test = xgb.DMatrix(X_test, label=y_test)

        params = params if params else {
            'max_depth': 3,
            'eta': 0.01,
            'objective':'multi:softprob',
            'num_class':2,
            # 'n_estimators':400
        }

        epochs = 750

        model = xgb.train(params, train, epochs)
        predictions = model.predict(test)
        y_pred = []

        for z in predictions:
            y_pred.append(np.argmax(z))
        acc = round(accuracy_score(y_test, y_pred)*100, 1)
        print(acc)
        acc_results.append(acc)

        if acc == max(acc_results):
            model.save_model('./backend/model/models/XGBoot_{}%_ML.json'.format(acc))
def tune_xgboost_nhl_ml():
    df = pd.read_csv("all_games_preproc.csv", low_memory=False)
    df["gameDate"] = pd.to_datetime(df["gameDate"], format="%Y%m%d")
    df["lastGameFor"] = df.groupby(df["team"])["gameDate"].shift()
    df["lastGameAgainst"] = df.groupby(df["opposingTeam"])["gameDate"].shift()
    df["daysRestFor"] = (df["gameDate"] - df["lastGameFor"]).dt.days
    df["daysRestAgainst"] = (df["gameDate"] - df["lastGameAgainst"]).dt.days
    df = df.dropna(axis=0)
    df = df[df["goalDiffFor"] != 0]
    # goal_cols = []
    # for col in df.columns:
    #     if "goal" in col or "GoalsFor" in col:
    #         goal_cols.append(col)
    # df.drop(goal_cols, axis = 1, inplace=True)
    # df = df.fillna(0)

    print(df)

    print(df.shape)
    
    # Move to preprocess

    # df["total_goals"] = df[df["goalsFor"] + df["goalsAgainst"]]
    df["eloExpectedAgainst"] = 1 - df["eloExpectedFor"]
    X = df.loc[:, df.columns.str.contains("ema") 
    +df.columns.str.contains('elo') 
    + df.columns.str.contains('daysRest')
    + df.columns.str.contains('winPercentage') ] #+ df.columns.str.contains('eloExpectedFor')
    # X = (X -X.mean())/(X-X.std())
    X = X.drop(columns=[col for col in X.columns if "winner" in col])
    y = df.loc[:, "winner"]

    gb = XGBClassifier(objective="multi:softprob", num_class=2, learning_rate=0.013)
    gb.fit(X, y)
    # feature_importance = np.mean(np.abs(gnb.theta_), axis=0)
    # important = pd.DataFrame({'Feature': X.columns, 'Importance': gb.feature_importances})
    # important.sort_values(by='Importance', ascending=False, inplace=True)
    important = pd.DataFrame({'Feature': X.columns, 'Importance':gb.feature_importances_})
    important.sort_values(by='Importance', ascending=False, inplace=True)
    top_features = important['Feature'].head(100)
    print(top_features)
    X = X.loc[:, top_features]


    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    param_dist = {
        # Core Parameters
        'n_estimators': randint(100, 500),  # Number of trees
        'max_depth': [1, 2, 3, 4],  # Maximum depth of trees
        'learning_rate': uniform(0.01, 0.05),  # Learning rate (eta)
        'reg_alpha': uniform(0, 1),  # L1 regularization (alpha)
        'reg_lambda': uniform(0, 1),  # L2 regularization (lambda)

        # Learning Task Parameters
        'objective': ['multi:softprob'],  # Objective function
        'eval_metric': ['mlogloss'],
        'num_class': [2],
        'seed': [42],  # Random seed
    }

    # Initialize the XGBClassifier with GPU support
    est = XGBClassifier(tree_method='gpu_hist', random_state=42)

    # Initialize RandomizedSearchCV
    random_search = RandomizedSearchCV(
        estimator=est,
        param_distributions=param_dist,
        n_iter=50,
        scoring='neg_log_loss',
        cv=3,
        verbose=2,
        n_jobs=-1,
        random_state=42
    )

    # Fit the model
    random_search.fit(X_train, y_train)

    print("Best parameters found: ", random_search.best_params_)
    print("Best score: ", random_search.best_score_)

    best_model = random_search.best_estimator_
    y_pred_proba = best_model.predict_proba(X_test)[:, 1]  # Probability of home team winning

    # Calculate ROC AUC, accuracy, and log loss
    y_pred = (y_pred_proba > 0.5).astype(int)  # Convert probabilities to class labels
    test_accuracy = accuracy_score(y_test, y_pred)
    test_log_loss = log_loss(y_test, y_pred_proba)

    # print(f"Test set ROC AUC: {test_roc_auc:.4f}")
    print(f"Test set accuracy: {test_accuracy:.4f}")
    print(f"Test set log loss: {test_log_loss:.4f}")

    train_nhl_ml(X, y, params=random_search.best_params_) 
def test_xgboost_nhl_ml():
    df = pd.read_csv("all_games_preproc.csv", low_memory=False)
    df["gameDate"] = pd.to_datetime(df["gameDate"], format="%Y%m%d")
    df["lastGameFor"] = df.groupby(df["team"])["gameDate"].shift()
    df["lastGameAgainst"] = df.groupby(df["opposingTeam"])["gameDate"].shift()
    df["daysRestFor"] = (df["gameDate"] - df["lastGameFor"]).dt.days
    df["daysRestAgainst"] = (df["gameDate"] - df["lastGameAgainst"]).dt.days
    df = df[df["goalDiffFor"] != 0]
    df = df.dropna()
    # df["tie"] = np.where(df["goalDiffFor"] == 0.0, 1.0, 0.0)
    # goal_cols = []
    # for col in df.columns:
    #     if "goal" in col or "GoalsFor" in col:
    #         goal_cols.append(col)
    # df.drop(goal_cols, axis = 1, inplace=True)
    # df = df.fillna(0)

    print(df)

    print(df.shape)
    
    # Move to preprocess

    # df["total_goals"] = df[df["goalsFor"] + df["goalsAgainst"]]
    df["eloExpectedAgainst"] = 1 - df["eloExpectedFor"]
    X = df.loc[:, df.columns.str.contains("ema") 
    +df.columns.str.contains('eloExpected') 
    + df.columns.str.contains('daysRest')
    + df.columns.str.contains('winPercentage') ] #+ df.columns.str.contains('eloExpectedFor')
    # X = (X -X.mean())/(X-X.std())
    X = X.drop(columns=[col for col in X.columns if "winner" in col])
    y = df.loc[:, "winner"]

    gb = XGBClassifier(objective="multi:softprob", num_class=2, learning_rate=0.013, max_depth=3)
    gb.fit(X, y)
    # feature_importance = np.mean(np.abs(gnb.theta_), axis=0)
    # important = pd.DataFrame({'Feature': X.columns, 'Importance': gb.feature_importances})
    # important.sort_values(by='Importance', ascending=False, inplace=True)
    important = pd.DataFrame({'Feature': X.columns, 'Importance':gb.feature_importances_})
    important.sort_values(by='Importance', ascending=False, inplace=True)
    top_features = important['Feature'].head(90)
    print(top_features)
    X = X.loc[:, top_features]


    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)

    params = {
    'objective': 'multi:softprob',
    'eval_metric': 'mlogloss',
    'booster': 'gbtree',
    'learning_rate': 0.04595,
    'max_depth': 1,
    'num_class':2,
    'n_estimators': 417,
    'alpha': np.float64('0.005061583846218687'),
    'lambda': np.float64('0.16080805141749865')
    # 'lambda': 0.2,  # L2 regularization term
    # 'alpha': 0.1,  # L1 regularization term
    # 'scale_pos_weight': [1., 2, 3]  # for imbalanced classes
    }

    # Perform cross-validation with xgb.cv
    cv_results = xgb.cv(
    params=params,
    dtrain=dtrain,
    num_boost_round=1000,   # Number of boosting rounds (trees)
    nfold=10,               # Number of folds for cross-validation
    early_stopping_rounds=25,  # Stop early if the validation metric does not improve
    metrics={'mlogloss'},  # You can use logloss for classification
    as_pandas=True        # Return results as a pandas DataFrame
    )

    print(cv_results)

    best_num_boost_round = cv_results['test-mlogloss-mean'].idxmin()  # Find best boosting round
    params['num_boost_round'] = best_num_boost_round

    final_model = xgb.train(params, dtrain, num_boost_round=best_num_boost_round)

    y_pred_prob = final_model.predict(dtest)[:, 1]
    y_pred = (y_pred_prob >= 0.5).astype(int)

    y_train_pred = (final_model.predict(dtrain)[:, 1] >= 0.5).astype(int)

    accuracy = accuracy_score(y_test, y_pred)
    train_acc = accuracy_score(y_train, y_train_pred)
    print(f"Test Accuracy: {accuracy * 100:.2f}%")
    print(f"Training Accuracy: {train_acc * 100:.2f}%")

    train_nhl_ml(X, y, params=params)
def test_random_forest_nhl_ml():
    df = pd.read_csv("all_games_preproc.csv", low_memory=False)
    df["gameDate"] = pd.to_datetime(df["gameDate"], format="%Y%m%d")
    df["lastGameFor"] = df.groupby(df["team"])["gameDate"].shift()
    df["lastGameAgainst"] = df.groupby(df["opposingTeam"])["gameDate"].shift()
    df["daysRestFor"] = (df["gameDate"] - df["lastGameFor"]).dt.days
    df["daysRestAgainst"] = (df["gameDate"] - df["lastGameAgainst"]).dt.days
    df = df.dropna()
    # goal_cols = []
    # for col in df.columns:
    #     if "goal" in col or "GoalsFor" in col:
    #         goal_cols.append(col)
    # df.drop(goal_cols, axis = 1, inplace=True)
    # df = df.fillna(0)

    print(df)

    print(df.shape)
    # df["total_goals"] = df[df["goalsFor"] + df["goalsAgainst"]]
    df["eloExpectedAgainst"] = 1 - df["eloExpectedFor"]
    X = df.loc[:, df.columns.str.contains("ema") 
    +df.columns.str.contains('elo') 
    + df.columns.str.contains('daysRest') ] #+ df.columns.str.contains('eloExpectedFor')
    X = (X -X.mean())/(X-X.std())
    y = df.loc[:, "winner"]

    rf = RandomForestClassifier()
    rf.fit(X, y)

    important = pd.DataFrame({'Feature': X.columns, 'Importance':rf.feature_importances_})
    important.sort_values(by='Importance', ascending=False, inplace=True)
    top_features = important['Feature'].head(90)
    print(top_features)
    X = X.loc[:, top_features]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


    param_dist = {
    'n_estimators': randint(50, 500),  # Randomly sample number of trees
    'max_depth': [None] + list(randint(5, 50).rvs(10)),  # Randomly sample max depth
    'min_samples_split': randint(2, 20),  # Randomly sample min_samples_split
    'min_samples_leaf': randint(1, 10),  # Randomly sample min_samples_leaf
    'max_features': ['sqrt', 'log2', None],  # Fixed options
    'bootstrap': [True, False],  # Fixed options
    'criterion': ['gini', 'entropy']  # Fixed options
    }



# Initialize RandomizedSearchCV
    random_search = RandomizedSearchCV(estimator=rf, param_distributions=param_dist, n_iter=50, scoring='accuracy', cv=3, verbose=2, n_jobs=-1, random_state=42)

    # random_search = RandomForestClassifier()

    # Fit the model
    random_search.fit(X_train, y_train)

    # print("Best parameters found: ", random_search.best_params_)
    # print("Best cross-validation accuracy: ", random_search.best_score_)

    # # Evaluate the best model
    best_model = random_search.best_estimator_
    y_pred = best_model.predict(X_test)
    y_pred = random_search.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Test set accuracy: {accuracy:.2f}")

if __name__ == "__main__":
    if sys.argv[1] == "ml":
        test_xgboost_nhl_ml()
    # test_random_forest_nhl_ml()
    if sys.argv[1] == "spread":
        spread = float(sys.argv[2])
        test_xgbosot_nhl_spread(spread)
    if sys.argv[1] == "ou":
        ou = float(sys.argv[2])
        test_xgboost_nhl_ou(ou)
