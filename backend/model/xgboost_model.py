from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPRegressor
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, RandomForestRegressor, StackingClassifier
from sklearn.multioutput import MultiOutputRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score
from skopt import BayesSearchCV
from sklearn.naive_bayes import GaussianNB
import xgboost as xgb
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')

def test_xgboost_nhl_ou():
    df = pd.read_csv("all_games_preproc.csv", low_memory=False)
    X = df.loc[:, df.columns.str.contains("ema") + df.columns.str.contains("eloExpected")]
    df["total_goals"] = df["goalsFor"] + df["goalsAgainst"]

    # X = (X -X.mean())/X.std()

    # df.loc[:, "total_goals"] =  (df['total_goals'] - df['total_goals'].min()) / (df['total_goals'].max() - df['total_goals'].min())

    y = df.loc[:, 'total_goals']

    df = df.fillna(0)

    rf = RandomForestRegressor(n_jobs=-1)

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
        'objective': 'reg:squarederror',  # You can change this to multi-class if needed
        'eval_metric': 'rmse',  # You can also use 'error', 'auc', etc.
        'booster': 'gbtree',
        'learning_rate': 0.01,
        'max_depth': 7,
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
    metrics={'rmse'},  # You can use logloss for classification
    as_pandas=True        # Return results as a pandas DataFrame
    )

    # Display the cross-validation results
    print(cv_results)

    # Train the final model with the best parameters
    best_num_boost_round = cv_results['test-rmse-mean'].idxmin()  # Find best boosting round
    params['num_boost_round'] = best_num_boost_round

    # Train model on entire training set using the best boosting rounds
    final_model = xgb.train(params, dtrain, num_boost_round=best_num_boost_round)

    # Make predictions with the final model
    y_pred = final_model.predict(dtest)
    print(y_pred)
    print(y_test)
    y_train_pred = final_model.predict(dtrain)

    # Evaluate accuracy on the test set
    accuracy = r2_score(y_test, y_pred)
    train_acc = r2_score(y_train, y_train_pred)
    print(f"Test Accuracy: {accuracy * 100:.2f}%")
    print(f"Training Accuracy: {train_acc * 100:.2f}%")

def train_nhl_ml(X, y, params=None):
    acc_results = []
    for x in tqdm(range(300)):
        print(len(X), len(y))
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        
        train = xgb.DMatrix(X_train, label=y_train)
        test = xgb.DMatrix(X_test, label=y_test)

        params = params if params else {
            'max_depth': 5,
            'eta': 0.015,
            # 'n_estimators': 180,
            'objective':'multi:softprob',
            'num_class':2
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
def test_xgboost_nhl_ml():
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
    
    # Move to preprocess

    # df["total_goals"] = df[df["goalsFor"] + df["goalsAgainst"]]
    df["eloExpectedAgainst"] = 1 - df["eloExpectedFor"]
    X = df.loc[:, df.columns.str.contains("ema") 
    +df.columns.str.contains('eloExpected') 
    + df.columns.str.contains('daysRest')
    + df.columns.str.contains('winPercentage') ] #+ df.columns.str.contains('eloExpectedFor')
    X = (X -X.mean())/(X-X.std())
    y = df.loc[:, "winner"]

    gb = GradientBoostingClassifier()
    gb.fit(X, y)
    # feature_importance = np.mean(np.abs(gnb.theta_), axis=0)
    # important = pd.DataFrame({'Feature': X.columns, 'Importance': gb.feature_importances})
    # important.sort_values(by='Importance', ascending=False, inplace=True)
    important = pd.DataFrame({'Feature': X.columns, 'Importance':gb.feature_importances_})
    important.sort_values(by='Importance', ascending=False, inplace=True)
    top_features = important['Feature'].head(15)
    print(top_features)
    X = X.loc[:, top_features]


    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)

    params = {
    'objective': 'multi:softprob',
    'eval_metric': 'mlogloss',
    'booster': 'gbtree',
    'learning_rate': 0.01,
    'max_depth': 3,
    'num_class':2,
    'n_estimators': 400
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

def test_naive_bayes_nhl_ml():
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
    +df.columns.str.contains('eloExpected') 
    + df.columns.str.contains('daysRest') ] #+ df.columns.str.contains('eloExpectedFor')
    X = (X -X.mean())/(X-X.std())
    y = df.loc[:, "winner"]

    rf = GradientBoostingClassifier(random_state = 42)
    rf.fit(X, y)

    important = pd.DataFrame({'Feature': X.columns, 'Importance':rf.feature_importances_})
    important.sort_values(by='Importance', ascending=False, inplace=True)
    top_features = important['Feature'].head(10)
    print(top_features)
    X = X.loc[:, top_features]

    params = {
    'objective': 'multi:softprob',
    'eval_metric': 'mlogloss',
    'booster': 'gbtree',
    'learning_rate': 0.01,
    'max_depth': 5,
    'num_class':2
    }


    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    base_models = [
    ('gnb', GaussianNB()),
    ('gb', GradientBoostingClassifier(n_estimators=250, random_state=42)),
    ('rf', RandomForestClassifier(n_estimators=250, random_state=42))
    ]

    # Define meta-model
    meta_model = LogisticRegression()

    # Create Stacking Classifier
    model = StackingClassifier(estimators=base_models, final_estimator=meta_model, cv=5)


    # model = GaussianNB()

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    print(f"Accuracy score: {accuracy_score(y_test, y_pred) * 100}")
test_xgboost_nhl_ml()

# # Hyperparameters grid to search over
# param_grid = {
#     'max_depth': [6, 8, 10, 12],  # Vary the depth of the tree
#     'eta': [0.01, 0.05, 0.1, 0.2],  # Learning rate # Binary classification
#     'n_estimators': [50, 100, 200],  # Number of boosting rounds
#     'gamma': [0, 0.1, 0.2],  # Regularization term for controlling complexity
#     'min_child_weight': [1, 2, 3],  # Minimum sum of instance weight (hessian) in a child
#     'subsample': [0.6, 0.8, 1.0],  # Fraction of samples used for each tree
#     'colsample_bytree': [0.6, 0.8, 1.0],  # Fraction of features used per tree
#     'objective': ['binary:logistic'],  # Multi-class classification
#     'random_state': [42]  # Fix random seed for reproducibility
# }

# xgb_model = XGBClassifierWrapper()

# # Perform GridSearchCV
# grid_search = GridSearchCV(estimator=xgb_model, param_grid=param_grid, scoring='accuracy', cv=3, n_jobs=-1, verbose=1)

# acc_results = []
# for x in tqdm(range(300)):
#     print(len(X), len(y))
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=None)  # Removed fixed random_state
    
#     # Fit the GridSearch
#     grid_search.fit(X_train, y_train)
    
#     # Get the best parameters from grid search
#     best_params = grid_search.best_params_
#     print("Best hyperparameters:", best_params)

#     # Make predictions with the best model
#     best_model = grid_search.best_estimator_
#     y_pred_prob = best_model.predict_proba(X_test)   # This returns probabilities

#     # Convert probabilities to class labels (binary classification)
#     # Ensure y_test is 1D (flatten if necessary)
#     y_test = np.array(y_test).ravel()

#     print(f"y_test: {y_test[:10]}")  # Print first 10 to inspect
#     print(f"y_pred: {y_pred[:10]}")  # Print first 10 to inspect

#     # Ensure y_pred and y_test are both binary (0 or 1)
#     assert set(np.unique(y_pred)) <= {0, 1}, f"y_pred contains unexpected values: {np.unique(y_pred)}"
#     assert set(np.unique(y_test)) <= {0, 1}, f"y_test contains unexpected values: {np.unique(y_test)}"

#     # Compute accuracy
#     acc = round(accuracy_score(y_test, y_pred) * 100, 1)
#     print(acc)
#     acc_results.append(acc)

#     # Save the model if it achieves the best accuracy so far
#     if acc == max(acc_results):
#         best_model.model.save_model(f'./model/models/XGBoot_{acc}%_ML.json')





# #n_estimators -> number of decision trees in the model

# model.fit(X_train, y_train)

# # Get the feature importance scores
# importance_scores = model.feature_importances_

# print(importance_scores)

# # Select the top 10 most important features
# selected_features = importance_scores.argsort()[-10:]

# print(selected_features)
# print(X_train.iloc[:, selected_features])

# # Create a new XGBClassifier with the selected features
# selected_model = XGBClassifier(n_estimators=100, learning_rate=0.1, random_state=42)

# # Train the new model using only the selected features
# selected_model.fit(X_train.iloc[:, selected_features], y_train)

# # Evaluate the original model
# y_pred = model.predict(X_test)
# original_accuracy = accuracy_score(y_test, y_pred)

# # Evaluate the model with selected features
# y_pred_selected = selected_model.predict(X_test.iloc[:, selected_features])
# selected_accuracy = accuracy_score(y_test, y_pred_selected)

# print(f"Original Model Accuracy: {original_accuracy:.4f}")
# print(f"Selected Features Model Accuracy: {selected_accuracy:.4f}")

# # Using scikit-learn's SelectFromModel for feature selection
# selector = SelectFromModel(model, prefit=True)
# X_train_selected = selector.transform(X_train)
# X_test_selected = selector.transform(X_test)

# # Train a new model using the selected features
# selected_model_pipeline = XGBClassifier(n_estimators=100, learning_rate=0.1, random_state=42)
# selected_model_pipeline.fit(X_train_selected, y_train)

# # Evaluate the model with selected features using the pipeline
# y_pred_pipeline = selected_model_pipeline.predict(X_test_selected)
# selected_accuracy_pipeline = accuracy_score(y_test, y_pred_pipeline)

# print(f"Selected Features Model Accuracy (Pipeline): {selected_accuracy_pipeline:.4f}")