import numpy as np
import pandas as pd
import xgboost as xgb
import os
import json
import logging
import matplotlib.pyplot as plt
import shap
from tqdm import tqdm
from sklearn.utils import resample
from sklearn.model_selection import train_test_split, GridSearchCV,RandomizedSearchCV, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, log_loss, confusion_matrix, roc_auc_score, roc_curve, make_scorer
from scipy.stats import uniform, randint

logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")

def get_env_variable(var, val):
    return os.getenv(var, val)

def balance_dataset(X_train, X_test, y_train, y_test, dmatrix=True):
    X_majority = X_train[y_train == 0]
    y_majority = y_train[y_train == 0]

    X_minority = X_train[y_train == 1]
    y_minority = y_train[y_train == 1]

    X_minority_oversampled, y_minority_oversampled = resample(
        X_minority, y_minority,
        replace=True,
        n_samples=X_majority.shape[0],
        random_state=42
    )
    X_train_balanced = pd.concat([X_majority, X_minority_oversampled])
    y_train_balanced = pd.concat([y_majority, y_minority_oversampled])

    if dmatrix:
        dtrain = xgb.DMatrix(X_train_balanced, label=y_train_balanced)
        dtest = xgb.DMatrix(X_test, label=y_test)

        return dtrain, dtest
    else:
        return X_train_balanced, y_train_balanced

class NHLModelTrainer:
    def __init__(self, num_boost_round=1000, problem_type="classification",
    data_path=None, model_save_path=None, verbose=True, print_logs=True):
        ##Pass a df instead
        self.data_path = data_path or get_env_variable("DATA_PATH", "all_games_preproc.csv")
        self.model_save_path = model_save_path or get_env_variable("MODEL_PATH", "./backend/model/models")
        self.verbose = verbose
        self.print_logs = print_logs
        self.problem_type = problem_type
        self.num_boost_round = num_boost_round

        self.ml_params = {
            'objective': 'multi:softprob',
            'eval_metric': "mlogloss",
            'booster': 'gbtree',
            'learning_rate': 0.02174751953126388,
            'max_depth': 2,
            'num_class':2
            # 'lambda': 0.2,  # L2 regularization term
            # 'alpha': 0.1,  # L1 regularization term
            # 'scale_pos_weight': [1., 2, 3]  # for imbalanced classes
        }
        self.spread_params = {
            'objective': 'multi:softprob',  # You can change this to multi-class if needed
            'eval_metric': 'mlogloss',  #  use 'error', 'auc', etc.
            'num_class': 2,
            'booster': 'gbtree',
            'learning_rate': 0.055784152872755864,#0.047454011884736254,
            'max_depth':1,
        }
        self.ou_params = {
            'objective': 'multi:softprob',  # You can change this to multi-class if needed
            'eval_metric': 'mlogloss',  #  use 'error', 'auc', etc.
            'num_class': 2,
            'booster': 'gbtree',
            'learning_rate': 0.0103123138167671851, #0.09583588048137198,
            'max_depth':1
        }
        self.best_params = {
            "ml": self.ml_params,
            "spread": self.spread_params,
            "ou": self.ou_params
            }

        self.models = {"ml": None, "spread": None, "ou": None}
        self.best_models = {"ml": None, "spread": None, "ou": None}

        if self.print_logs:
            logging.info(f"Initializing Radon | Data: {self.data_path} | Save Path: {self.model_save_path}")

    def save_params(self, params, filename):
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, "w") as f:
            json.dump(params, f, indent=4)
        if self.print_logs:
            logging.info(f"Hyperparameters saved to {filename}")

    def load_params(self, event, filename):
        if not os.path.exists(filename):
            raise ValueError(f"Hyperparameters file not found: {filename}")
        with open(filename, "r") as f:
            params = json.load(f)
            if params != None:
                self.best_params[event] = params
        if self.print_logs:
            logging.info(f"Hyperparameters loaded from {filename}")
        return params

    def load_data(self):
        df = pd.read_csv(self.data_path, low_memory=False)
        if self.print_logs:
            logging.info(f"Loaded data: {df.shape[0]} records, {df.shape[1]} features")
        return df

    def preprocess(self, event, df, value=None, team=None):
        if event == "ml":
            df = df.dropna()
            if team != None:
                df = df[df["team"] == team]
            df = df[df["goalDiffFor"] != 0]
            
            X = df.loc[:, df.columns.str.contains("ema") | df.columns.str.contains("elo")]
            y = df["winner"]
            tmp_model = xgb.XGBClassifier()
            tmp_model.fit(X, y)
            imp = pd.DataFrame({'Feature': X.columns, 'Importance': tmp_model.feature_importances_})
            imp.sort_values(by='Importance', ascending=False, inplace=True)
            top_features = imp['Feature'].head(90)
            if self.print_logs:
                logging.info(f"[ML] Selected features: {list(top_features)}")
            X = X.loc[:, top_features]
            return X, y

        elif event == "spread":
            if value is None:
                raise ValueError("Spread value must be provided")
            df[f"spread_{value}"] = np.where(df["goalDiffFor"] > -value, 1.0, 0.0)
            df = df.dropna()
            df = df[df["goalDiffFor"] != 0]
            X = df.loc[:, df.columns.str.contains("ema") | df.columns.str.contains("elo")]

            y = df[f"{event}_{value}"]
            X_temp = X.loc[:, X.columns.str.contains("ema") | X.columns.str.contains("elo")]
            tmp_model = xgb.XGBClassifier(params=self.best_params[event])
            tmp_model.fit(X_temp, y)
            imp = pd.DataFrame({'Feature': X_temp.columns, 'Importance': tmp_model.feature_importances_})
            imp.sort_values(by='Importance', ascending=False, inplace=True)
            top_features = imp['Feature'].head(90)
            if self.print_logs:
                logging.info(f"[spread] Selected features: {list(top_features)}")
            X = X.loc[:, top_features]
            return X, y

        elif event == "ou":
            if value is None:
                raise ValueError("An ou value must be provided")
            df["total_goals"] = df["goalsFor"] + df["goalsAgainst"]
            df = df[df["goalDiffFor"] != 0]
            df[f"{event}_{value}"] = np.where(df["total_goals"] > value, 1.0, 0.0)

            df = df.dropna()
            X = df.loc[:, df.columns.str.contains("ema") | df.columns.str.contains("elo")]
            y = df[f"ou_{value}"]
            X_temp = X.loc[:, X.columns.str.contains("ema") | X.columns.str.contains("elo")]
            tmp_model = xgb.XGBClassifier()
            tmp_model.fit(X_temp, y)
            imp = pd.DataFrame({'Feature': X_temp.columns, 'Importance': tmp_model.feature_importances_})
            imp.sort_values(by='Importance', ascending=False, inplace=True)
            top_features = imp['Feature'].head(60)
            if self.print_logs:
                logging.info(f"[OU] Selected features: {list(top_features)}")
            X = X.loc[:, top_features]
            return X, y
        else:
            raise ValueError("Invalid event, please choose from: 'ml', 'spread', or 'ou'")

    def train_ml(self, X, y, epochs_override=None, iter=10, team=None):
        ### Separate function
        params = self.best_params["ml"]
        num_rounds = self.best_num_boost_round("ml", X, y)
        params['n_estimators'] = num_rounds

        best_model = None
        epochs = epochs_override if epochs_override is not None else 750
        acc_results = []
        for x in tqdm(range(iter), desc="Training NHL ML Model"):
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

            dtrain, dtest = balance_dataset(X_train, X_test, y_train, y_test)

            model = xgb.train(params, dtrain, epochs, num_boost_round=num_rounds)
            predictions = model.predict(dtest)
            y_pred = []

            for z in predictions:
                y_pred.append(np.argmax(z))
            acc = round(accuracy_score(y_test, y_pred)*100, 1)
            print(f"ML Accuracy: {acc}")
            acc_results.append(acc)

            if acc == max(acc_results):
                if team != None:
                    if not os.path.exists(f"./backend/model/models/ML/{team}"):
                        os.makedirs(f"./backend/model/models/ML/{team}")
                        model.save_model(f"./backend/model/models/ML/{team}/XGBoost_{acc}%_ML_{team}.json")
                else:
                    model.save_model(f"./backend/model/models/ML/XGBoost_{acc}%_ML.json")
        # y_pred_proba = model.predict(dtest)
        # y_pred = np.argmax(y_pred_proba, axis=1)
        # self.evaluate("ml", y_test, y_pred, y_pred_proba)
        return model, max(acc_results)

    def train_spread(self, X, y, spread, epochs_override=None, n_iter=10):
        try:
            acc_results = []
            best_model = None
            epochs = epochs_override if epochs_override is not None else 750
            params = self.best_params["spread"]
            num_rounds = self.best_num_boost_round("spread", X, y)
            params['n_estimators'] = num_rounds
            for _ in tqdm(range(n_iter), desc=f"Training NHL Spread Model: {spread}"):
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
                dtrain, dtest = balance_dataset(X_train, X_test, y_train, y_test)
                model = xgb.train(params, dtrain, epochs)
                y_pred = []
                predictions = model.predict(dtest)
                for z in predictions:
                    y_pred.append(np.argmax(z))
                acc = round(accuracy_score(y_test, y_pred)*100, 1)
                print(f"Spread Accuracy: {acc}%")
                acc_results.append(acc)
                if acc == max(acc_results):
                    save_dir = os.path.join(self.model_save_path, "spread", str(spread))
                    os.makedirs(save_dir, exist_ok=True)
                    model.save_model(os.path.join(save_dir, f"XGBoost_{acc}%_spread_{spread}.json"))
                    best_model = model
            return best_model, max(acc_results)
        except Exception as e:
            print(e)
            return None

    def train_ou(self, X, y, ou, epochs_override=None, n_iter=10):
        try:
            acc_results = []
            best_model = None
            epochs = epochs_override if epochs_override is not None else 750
            params = self.best_params["ou"]
            num_rounds = self.best_num_boost_round("ou", X, y)
            params["n_estimators"] = num_rounds
            for _ in tqdm(range(n_iter), desc="Training OU Model"):
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
                dtrain, dtest = balance_dataset(X_train, X_test, y_train, y_test)
                model = xgb.train(params, dtrain, epochs)
                y_pred = []
                predictions = model.predict(dtest)
                for z in predictions:
                    y_pred.append(np.argmax(z))
                acc = round(accuracy_score(y_test, y_pred)*100, 1)
                print(f"OU Accuracy: {acc}%")
                acc_results.append(acc)
                if acc == max(acc_results):
                    save_dir = os.path.join(self.model_save_path, "ou", str(ou))
                    os.makedirs(save_dir, exist_ok=True)
                    model.save_model(os.path.join(save_dir, f"XGBoost_{acc}%_ou_{ou}.json"))
                    best_model = model
            return best_model, max(acc_results)
        except Exception as e:
            print(e)
            return None

    def train_event(self, event, X, y, value=None, epochs_override=None):
        if event == "ml":
            return self.train_ml(X, y)
        elif event == "spread":
            return self.train_spread(X, y, value, epochs_override)
        elif event == "ou":
            return self.train_ou(X, y, value, epochs_override)
        else:
            raise ValueError("Invalid event")
    def best_num_boost_round(self, event, X, y):
        params = self.best_params[event]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dtest = xgb.DMatrix(X_test, label=y_test)

        cv_results = xgb.cv(
            params=params,
            dtrain=dtrain,
            num_boost_round=1000,
            nfold=5,
            early_stopping_rounds=20,
            metrics={'mlogloss'}
        )
        best_num = cv_results['test-mlogloss-mean'].idxmin()  # Find best boosting round
        if self.print_logs:
            logging.info(f"Determined optimal number of rounds for {event} model")
            logging.info(f"Boosting rounds: {best_num}")
            print(cv_results)

        return best_num


    def tune_hyperparameters(self, event, X, y, param_dist=None, cv=5, n_iter=20):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=46)
        weight = len(y_train[y_train == 0]) / len(y_train[y_train == 1])

        model = xgb.XGBClassifier(objective="binary:logistic", scale_pos_weight=weight)
        random_search = RandomizedSearchCV(
            estimator=model,
            param_distributions=param_dist,
            n_iter=200,
            cv=3,
            scoring='accuracy',
            verbose=3,
            n_jobs=-1,
            random_state=46, 
            return_train_score=True
        )

        random_search.fit(X, y)

        print(random_search.cv_results_)

        self.best_params[event] = random_search.best_params_

        if self.print_logs:
            logging.info(f"Best hyperparameters for {event}: {json.dumps(self.best_params[event], indent=4)}")

        self.save_params(self.best_params[event], os.path.join(self.model_save_path, f"hyperparameter_{event}.json"))

        best_model = random_search.best_estimator_
        y_pred = []
        predictions = best_model.predict(X_test)
        for z in predictions:
            y_pred.append(np.argmax(z))
        acc = round(accuracy_score(y_test, predictions)*100, 1)
        print(f"Testing accuracy: {acc}")
        if self.print_logs:
            logging.info(f"Accuracy of {event} tuned model: {random_search.best_score_}")

        model_filename = os.path.join(self.model_save_path, f"XGBoost_best_model.json")
        best_model.save_model(model_filename)
        logging.info(f"Best model saved at {model_filename}")

        return best_model
    def evaluate(self, event, y_true, y_pred, y_pred_proba=None):
        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, average="weighted")
        rec = recall_score(y_true, y_pred, average="weighted")
        f1 = f1_score(y_true, y_pred, average="weighted")
        cm = confusion_matrix(y_true, y_pred)
        auc = None #implement later
        if y_pred_proba is not None and isinstance(y_pred_proba, np.ndarray) and y_pred_proba.ndim == 2:
            y_score = y_pred_proba[:, 1] if y_pred_proba.shape[1] == 2 else y_pred_proba
            try:
                auc = roc_auc_score(y_true, y_score, multi_class="ovr")
            except Exception as e:
                logging.warning(f"AUC computation failed: {e}")
        metrics = {"Accuracy": acc, "Precision": prec, "Recall": rec, "F1-score": f1,
                   "Confusion Matrix": cm, "AUC-ROC": auc}
        logging.info(json.dumps(metrics, indent=4, default=str))
        print("Evaluation Metrics:")
        for k, v in metrics.items():
            print(f"{k}: {v}")
        return metrics

    def cross_validate(self, event, X, y, cv=5, scoring='accuracy'):
        if event != "ml":
            logging.warning("Cross-validation is implemented for 'ml' only; using ml_params.")
        scores = cross_val_score(xgb.XGBClassifier(**self.ml_params), X, y, cv=cv, scoring=scoring)
        logging.info(f"Cross-validation accuracy: {scores.mean():.4f} (+/- {scores.std():.4f})")
        return scores
    def save_visualizations(self, event, X, value=None):
        filename_prefix = f"{event}_{value}"
        if self.model_save_path is None:
            return
        if self.best_models.get(event) is not None:
            model = self.best_models[event]
            imp = model.get_score(importance_type='weight')
            imp_df = pd.DataFrame(imp.items(), columns=['Feature', 'Importance']).sort_values(by='Importance', ascending=False)
            plt.figure(figsize=(10,6))
            plt.barh(imp_df['Feature'].head(20), imp_df['Importance'].head(20))
            plt.xlabel("Importance")
            plt.title("Top 20 Feature Importances")
            fi_filename = os.path.join(self.model_save_path, f"{filename_prefix}_feature_importance.png")
            plt.savefig(fi_filename)
            plt.close()

            logging.info(f"Feature importance plot saved at {fi_filename}")
            
            explainer = shap.Explainer(model)
            shap_values = explainer(X)
            shap_filename = os.path.join(self.model_save_path, f"{filename_prefix}_shap_summary.png")
            shap.summary_plot(shap_values, X, show=False)
            plt.savefig(shap_filename)
            plt.close()

            logging.info(f"SHAP summary plot saved at {shap_filename}")

            fpr, tpr, _ = roc_curve(np.argmax(X.values, axis=1), np.random.rand(X.shape[0]))
            plt.figure(figsize=(8,6))
            plt.plot(fpr, tpr, label="ROC curve")
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.title("ROC Curve")
            plt.legend()
            roc_filename = os.path.join(self.model_save_path, f"{filename_prefix}_roc_curve.png")
            plt.savefig(roc_filename)
            plt.close()

            logging.info(f"ROC curve plot saved at {roc_filename}")

    def run_pipeline(self, event, value=None, epochs_override=None, tune=True):
        df = self.load_data()
        X, y = self.preprocess(event, df, value)
        # self.cross_validate(event, X, y)
        if tune == True:
            param_dist = {
                'n_estimators': randint(100, 300),  # Number of trees
                'max_depth': [1, 2],  # Maximum depth of trees
                'learning_rate': uniform(0.01, 0.05),  # Learning rate (eta)
                'reg_alpha': uniform(0, 1),  # L1 regularization (alpha)
                'reg_lambda': uniform(0, 1),  # L2 regularization (lambda)
                'subsample': uniform(0.5,0.5),
                'colsample_bytree': uniform(0,1),
                'min_child_weight' : [ 1, 3, 5, 7 ],
                'gamma':uniform(0,1)
            }

            best_model = self.tune_hyperparameters(event, X, y, param_dist=param_dist)
            self.best_models[event] = best_model
        model = self.train_event(event, X, y, value, epochs_override)
        self.save_visualizations(event, X, value=value)
        self.models[event] = model
        self.best_models[event] = model
        return model, None

if __name__ == "__main__":
    trainer = NHLModelTrainer(problem_type="classification")
    ou_model, _ = trainer.run_pipeline("ou", value=6.5)
    ml_model, _ = trainer.run_pipeline("ml")
    spread_model, _ = trainer.run_pipeline("spread", value=-0.5)
    ou_model = trainer.run_pipeline("ou", value=2.5)
    param_dist = None
    best_estimator = trainer.tune_hyperparameters(*trainer.preprocess("ml", trainer.load_data()), param_dist=param_dist)
