import numpy as np
import pandas as pd
import xgboost as xgb
import os
import json
import logging
import matplotlib.pyplot as plt
import shap
from tqdm import tqdm
from sklearn.decomposition import PCA
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
            'objective': 'binary:logistic',
            'eval_metric': "logloss",
            'booster': 'gbtree',
            'learning_rate': 0.02146131074517304,
            'max_depth': 2
            # 'lambda': 0.2,  # L2 regularization term
            # 'alpha': 0.1,  # L1 regularization term
            # 'scale_pos_weight': [1., 2, 3]  # for imbalanced classes
        }
        self.spread_params = {
            'objective': 'binary:logistic',  # You can change this to multi-class if needed
            'eval_metric': "logloss",
            'booster': 'gbtree',
            'learning_rate': 0.055784152872755864,#0.047454011884736254,
            'max_depth':1,
        }
        self.ou_params = {
            'objective': 'binary:logistic',  # You can change this to multi-class if needed
            'eval_metric': "logloss",
            'booster': 'gbtree',
            "learning_rate": 0.03856269963252939,
            "max_depth": 2,
            "min_child_weight": 3,
            "n_estimators": 123,
            "reg_alpha": 0.3956729259944567,
            "subsample": 0.9240337065369795
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
            if self.print_logs:
                logging.info(f"Hyperparameters file not found: {filename}")
            return None
        with open(filename, "r") as f:
            params = json.load(f)
            if params != None:
                return params
        if self.print_logs:
            logging.info(f"Hyperparameters loaded from {filename}")
        return params

    def load_data(self):
        df = pd.read_csv(self.data_path, low_memory=False)
        if self.print_logs:
            logging.info(f"Loaded data: {df.shape[0]} records, {df.shape[1]} features")
        return df

    def preprocess(self, event, df, value=None, team=None):
        path = os.path.join(self.model_save_path, f"{event}/hyperparameters_ml.json") if event == "ml" else \
        os.path.join(self.model_save_path, f"{event}/{value}/hyperparameters_{event}_{value}.json")

        params = self.load_params(event,path)
        if params != None:
            for k,v in params.items():
                self.best_params[event][k] = v
        self.best_params[event]["objective"] = "binary:logistic"
        df = df[df["gameId"].astype(int) <= 2018020001 ]
        if event == "ml":
            df = df.dropna()
            df = df.drop(columns=df.columns[df.columns.str.contains('winner_seasonal_ema_span')])
            if team != None:
                df = df[df["team"] == team]
            df = df[df["goalDiffFor"] != 0]
            
            X = df.loc[:, df.columns.str.contains("ema") | df.columns.str.contains("elo")]
            y = df["winner"]
            tmp_model = xgb.XGBClassifier()
            tmp_model.fit(X, y)
            imp = pd.DataFrame({'Feature': X.columns, 'Importance': tmp_model.feature_importances_})
            imp.sort_values(by='Importance', ascending=False, inplace=True)
            top_features = imp['Feature'].head(60)
            if self.print_logs:
                logging.info(f"[ML] Selected features: {list(top_features)}")
            X = X.loc[:, top_features]
            # X = (X - X.mean())/X.std()
            # pca = PCA(n_components=0.95, svd_solver="auto", whiten=False, random_state=42)
            # X_pca = pca.fit_transform(X)
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
            top_features = imp['Feature'].head(60)
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
            # X = (X - X.mean())/X.std()
            return X, y
        else:
            raise ValueError("Invalid event, please choose from: 'ml', 'spread', or 'ou'")

    def train_ml(self, X, y, epochs_override=None, iter=10, team=None):
        ### Separate function
        params = self.best_params["ml"]
        num_rounds = self.best_num_boost_round("ml", X, y)

        best_model = None
        epochs = epochs_override if epochs_override is not None else 750
        acc_results = []
        for x in tqdm(range(iter), desc="Training NHL ML Model"):
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
            weight = len(y_train[y_train == 0]) / len(y_train[y_train == 1])

            dtrain = xgb.DMatrix(X_train, label=y_train)
            dtest = xgb.DMatrix(X_test, label=y_test)

            params["scale_pos_weight"] = weight
            model = xgb.train(params, dtrain, epochs, num_boost_round=num_rounds)
            predictions = model.predict(dtest)
            y_pred = (predictions > 0.5).astype(int)
            acc = round(accuracy_score(y_test, y_pred)*100, 2)
            print(f"ML Accuracy: {acc}")
            acc_results.append(acc)

            if acc == max(acc_results):
                if self.print_logs:
                    self.evaluate(y_test, y_pred)
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
            for _ in tqdm(range(n_iter), desc=f"Training NHL Spread Model: {spread}"):
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
                weight = len(y_train[y_train == 0]) / len(y_train[y_train == 1])

                dtrain = xgb.DMatrix(X_train, label=y_train)
                dtest = xgb.DMatrix(X_test, label=y_test)

                params["scale_pos_weight"] = weight
                model = xgb.train(params, dtrain, epochs, num_boost_round=num_rounds)

                predictions = model.predict(dtest)
                y_pred = (predictions > 0.5).astype(int)
                acc = round(accuracy_score(y_test, y_pred)*100, 2)
                print(f"Spread Accuracy: {acc}%")
                acc_results.append(acc)
                if acc == max(acc_results):
                    save_dir = os.path.join(self.model_save_path, "spread", str(spread))
                    os.makedirs(save_dir, exist_ok=True)
                    model.save_model(os.path.join(save_dir, f"XGBoost_{acc}%_spread_{spread}.json"))
                    best_model = model
                    if self.print_logs:
                        self.evaluate(y_test, y_pred)
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
            for _ in tqdm(range(n_iter), desc="Training OU Model"):
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
                
                # weight = len(y_train[y_train == 0]) / len(y_train[y_train == 1])
                dtrain = xgb.DMatrix(X_train, label=y_train)
                dtest = xgb.DMatrix(X_test, label=y_test)
                # params["scale_pos_weight"] = weight
                model = xgb.train(params, dtrain, epochs, num_boost_round=num_rounds)

                predictions = model.predict(dtest)
                y_pred = (predictions > 0.5).astype(int)
                acc = round(accuracy_score(y_test, y_pred)*100, 2)

                print(f"OU Accuracy: {acc}%")
                acc_results.append(acc)
                if acc == max(acc_results):
                    save_dir = os.path.join(self.model_save_path, "ou", str(ou))
                    os.makedirs(save_dir, exist_ok=True)
                    model.save_model(os.path.join(save_dir, f"XGBoost_{acc}%_ou_{ou}.json"))
                    best_model = model
                    if self.print_logs:
                        self.evaluate(y_test, y_pred)
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
            metrics={'logloss'}
        )
        best_num = cv_results['test-logloss-mean'].idxmin()  # Find best boosting round
        if self.print_logs:
            logging.info(f"Determined optimal number of rounds for {event} model")
            logging.info(f"Boosting rounds: {best_num}")
            print(cv_results)

        return best_num


    def tune_hyperparameters(self, event, value, X, y, param_dist=None, cv=5, n_iter=20):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=46)
        weight = len(y_train[y_train == 0]) / len(y_train[y_train == 1])

        model = xgb.XGBClassifier(objective="binary:logistic",
         scale_pos_weight=weight)
        random_search = RandomizedSearchCV(
            estimator=model,
            param_distributions=param_dist,
            n_iter=200,
            cv=3,
            scoring=['neg_log_loss', 'roc_auc','balanced_accuracy'],
            refit='balanced_accuracy',
            verbose=3,
            n_jobs=-1,
            random_state=46, 
            return_train_score=True
        )

        random_search.fit(X_train, y_train)

        print(random_search.cv_results_)
        self.best_params[event] = random_search.best_params_
        self.best_params["objective"] = "binary:logistic"

        if self.print_logs:
            logging.info(f"Best hyperparameters for {event}: {json.dumps(self.best_params[event], indent=4)}")
        hyperparam_file = lambda event, value: f"hyperparameters_{event}_{value}.json" if value != None else f"hyperparameters_{event}.json"
        save_path = lambda event, value: os.path.join(self.model_save_path, event, str(value)) if value != None else os.path.join(self.model_save_path, event)
        self.save_params(self.best_params[event], os.path.join(save_path(event,value), hyperparam_file(event, value)))

        best_model = random_search.best_estimator_
        y_pred = []
        predictions = best_model.predict(X_test)
        for z in predictions:
            y_pred.append(np.argmax(z))
        acc = round(accuracy_score(y_test, predictions)*100, 1)
        print(confusion_matrix(y_test, predictions))
        score = 100 * random_search.best_score_
        print(f"Testing accuracy: {acc}")
        if self.print_logs:
            logging.info(f"Accuracy of {event} tuned model: {score}")

        model_filename = os.path.join(save_path(event, value), f"XGBoost_tuned_{score:.2f}.json")
        best_model.save_model(model_filename)
        logging.info(f"Best model saved at {model_filename}")

        return best_model
    def evaluate(self,y_true, y_pred, y_pred_proba=None):
        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, average="weighted")
        rec = recall_score(y_true, y_pred, average="weighted")
        f1 = f1_score(y_true, y_pred, average="weighted")
        cm = confusion_matrix(y_true, y_pred)

        logloss = None
        roc_auc = roc_auc_score(y_true, y_pred)

        if y_pred_proba is not None:
            try:
                logloss = log_loss(y_true, y_pred_proba)
            except Exception as e:
                logging.warning(f"LogLoss computation failed: {e}")

            try:
                # Binary or multiclass ROC AUC
                roc_auc = roc_auc_score(y_true, y_pred_proba, multi_class="ovr" if len(np.unique(y_true)) > 2 else None)
            except Exception as e:
                logging.warning(f"ROC AUC computation failed: {e}")

        metrics = {
            "Accuracy": acc,
            "Precision": prec,
            "Recall": rec,
            "F1-score": f1,
            "Confusion Matrix": cm,
            "Log Loss": logloss,
            "ROC AUC": roc_auc
        }

        logging.info(json.dumps(metrics, indent=4, default=str))
        print("Evaluation Metrics:")
        for k, v in metrics.items():
            if k == "Confusion Matrix":
                print(f"{k}")
                print(v)
            else:
                print(f"{k}: {v}")

        return metrics
    def cross_validate(self, event, X, y, cv=5, scoring='accuracy'):
        if event != "ml":
            logging.warning("Cross-validation is implemented for 'ml' only; using ml_params.")
        scores = cross_val_score(xgb.XGBClassifier(**self.ml_params), X, y, cv=cv, scoring=scoring)
        logging.info(f"Cross-validation accuracy: {scores.mean():.4f} (+/- {scores.std():.4f})")
        return scores
    def save_visualizations(self, event, X,y, value=None):
        filename_prefix = f"{event}_{value}"
        if self.model_save_path is None:
            return
        if self.best_models.get(event) is not None:
            model = self.best_models[event]
            # imp = model.get_score(importance_type='weight')
            # imp_df = pd.DataFrame(imp.items(), columns=['Feature', 'Importance']).sort_values(by='Importance', ascending=False)
            # plt.figure(figsize=(10,6))
            # plt.barh(imp_df['Feature'].head(20), imp_df['Importance'].head(20))
            # plt.xlabel("Importance")
            # plt.title("Top 20 Feature Importances")
            # fi_filename = os.path.join(self.model_save_path, f"{filename_prefix}_feature_importance.png")
            # plt.savefig(fi_filename)
            # plt.close()

            # logging.info(f"Feature importance plot saved at {fi_filename}")
            
            explainer = shap.Explainer(model)
            shap_values = explainer(X)
            shap_filename = os.path.join(self.model_save_path, f"{filename_prefix}_shap_summary.png")
            shap.summary_plot(shap_values, X, show=False)
            plt.savefig(shap_filename)
            plt.close()

            logging.info(f"SHAP summary plot saved at {shap_filename}")
            
            y_pred = model.predict_proba(X)[:, 1]
            fpr, tpr, _ = roc_curve(y, y_pred)
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
                'max_depth': [1, 2, 3],  # Maximum depth of trees
                'learning_rate': uniform(0.01, 0.03),  # Learning rate (eta)
                'reg_alpha': uniform(0, 1),  # L1 regularization (alpha)
                'reg_lambda': uniform(0, 1),  # L2 regularization (lambda)
                'subsample': uniform(0.5,0.5)
                # 'colsample_bytree': uniform(0,1),
                # 'min_child_weight' : [ 1, 3, 5, 7 ],
                # 'gamma':uniform(0,1),
                # 'colsample_bynode':uniform(0,1),
                # 'colsample_bylevel':uniform(0,1)
            }

            best_model = self.tune_hyperparameters(event, value, X, y, param_dist=param_dist)
            self.best_models[event] = best_model
        model = self.train_event(event, X, y, value, epochs_override)
        self.save_visualizations(event, X,y, value=value)
        self.models[event] = model
        self.best_models[event] = model
        return model, None

if __name__ == "__main__":
    trainer = NHLModelTrainer(problem_type="classification")
    ml_model, _ = trainer.run_pipeline("ml")
    ou_model, _ = trainer.run_pipeline("ou", value=5.0)
    ou_model, _ = trainer.run_pipeline("ou", value=5.5)
    ou_model, _ = trainer.run_pipeline("ou", value=6.0)
    ou_model, _ = trainer.run_pipeline("ou", value=6.5)
    spread_model, _ = trainer.run_pipeline("spread", value=1.5)
    spread_model, _ = trainer.run_pipeline("spread", value=2.5)
    spread_model, _ = trainer.run_pipeline("spread", value=-1.5)
    spread_model, _ = trainer.run_pipeline("spread", value=-2.5)
    spread_model, _ = trainer.run_pipeline("spread", value=1.5)
    ml_model, _ = trainer.run_pipeline("ml")
    param_dist = None
    best_estimator = trainer.tune_hyperparameters(*trainer.preprocess("ml", trainer.load_data()), param_dist=param_dist)
