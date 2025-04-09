import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from scipy.stats import mode
from sklearn.metrics import accuracy_score, roc_auc_score
import xgboost as xgb

if __name__ == "__main__":
    df = pd.read_csv("all_games_preproc.csv")
    df = df.dropna()
    df = df[df["goalDiffFor"] != 0]
    X = df.loc[:, df.columns.str.contains("ema") | df.columns.str.contains("elo")]
    y = df["winner"]
    tmp_model = xgb.XGBClassifier()
    tmp_model.fit(X, y)
    imp = pd.DataFrame({'Feature': X.columns, 'Importance': tmp_model.feature_importances_})
    imp.sort_values(by='Importance', ascending=False, inplace=True)
    top_features = imp['Feature'].head(600)

    X = X.loc[:, top_features]

    X_scaled = (X - X.mean())/X.std()

    pca = PCA(n_components=0.95, svd_solver="auto", whiten=False, random_state=42)
    X_pca = pca.fit_transform(X_scaled)

    print(X_pca)

    gmm = GaussianMixture(n_components=2, covariance_type="tied", random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.2, random_state=42)
    gmm.fit(X_train)

    clusters = gmm.predict(X_train)

    y_true = y.values

    def map_clusters_to_labels(clusters, y_true):
        mapping = {}
        for cluster in np.unique(clusters):
            # Extract the mode correctly
            common_label = mode(y_true[clusters == cluster], keepdims=True).mode[0]
            mapping[cluster] = common_label
        return np.array([mapping[cluster] for cluster in clusters])
    y_train_pred = map_clusters_to_labels(clusters, y_train.values)

    acc = accuracy_score(y_train, y_train_pred)
    print(f"GMM Classification Training Accuracy: {acc:.6f}")

    test_clusters = gmm.predict(X_test)

    y_test_pred = map_clusters_to_labels(test_clusters, y_test.values)

    accuracy = accuracy_score(y_test, y_test_pred)
    print(f"GMM Classification Test Accuracy: {accuracy:.6f}")
