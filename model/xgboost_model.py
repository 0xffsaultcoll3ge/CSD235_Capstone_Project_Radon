from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectFromModel
from xgboost import XGBClassifier
import pandas as pd
import numpy as np

df = pd.read_csv("all_teams_1.csv", low_memory=False)

goal_cols = []
for col in df.columns:
    if "goal" in col:
        goal_cols.append(col)
df.drop(goal_cols, axis = 1, inplace=True)
df = df.fillna(0)

print(df)

print(df.shape)

X = df.loc[:, df.columns.str.contains('ema')]
y = df["winner"]

print(X.shape, y.shape)

#Pulled this code from an online tutorial, can't find the url

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = XGBClassifier(n_estimators=100, learning_rate=0.1, random_state=42)

#n_estimators -> number of decision trees in the model

model.fit(X_train, y_train)

# Get the feature importance scores
importance_scores = model.feature_importances_

print(importance_scores)

# Select the top 10 most important features
selected_features = importance_scores.argsort()[-10:]

print(selected_features)
print(X_train.iloc[:, selected_features])

# Create a new XGBClassifier with the selected features
selected_model = XGBClassifier(n_estimators=100, learning_rate=0.1, random_state=42)

# Train the new model using only the selected features
selected_model.fit(X_train.iloc[:, selected_features], y_train)

# Evaluate the original model
y_pred = model.predict(X_test)
original_accuracy = accuracy_score(y_test, y_pred)

# Evaluate the model with selected features
y_pred_selected = selected_model.predict(X_test.iloc[:, selected_features])
selected_accuracy = accuracy_score(y_test, y_pred_selected)

print(f"Original Model Accuracy: {original_accuracy:.4f}")
print(f"Selected Features Model Accuracy: {selected_accuracy:.4f}")

# Using scikit-learn's SelectFromModel for feature selection
selector = SelectFromModel(model, prefit=True)
X_train_selected = selector.transform(X_train)
X_test_selected = selector.transform(X_test)

# Train a new model using the selected features
selected_model_pipeline = XGBClassifier(n_estimators=100, learning_rate=0.1, random_state=42)
selected_model_pipeline.fit(X_train_selected, y_train)

# Evaluate the model with selected features using the pipeline
y_pred_pipeline = selected_model_pipeline.predict(X_test_selected)
selected_accuracy_pipeline = accuracy_score(y_test, y_pred_pipeline)

print(f"Selected Features Model Accuracy (Pipeline): {selected_accuracy_pipeline:.4f}")