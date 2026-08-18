"""
train_models.py
----------------
Trains 5 classification models on the Breast Cancer Wisconsin (Diagnostic)
dataset, evaluates each with 6 metrics, saves the trained models + scaler,
saves a train/test split (test split is exported as test_data.csv for the
Streamlit app), and prints a metrics comparison table.

Dataset source: UCI Machine Learning Repository / scikit-learn
https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic
- 569 instances, 30 numeric features, binary target (malignant / benign)
"""

import os
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
)

RANDOM_STATE = 42
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


def load_data():
    data = load_breast_cancer(as_frame=True)
    X = data.data
    y = data.target  # 0 = malignant, 1 = benign
    feature_names = list(X.columns)
    return X, y, feature_names


def build_models():
    return {
        "Logistic Regression": LogisticRegression(max_iter=5000, random_state=RANDOM_STATE),
        "Decision Tree": DecisionTreeClassifier(random_state=RANDOM_STATE),
        "kNN": KNeighborsClassifier(n_neighbors=5),
        "Naive Bayes": GaussianNB(),
        "Random Forest (Ensemble)": RandomForestClassifier(
            n_estimators=200, random_state=RANDOM_STATE
        ),
    }


def evaluate(model, X_test, y_test):
    y_pred = model.predict(X_test)
    if hasattr(model, "predict_proba"):
        y_score = model.predict_proba(X_test)[:, 1]
    else:
        y_score = y_pred

    return {
        "Accuracy": accuracy_score(y_test, y_pred),
        "AUC": roc_auc_score(y_test, y_score),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1": f1_score(y_test, y_pred),
        "MCC": matthews_corrcoef(y_test, y_pred),
    }


def main():
    X, y, feature_names = load_data()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    models = build_models()
    results = {}

    os.makedirs(HERE, exist_ok=True)

    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        metrics = evaluate(model, X_test_scaled, y_test)
        results[name] = metrics

        fname = name.lower().replace(" ", "_").replace("(", "").replace(")", "")
        joblib.dump(model, os.path.join(HERE, f"{fname}.joblib"))

    joblib.dump(scaler, os.path.join(HERE, "scaler.joblib"))
    with open(os.path.join(HERE, "feature_names.json"), "w") as f:
        json.dump(feature_names, f)

    # Save test data (features + true label) for the Streamlit app upload feature
    test_df = X_test.copy()
    test_df["target"] = y_test.values
    test_df.to_csv(os.path.join(ROOT, "test_data.csv"), index=False)

    # Save metrics table
    results_df = pd.DataFrame(results).T
    results_df = results_df[["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]]
    results_df.to_csv(os.path.join(HERE, "metrics_summary.csv"))

    print("\n=== Model Comparison Table ===\n")
    print(results_df.round(4).to_string())
    print("\nSaved models, scaler, and test_data.csv successfully.")


if __name__ == "__main__":
    main()
