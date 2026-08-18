"""
Streamlit app for ML Assignment 2 — Breast Cancer Classification Demo
Author: <YOUR NAME HERE>
"""

import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report,
)

st.set_page_config(page_title="Breast Cancer Classifier Demo", layout="wide")

MODEL_FILES = {
    "Logistic Regression": "model/logistic_regression.joblib",
    "Decision Tree": "model/decision_tree.joblib",
    "kNN": "model/knn.joblib",
    "Naive Bayes": "model/naive_bayes.joblib",
    "Random Forest (Ensemble)": "model/random_forest_ensemble.joblib",
}


@st.cache_resource
def load_scaler():
    return joblib.load("model/scaler.joblib")


@st.cache_resource
def load_model(model_name):
    return joblib.load(MODEL_FILES[model_name])


@st.cache_data
def load_feature_names():
    with open("model/feature_names.json") as f:
        return json.load(f)


def compute_metrics(y_true, y_pred, y_score):
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "AUC": roc_auc_score(y_true, y_score),
        "Precision": precision_score(y_true, y_pred),
        "Recall": recall_score(y_true, y_pred),
        "F1 Score": f1_score(y_true, y_pred),
        "MCC": matthews_corrcoef(y_true, y_pred),
    }


st.title("🩺 Breast Cancer Classification — Model Demo")
st.markdown(
    """
This app demonstrates **5 classification models** trained on the
[UCI Breast Cancer Wisconsin (Diagnostic)](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic)
dataset: Logistic Regression, Decision Tree, kNN, Naive Bayes, and Random
Forest (Ensemble).

Upload the provided `test_data.csv` (or any CSV with the same 30 feature
columns + a `target` column) to see live predictions and evaluation metrics.
"""
)

# --- Sidebar controls ---
st.sidebar.header("⚙️ Controls")
model_name = st.sidebar.selectbox("Choose a model", list(MODEL_FILES.keys()))
uploaded_file = st.sidebar.file_uploader("Upload test data (CSV)", type=["csv"])

feature_names = load_feature_names()
scaler = load_scaler()
model = load_model(model_name)

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
else:
    st.info("👈 Upload `test_data.csv` from the repo root to get started, or drop in your own CSV with the same columns.")
    st.stop()

missing_cols = [c for c in feature_names if c not in df.columns]
if missing_cols:
    st.error(f"Uploaded CSV is missing required feature columns: {missing_cols}")
    st.stop()

has_target = "target" in df.columns

X = df[feature_names]
X_scaled = scaler.transform(X)

y_pred = model.predict(X_scaled)
y_score = model.predict_proba(X_scaled)[:, 1] if hasattr(model, "predict_proba") else y_pred

st.subheader(f"Predictions — {model_name}")
pred_df = df.copy()
pred_df["prediction"] = y_pred
pred_df["prediction_label"] = np.where(y_pred == 1, "Benign", "Malignant")
st.dataframe(pred_df.head(20), use_container_width=True)

if has_target:
    y_true = df["target"]
    metrics = compute_metrics(y_true, y_pred, y_score)

    st.subheader("📊 Evaluation Metrics")
    cols = st.columns(6)
    for col, (k, v) in zip(cols, metrics.items()):
        col.metric(k, f"{v:.4f}")

    st.subheader("🔢 Confusion Matrix")
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(4, 3))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["Malignant", "Benign"],
        yticklabels=["Malignant", "Benign"],
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)

    st.subheader("📄 Classification Report")
    report = classification_report(
        y_true, y_pred, target_names=["Malignant", "Benign"], output_dict=True
    )
    st.dataframe(pd.DataFrame(report).T.round(3), use_container_width=True)
else:
    st.warning("Uploaded CSV has no `target` column, so metrics/confusion matrix can't be computed — predictions only.")

st.markdown("---")
st.subheader("📈 All-Model Comparison (from training run)")
try:
    summary = pd.read_csv("model/metrics_summary.csv", index_col=0)
    st.dataframe(summary.round(4), use_container_width=True)
except FileNotFoundError:
    st.info("Run `model/train_models.py` to generate the comparison table.")
