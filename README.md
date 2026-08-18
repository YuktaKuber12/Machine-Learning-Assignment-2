# ML Assignment 2 — Breast Cancer Classification

## a. Problem Statement

The goal of this project is to build, evaluate, and deploy multiple
classification models that predict whether a breast tumor is **malignant**
or **benign** based on features computed from a digitized image of a fine
needle aspirate (FNA) of a breast mass. This is a binary classification
problem with real-world clinical relevance: an accurate, interpretable
classifier can support early diagnosis. Five models are implemented on the
same dataset, evaluated with six standard classification metrics, and
served through an interactive Streamlit web application.

## b. Dataset Description

- **Name:** Breast Cancer Wisconsin (Diagnostic) Data Set
- **Source:** UCI Machine Learning Repository — https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic
  (also available via `sklearn.datasets.load_breast_cancer`, which mirrors
  the original UCI data)
- **Instances:** 569 (meets the ≥500 minimum)
- **Features:** 30 numeric features (meets the ≥12 minimum) — mean,
  standard error, and "worst" (largest) values of 10 real-valued
  characteristics computed for each cell nucleus: radius, texture,
  perimeter, area, smoothness, compactness, concavity, concave points,
  symmetry, and fractal dimension.
- **Target:** Binary — `0 = malignant`, `1 = benign`
- **Class balance:** 212 malignant, 357 benign
- **Preprocessing:** All features were standardized with `StandardScaler`
  (zero mean, unit variance) before training, fit only on the training
  split to avoid data leakage. Data was split 80/20 (train/test) with
  stratification on the target.

## c. GitHub Repository Link

> ** https://github.com/YuktaKuber12/Machine-Learning-Assignment-2.git
>
> **

## d. Models Used

All 5 models below were trained on the **same** train/test split of the
same dataset and evaluated on the same held-out 20% test set (114 samples).

### Comparison Table

| ML Model Name            | Accuracy | AUC    | Precision | Recall | F1     | MCC    |
| ------------------------ | -------- | ------ | --------- | ------ | ------ | ------ |
| Logistic Regression      | 0.9825   | 0.9954 | 0.9861    | 0.9861 | 0.9861 | 0.9623 |
| Decision Tree            | 0.9123   | 0.9157 | 0.9559    | 0.9028 | 0.9286 | 0.8174 |
| kNN                      | 0.9561   | 0.9788 | 0.9589    | 0.9722 | 0.9655 | 0.9054 |
| Naive Bayes              | 0.9298   | 0.9868 | 0.9444    | 0.9444 | 0.9444 | 0.8492 |
| Random Forest (Ensemble) | 0.9561   | 0.9932 | 0.9589    | 0.9722 | 0.9655 | 0.9054 |

### Observations

| ML Model Name                              | Observation about model performance                                                                                                                                                                                                                      |
| ------------------------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Logistic Regression                        | Best overall performer on this dataset — highest accuracy, AUC, and MCC. The classes are close to linearly separable after standardization, which logistic regression exploits well; it's also the most interpretable model here.                       |
| Decision Tree                              | Weakest performer across every metric. A single unconstrained tree overfits the training split and generalizes worse than the ensemble or linear models; recall in particular drops, meaning it misses more malignant cases.                             |
| kNN                                        | Solid, well-balanced performance after scaling (kNN is distance-based, so standardization was essential). Ties with Random Forest on Accuracy/F1 but has noticeably lower AUC, meaning its confidence scores rank cases less reliably than the ensemble. |
| Naive Bayes                                | Middling accuracy/F1 despite a strong AUC. The Gaussian independence assumption between the 30 (correlated) features hurts hard-label accuracy, but the model's probability ranking is still quite good.                                                 |
| Random Forest (Ensemble)                   | Matches kNN on accuracy/F1 but with the second-highest AUC and MCC overall, confirming ensembling reduces the variance/overfitting seen in the single Decision Tree.                                                                                     |
| **Overall Winner for your dataset?** | **Logistic Regression** — highest Accuracy (0.9825), AUC (0.9954), and MCC (0.9623) of all five models. This is expected for this dataset, since Wisconsin breast cancer features are known to be strongly linearly separable after scaling.      |

## Repository Structure

```
project-folder/
│-- app.py                  # Streamlit app
│-- requirements.txt
│-- README.md
│-- test_data.csv           # held-out test split used in the app/metrics
│-- model/
│   │-- train_models.py     # trains all 5 models, computes metrics, saves artifacts
│   │-- *.joblib            # saved trained models + scaler
│   │-- feature_names.json
│   │-- metrics_summary.csv
```

## How to Run Locally

```bash
pip install -r requirements.txt
python model/train_models.py   # regenerates models + test_data.csv (optional, already included)
streamlit run app.py
```

## Live App

> **`http://localhost:8501/`**

## App Features

- CSV upload of test data
- Model selection dropdown (5 models)
- Live evaluation metrics (Accuracy, AUC, Precision, Recall, F1, MCC)
- Confusion matrix heatmap
- Classification report table
- Full 5-model comparison table from the training run
