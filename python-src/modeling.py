"""
modeling.py
===========
Step 6 (Baseline + comparison models) and Step 7 (Fairness check).

Trains 11 classifiers (>= 5 required) inside a leakage-safe preprocessing
pipeline, ranks them by F1 on the held-out test set, then runs a fairness
check on the best model by `sex`.

Models:
    Baseline (most-frequent), Logistic Regression, k-NN, Naive Bayes,
    Decision Tree, Random Forest, Extra Trees, Gradient Boosting, AdaBoost,
    SVM, Neural Network (MLP).

Outputs:
    reports/model_results.csv   - metric table
    figures/chart7_model_comparison.png - metric bar chart

Run standalone:  python src/modeling.py
"""
from __future__ import annotations

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier, ExtraTreesClassifier,
    GradientBoostingClassifier, AdaBoostClassifier,
)
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
)

from eda import load_clean

_HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(_HERE)
FIG_DIR = os.path.join(PROJECT_ROOT, "figures")
REPORT_DIR = os.path.join(PROJECT_ROOT, "reports")

RANDOM_STATE = 42

# Features chosen as personal attributes (fnlwgt deliberately excluded).
FEATURE_COLS = [
    "age", "education-num", "hours-per-week",
    "capital-gain", "capital-loss", "sex",
]


def build_models() -> dict:
    return {
        "Baseline": DummyClassifier(strategy="most_frequent"),
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "K-Nearest Neighbors": KNeighborsClassifier(),
        "Naive Bayes": GaussianNB(),
        "Decision Tree": DecisionTreeClassifier(random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(random_state=RANDOM_STATE),
        "Extra Trees": ExtraTreesClassifier(random_state=RANDOM_STATE),
        "Gradient Boosting": GradientBoostingClassifier(random_state=RANDOM_STATE),
        "AdaBoost": AdaBoostClassifier(random_state=RANDOM_STATE),
        "Support Vector Machine": SVC(),
        "Neural Network (MLP)": MLPClassifier(max_iter=500, random_state=RANDOM_STATE),
    }


def prepare(df: pd.DataFrame):
    X = df[FEATURE_COLS].copy()
    y = df["income_binary"]
    # Robust across pandas 2/3 (string columns may be 'str' or 'object' dtype).
    numeric = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    categorical = [c for c in X.columns if c not in numeric]
    # Leakage-safe: preprocessing lives inside the pipeline, fit on train only.
    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), numeric),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
    ])
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    return X_train, X_test, y_train, y_test, preprocessor


def evaluate(df: pd.DataFrame | None = None):
    if df is None:
        df = load_clean()
    X_train, X_test, y_train, y_test, preprocessor = prepare(df)

    rows, fitted = [], {}
    for name, model in build_models().items():
        pipe = Pipeline([("preprocessor", preprocessor), ("model", model)])
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)
        fitted[name] = (pipe, pred)
        rows.append({
            "Model": name,
            "Accuracy": accuracy_score(y_test, pred),
            "Precision": precision_score(y_test, pred, zero_division=0),
            "Recall": recall_score(y_test, pred, zero_division=0),
            "F1 Score": f1_score(y_test, pred, zero_division=0),
        })

    results = (pd.DataFrame(rows)
               .sort_values("F1 Score", ascending=False)
               .reset_index(drop=True))
    return results, fitted, (X_test, y_test)


def fairness_check(fitted, X_test, y_test, best_name: str) -> pd.DataFrame:
    """Step 7 - accuracy/recall of the best model split by sex."""
    _, pred = fitted[best_name]
    audit = X_test.copy()
    audit["actual"] = y_test.values
    audit["predicted"] = pred
    rows = []
    for sex_val in sorted(audit["sex"].unique()):
        mask = audit["sex"] == sex_val
        rows.append({
            "Group": sex_val,
            "n": int(mask.sum()),
            "Accuracy": accuracy_score(audit.loc[mask, "actual"],
                                       audit.loc[mask, "predicted"]),
            "Recall(>50K)": recall_score(audit.loc[mask, "actual"],
                                         audit.loc[mask, "predicted"],
                                         zero_division=0),
        })
    return pd.DataFrame(rows)


def plot_comparison(results: pd.DataFrame) -> str:
    fig, ax = plt.subplots(figsize=(12, 6))
    results.plot(x="Model", y=["Accuracy", "Precision", "Recall", "F1 Score"],
                 kind="bar", ax=ax)
    ax.set_title("Comparison of Machine Learning Algorithms "
                 "(ranked by F1 on the >$50K class)")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1)
    ax.grid(axis="y")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    os.makedirs(FIG_DIR, exist_ok=True)
    path = os.path.join(FIG_DIR, "chart7_model_comparison.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")
    return path


def main():
    os.makedirs(REPORT_DIR, exist_ok=True)
    results, fitted, (X_test, y_test) = evaluate()
    print("=== Model comparison (sorted by F1) ===")
    print(results.to_string(index=False,
          float_format=lambda x: f"{x:.4f}"))

    csv_path = os.path.join(REPORT_DIR, "model_results.csv")
    results.to_csv(csv_path, index=False)
    print(f"\nSaved {csv_path}")

    plot_comparison(results)

    best = results.iloc[0]["Model"]
    print(f"\n=== Fairness check on best model: {best} (by sex) ===")
    fair = fairness_check(fitted, X_test, y_test, best)
    print(fair.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    fair.to_csv(os.path.join(REPORT_DIR, "fairness_by_sex.csv"), index=False)
    return results, fair


if __name__ == "__main__":
    main()
