"""
Train and compare churn-prediction models on the Telco Customer Churn dataset.

Usage:
    python src/train.py

Loads data/telco_churn.csv (real Kaggle file if present, else the synthetic
fallback), builds a preprocessing + model pipeline, evaluates Logistic
Regression vs XGBoost with cross-validation, and saves the best model plus
a metrics report.
"""
import json
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    RocCurveDisplay,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

DATA_PATH = "data/telco_churn.csv"
ARTIFACT_DIR = "artifacts"
os.makedirs(ARTIFACT_DIR, exist_ok=True)


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # TotalCharges has blank strings for brand-new customers in the real dataset
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df = df.drop(columns=["customerID"])
    df["Churn"] = (df["Churn"] == "Yes").astype(int)
    return df


def build_preprocessor(df: pd.DataFrame) -> ColumnTransformer:
    numeric_cols = ["tenure", "MonthlyCharges", "TotalCharges"]
    categorical_cols = [c for c in df.columns if c not in numeric_cols + ["Churn"]]

    numeric_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    categorical_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    return ColumnTransformer([
        ("num", numeric_pipe, numeric_cols),
        ("cat", categorical_pipe, categorical_cols),
    ])


def main():
    df = load_data(DATA_PATH)
    X, y = df.drop(columns=["Churn"]), df["Churn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    preprocessor = build_preprocessor(df)

    models = {
        "logistic_regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "xgboost": XGBClassifier(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            scale_pos_weight=(y_train == 0).sum() / (y_train == 1).sum(),
            random_state=42,
        ),
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results = {}
    fitted_pipelines = {}

    for name, model in models.items():
        pipe = Pipeline([("prep", preprocessor), ("model", model)])
        cv_auc = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="roc_auc")
        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_test)
        probs = pipe.predict_proba(X_test)[:, 1]

        results[name] = {
            "cv_roc_auc_mean": round(float(cv_auc.mean()), 4),
            "cv_roc_auc_std": round(float(cv_auc.std()), 4),
            "test_roc_auc": round(float(roc_auc_score(y_test, probs)), 4),
            "test_f1": round(float(f1_score(y_test, preds)), 4),
            "classification_report": classification_report(y_test, preds, output_dict=True),
            "confusion_matrix": confusion_matrix(y_test, preds).tolist(),
        }
        fitted_pipelines[name] = pipe
        print(f"\n=== {name} ===")
        print(f"CV ROC-AUC: {cv_auc.mean():.4f} (+/- {cv_auc.std():.4f})")
        print(f"Test ROC-AUC: {results[name]['test_roc_auc']}")
        print(classification_report(y_test, preds))

    best_name = max(results, key=lambda k: results[k]["test_roc_auc"])
    best_pipe = fitted_pipelines[best_name]
    print(f"\nBest model: {best_name} (test ROC-AUC = {results[best_name]['test_roc_auc']})")

    joblib.dump(best_pipe, os.path.join(ARTIFACT_DIR, "churn_model.joblib"))
    with open(os.path.join(ARTIFACT_DIR, "metrics.json"), "w") as f:
        json.dump({"best_model": best_name, "results": results}, f, indent=2)

    # feature importance for the tree model, useful talking point in interview
    if "xgboost" in fitted_pipelines:
        xgb_pipe = fitted_pipelines["xgboost"]
        feature_names = xgb_pipe.named_steps["prep"].get_feature_names_out()
        importances = xgb_pipe.named_steps["model"].feature_importances_
        top = sorted(zip(feature_names, importances), key=lambda x: -x[1])[:10]
        print("\nTop 10 features (XGBoost):")
        for feat, imp in top:
            print(f"  {feat}: {imp:.4f}")
        with open(os.path.join(ARTIFACT_DIR, "feature_importance.json"), "w") as f:
            json.dump([{"feature": f, "importance": float(i)} for f, i in top], f, indent=2)

    print(f"\nSaved model + metrics to {ARTIFACT_DIR}/")


if __name__ == "__main__":
    main()
