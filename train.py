"""
Churn model training pipeline.

Full pipeline (fitted inside every CV fold, so nothing leaks):
    FeatureEngineer -> OneHot encoding -> SMOTE -> SelectFromModel(RandomForest) -> Classifier

Models compared: Random Forest, XGBoost, LightGBM
Tuning: RandomizedSearchCV, stratified 5-fold CV, refit on ROC-AUC
Best model is chosen by CV ROC-AUC; the test set is used once, for final reporting.
"""
import os
import json
import warnings
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, roc_curve, classification_report,
                             confusion_matrix)
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
import xgboost as xgb
import lightgbm as lgb

from features import load_and_split_data, FeatureEngineer, build_preprocessor
from logging_setup import setup_logger

warnings.filterwarnings("ignore")
logger = setup_logger("train")

RANDOM_STATE = 42
N_ITER = 10          # random hyperparameter combinations tried per model
CV_FOLDS = 5
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")


def build_pipeline(model, X_train):
    """Chain every step so SMOTE and feature selection only ever see training folds."""
    selector = SelectFromModel(
        RandomForestClassifier(n_estimators=50, random_state=RANDOM_STATE, n_jobs=1)
    )
    return ImbPipeline(steps=[
        ("features", FeatureEngineer()),
        ("preprocess", build_preprocessor(X_train)),
        ("smote", SMOTE(random_state=RANDOM_STATE)),
        ("select", selector),
        ("model", model),
    ])


def setup_models():
    """Base models and hyperparameter search spaces.
    'select__threshold' tunes how many features SelectFromModel keeps."""
    select_space = {"select__threshold": ["mean", "median", "0.5*mean"]}
    return {
        "RandomForest": (
            RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=1),
            {**select_space,
             "model__n_estimators": [100, 200, 300],
             "model__max_depth": [4, 6, 8, 10],
             "model__min_samples_split": [5, 10, 20],
             "model__min_samples_leaf": [2, 5, 10],
             "model__max_features": ["sqrt", "log2"]},
        ),
        "XGBoost": (
            xgb.XGBClassifier(eval_metric="logloss", random_state=RANDOM_STATE,
                              n_jobs=1, verbosity=0),
            {**select_space,
             "model__n_estimators": [100, 200, 300],
             "model__max_depth": [3, 4, 5, 6],
             "model__learning_rate": [0.01, 0.05, 0.1],
             "model__subsample": [0.8, 1.0],
             "model__colsample_bytree": [0.8, 1.0],
             "model__min_child_weight": [1, 3, 5],
             "model__reg_lambda": [1, 2, 5]},
        ),
        "LightGBM": (
            lgb.LGBMClassifier(random_state=RANDOM_STATE, n_jobs=1, verbosity=-1),
            {**select_space,
             "model__n_estimators": [100, 200, 300],
             "model__max_depth": [3, 5, 7, -1],
             "model__learning_rate": [0.01, 0.05, 0.1],
             "model__num_leaves": [15, 31, 50],
             "model__subsample": [0.8, 1.0],
             "model__subsample_freq": [1],
             "model__colsample_bytree": [0.8, 1.0],
             "model__min_child_samples": [10, 20, 40]},
        ),
    }


def evaluate(model, X_test, y_test):
    """Test-set metrics at the default 0.5 threshold."""
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "classification_report": classification_report(y_test, y_pred),
    }


def selected_feature_names(pipeline):
    """Names of the features kept by SelectFromModel."""
    all_names = pipeline.named_steps["preprocess"].get_feature_names_out()
    mask = pipeline.named_steps["select"].get_support()
    return [n for n, keep in zip(all_names, mask) if keep]


def train_all(X_train, y_train, X_test, y_test):
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    scoring = {"roc_auc": "roc_auc", "recall": "recall",
               "precision": "precision", "f1": "f1"}
    results = {}

    for name, (model, params) in setup_models().items():
        print(f"\n{'=' * 60}\nTuning {name} ({N_ITER} combinations x {CV_FOLDS} folds)...")
        search = RandomizedSearchCV(
            build_pipeline(model, X_train), params, n_iter=N_ITER,
            scoring=scoring, refit="roc_auc", cv=cv,
            random_state=RANDOM_STATE, n_jobs=-1, verbose=0,
        )
        search.fit(X_train, y_train)
        best = search.best_estimator_
        i = search.best_index_
        cv_scores = {m: float(search.cv_results_[f"mean_test_{m}"][i]) for m in scoring}
        test_scores = evaluate(best, X_test, y_test)
        features = selected_feature_names(best)

        results[name] = {"pipeline": best, "best_params": search.best_params_,
                         "cv": cv_scores, "test": test_scores, "features": features}

        print(f"CV   ROC-AUC {cv_scores['roc_auc']:.4f} | Recall {cv_scores['recall']:.4f} | "
              f"Precision {cv_scores['precision']:.4f} | F1 {cv_scores['f1']:.4f}")
        print(f"TEST ROC-AUC {test_scores['roc_auc']:.4f} | Recall {test_scores['recall']:.4f} | "
              f"Precision {test_scores['precision']:.4f} | F1 {test_scores['f1']:.4f} | "
              f"Accuracy {test_scores['accuracy']:.4f}")
        print(f"Features kept by SelectFromModel: {len(features)}")
        logger.info(f"{name}: CV {cv_scores} | TEST roc_auc {test_scores['roc_auc']:.4f}")
    return results


def save_outputs(results, best_name, X_test, y_test):
    os.makedirs("models", exist_ok=True)
    os.makedirs("reports", exist_ok=True)

    # Models
    for name, r in results.items():
        joblib.dump(r["pipeline"], f"models/{name}_pipeline.joblib")
    joblib.dump(results[best_name]["pipeline"], "models/best_model.joblib")

    # Results table (CSV + markdown for the README)
    rows = []
    for name, r in results.items():
        t = r["test"]
        rows.append({"Model": name, "CV ROC-AUC": r["cv"]["roc_auc"],
                     "Test ROC-AUC": t["roc_auc"], "Precision": t["precision"],
                     "Recall": t["recall"], "F1": t["f1"], "Accuracy": t["accuracy"],
                     "Features kept": len(r["features"])})
    table = pd.DataFrame(rows).round(4)
    table.to_csv("reports/results.csv", index=False)
    with open("reports/results.md", "w") as f:
        f.write(table.to_markdown(index=False))

    # Metadata
    meta = {"timestamp": TIMESTAMP, "best_model": best_name,
            "selection_rule": "highest mean CV ROC-AUC",
            "models": {n: {"best_params": {k: str(v) for k, v in r["best_params"].items()},
                           "cv": r["cv"],
                           "test": {k: v for k, v in r["test"].items()
                                    if k != "classification_report"},
                           "selected_features": r["features"]}
                       for n, r in results.items()}}
    with open("reports/metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    # Text report
    with open("reports/training_report.txt", "w") as f:
        f.write(f"CHURN MODEL TRAINING REPORT ({TIMESTAMP})\n{'=' * 60}\n")
        f.write(table.to_string(index=False) + "\n\n")
        f.write(f"Best model (by CV ROC-AUC): {best_name}\n\n")
        for name, r in results.items():
            f.write(f"{name}\n{'-' * 40}\nBest params: {r['best_params']}\n")
            f.write(f"Selected features ({len(r['features'])}): {r['features']}\n")
            f.write(f"Confusion matrix [[TN FP] [FN TP]]: {r['test']['confusion_matrix']}\n")
            f.write(r["test"]["classification_report"] + "\n")

    # ROC curves
    plt.figure(figsize=(7, 6))
    for name, r in results.items():
        proba = r["pipeline"].predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, proba)
        plt.plot(fpr, tpr, label=f"{name} (AUC = {r['test']['roc_auc']:.3f})")
    plt.plot([0, 1], [0, 1], "k--", label="Random (AUC = 0.500)")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate (Recall)")
    plt.title("ROC Curves - Test Set")
    plt.legend()
    plt.tight_layout()
    plt.savefig("reports/roc_curves.png", dpi=150)
    plt.close()

    # Feature importance of best model (on selected features)
    best_pipe = results[best_name]["pipeline"]
    importances = best_pipe.named_steps["model"].feature_importances_
    imp = pd.Series(importances, index=results[best_name]["features"]).sort_values()
    plt.figure(figsize=(8, max(4, 0.35 * len(imp))))
    imp.plot(kind="barh")
    plt.title(f"Feature Importance - {best_name}")
    plt.tight_layout()
    plt.savefig("reports/feature_importance.png", dpi=150)
    plt.close()

    print(f"\nSaved: models/, reports/results.md, reports/roc_curves.png, "
          f"reports/feature_importance.png, reports/training_report.txt")
    return table


def main():
    print("=" * 60 + "\nCHURN PREDICTION TRAINING PIPELINE\n" + "=" * 60)
    X_train, X_test, y_train, y_test = load_and_split_data(random_state=RANDOM_STATE)
    results = train_all(X_train, y_train, X_test, y_test)
    best_name = max(results, key=lambda n: results[n]["cv"]["roc_auc"])
    table = save_outputs(results, best_name, X_test, y_test)
    print("\n" + "=" * 60 + "\nFINAL RESULTS\n" + "=" * 60)
    print(table.to_string(index=False))
    print(f"\nBest model (chosen by CV ROC-AUC, not test): {best_name}")
    print(f"Its selected features: {results[best_name]['features']}")


if __name__ == "__main__":
    main()
