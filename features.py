"""
Feature engineering and preprocessing.

Key design choice: every step that LEARNS something from data (median
threshold, one-hot categories) is a scikit-learn transformer. It is fitted on
the training data only and then applied to validation/test data, which
prevents data leakage.
"""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from ingestion_db import create_connection
from logging_setup import setup_logger

logger = setup_logger("features")

TARGET = "Churn"
SERVICE_COLS = ["OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport"]


def clean_raw_data(df):
    """Row-wise cleaning that learns nothing from the data (safe before split)."""
    df = df.copy()
    df.columns = df.columns.str.strip()
    df = df.drop(columns=["customerID"], errors="ignore")
    # TotalCharges is stored as text; blanks are brand-new customers (tenure 0)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0.0)
    df[TARGET] = df[TARGET].str.strip().str.lower().map({"yes": 1, "no": 0}).astype(int)
    logger.info(f"Cleaned raw data: {df.shape}")
    return df


def load_and_split_data(test_size=0.2, random_state=42):
    """Load from the database, clean, and do a stratified train/test split."""
    engine = create_connection()
    try:
        df = pd.read_sql_query("SELECT * FROM telecom_data", engine)
    finally:
        engine.dispose()
    df = clean_raw_data(df)
    X, y = df.drop(columns=[TARGET]), df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    print(f"Rows: {len(df)} | Churn rate: {y.mean():.1%} | "
          f"Train: {len(X_train)} | Test: {len(X_test)}")
    logger.info(f"Split done. Train {X_train.shape}, Test {X_test.shape}")
    return X_train, X_test, y_train, y_test


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Creates domain features. The only learned value (median monthly
    charge) is computed in fit() on training data only."""

    def fit(self, X, y=None):
        self.median_monthly_ = X["MonthlyCharges"].median()
        return self

    def transform(self, X):
        X = X.copy()
        X["tenure_segment"] = pd.cut(
            X["tenure"], bins=[-1, 6, 12, 24, 48, 100],
            labels=["0-6m", "6m-1yr", "1-2yr", "2-4yr", "4+yr"]
        ).astype(str)
        X["is_month_to_month"] = (X["Contract"] == "Month-to-month").astype(int)
        X["has_long_contract"] = X["Contract"].isin(["One year", "Two year"]).astype(int)
        X["service_bundle_score"] = (X[SERVICE_COLS] == "Yes").sum(axis=1)
        X["charges_per_tenure"] = X["TotalCharges"] / (X["tenure"] + 1)
        X["high_monthly_charges"] = (X["MonthlyCharges"] > self.median_monthly_).astype(int)
        X["high_risk_payment"] = (X["PaymentMethod"] == "Electronic check").astype(int)
        X["single_customer"] = ((X["Partner"] == "No") & (X["Dependents"] == "No")).astype(int)
        return X


def build_preprocessor(X_sample):
    """One-hot encode categorical columns, pass numeric columns through.
    Tree models do not need scaling."""
    engineered = FeatureEngineer().fit(X_sample).transform(X_sample)
    cat_cols = engineered.select_dtypes(include=["object", "string"]).columns.tolist()
    num_cols = [c for c in engineered.columns if c not in cat_cols]
    logger.info(f"Categorical: {cat_cols}")
    logger.info(f"Numeric: {num_cols}")
    return ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore", drop="if_binary",
                                  sparse_output=False), cat_cols),
            ("num", "passthrough", num_cols),
        ],
        verbose_feature_names_out=False,
    )


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_and_split_data()
    fe = FeatureEngineer().fit(X_train)
    pre = build_preprocessor(X_train).fit(fe.transform(X_train))
    out = pre.transform(fe.transform(X_train))
    print(f"Engineered + encoded train matrix: {out.shape}")
    print("Features:", list(pre.get_feature_names_out()))
