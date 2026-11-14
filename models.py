"""
Model Training Module for AML Detection Pipeline.
Trains Random Forest Classifier (Supervised) and Isolation Forest (Unsupervised) independently.
"""

import os
import sys
import logging
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, IsolationForest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.clean import clean_and_enrich_data
from src.features import AMLFeatureEngineer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def train_aml_models(
    data_path: str = "data/transactions_cleaned.parquet",
    models_dir: str = "models",
    test_size: float = 0.20,
    random_state: int = 42,
):
    """
    Train Random Forest and Isolation Forest independently on Dataset 1.
    """
    os.makedirs(models_dir, exist_ok=True)

    # 1. Load cleaned transactions
    if os.path.exists(data_path):
        logger.info(f"Loading cleaned data from {data_path}...")
        df = pd.read_parquet(data_path)
    else:
        logger.info("Cleaned cache not found. Running clean_and_enrich_data...")
        df = clean_and_enrich_data()

    y = df["is_laundering"].values
    logger.info(f"Loaded {len(df):,} transactions. Target distribution: {pd.Series(y).value_counts().to_dict()}")

    # 2. Stratified Train / Test split
    logger.info(f"Splitting dataset (test_size={test_size}, stratify=y, random_state={random_state})...")
    df_train, df_test, y_train, y_test = train_test_split(
        df, y, test_size=test_size, stratify=y, random_state=random_state
    )
    logger.info(f"Train set: {len(df_train):,} (Positives: {sum(y_train)}), Test set: {len(df_test):,} (Positives: {sum(y_test)})")

    # 3. Fit Feature Engineer on Train set ONLY (Zero Data Leakage)
    logger.info("Fitting AMLFeatureEngineer on training set...")
    feature_engineer = AMLFeatureEngineer(scale_numerical=True)
    feature_engineer.fit(df_train)

    X_train = feature_engineer.transform(df_train)
    X_test = feature_engineer.transform(df_test)
    logger.info(f"Feature matrix generated. X_train shape: {X_train.shape}, X_test shape: {X_test.shape}")

    # 4. Train Random Forest (Supervised, Independent)
    logger.info("Training Random Forest Classifier (independent supervised)...")
    rf_model = RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight="balanced_subsample",
        random_state=random_state,
        n_jobs=-1,
    )
    rf_model.fit(X_train, y_train)
    logger.info("Random Forest training completed.")

    # 5. Train Isolation Forest (Unsupervised Anomaly Detector, Independent)
    logger.info("Training Isolation Forest (independent unsupervised anomaly detector)...")
    iso_model = IsolationForest(
        n_estimators=200,
        max_samples="auto",
        contamination=0.001,  # Estimated anomaly proportion in AML
        random_state=random_state,
        n_jobs=-1,
    )
    # Isolation Forest trains strictly unsupervised on X_train (no labels)
    iso_model.fit(X_train)
    logger.info("Isolation Forest training completed.")

    # 6. Save Models and Pipeline Artifacts Separately
    logger.info(f"Saving models and feature pipeline to '{models_dir}'...")
    rf_path = os.path.join(models_dir, "random_forest_aml.joblib")
    iso_path = os.path.join(models_dir, "isolation_forest_aml.joblib")
    fe_path = os.path.join(models_dir, "feature_engineer.joblib")

    joblib.dump(rf_model, rf_path)
    joblib.dump(iso_model, iso_path)
    joblib.dump(feature_engineer, fe_path)

    # Save split data for evaluation module
    splits = {
        "df_train": df_train,
        "df_test": df_test,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
    }
    splits_path = os.path.join(models_dir, "split_data.joblib")
    joblib.dump(splits, splits_path)

    logger.info("Saved all model artifacts:")
    logger.info(f"  - Random Forest:    {rf_path}")
    logger.info(f"  - Isolation Forest: {iso_path}")
    logger.info(f"  - Feature Pipeline: {fe_path}")
    logger.info(f"  - Split Data:       {splits_path}")

    return {
        "rf_model": rf_model,
        "iso_model": iso_model,
        "feature_engineer": feature_engineer,
        "splits": splits,
    }


if __name__ == "__main__":
    train_aml_models()
