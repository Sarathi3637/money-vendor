"""
Inference Module for AML Detection Pipeline.
Loads saved models to perform dual-model inference on new transactions.
"""

import os
import sys
import json
import logging
import joblib
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.clean import COLUMN_MAPPING, load_accounts_metadata

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


class AMLPredictor:
    """Production Inference engine wrapping Random Forest and Isolation Forest."""

    def __init__(self, models_dir: str = "models", accounts_path: str = "data/HI-Small_accounts.csv"):
        self.models_dir = models_dir
        self.accounts_path = accounts_path
        self._load_models()

    def _load_models(self):
        logger.info(f"Loading models from '{self.models_dir}'...")
        self.rf_model = joblib.load(os.path.join(self.models_dir, "random_forest_aml.joblib"))
        self.iso_model = joblib.load(os.path.join(self.models_dir, "isolation_forest_aml.joblib"))
        self.feature_engineer = joblib.load(os.path.join(self.models_dir, "feature_engineer.joblib"))

        # Load optimal threshold
        eval_json_path = os.path.join(self.models_dir, "evaluation_metrics.json")
        self.threshold = 0.5
        if os.path.exists(eval_json_path):
            try:
                with open(eval_json_path, "r") as f:
                    data = json.load(f)
                    self.threshold = data.get("random_forest", {}).get("optimal_threshold", {}).get("threshold", 0.5)
            except Exception:
                pass
        
        # Load account metadata if available
        if os.path.exists(self.accounts_path):
            self.df_acc = load_accounts_metadata(self.accounts_path)
        else:
            self.df_acc = None

    def validate_compatibility(self, df: pd.DataFrame) -> tuple[bool, str]:
        """
        Validate whether the given dataset has required fields to construct the feature matrix.
        """
        if self.feature_engineer and hasattr(self.feature_engineer, "validate_feature_compatibility"):
            return self.feature_engineer.validate_feature_compatibility(df)
        required_cols = ["from_account", "to_account", "amount_paid", "timestamp"]
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            return False, f"Supervised model unavailable for this schema: missing required columns {missing}."
        return True, "Dataset schema is compatible with pre-trained feature representation."

    def predict(self, df_input: pd.DataFrame, is_realtime: bool = False, state_cache: dict = None) -> pd.DataFrame:
        """
        Run inference on a DataFrame of raw transactions.
        Supports both offline batch inference and stateful real-time mode.
        """
        is_compat, msg = self.validate_compatibility(df_input)
        if not is_compat:
            raise ValueError(msg)

        df = df_input.copy()

        # Rename standard columns if present
        for col_raw, col_std in COLUMN_MAPPING.items():
            if col_raw in df.columns and col_std not in df.columns:
                df = df.rename(columns={col_raw: col_std})

        # Standardize strings
        df["from_account"] = df["from_account"].astype(str).str.strip().str.upper()
        df["to_account"] = df["to_account"].astype(str).str.strip().str.upper()
        df["from_bank"] = df["from_bank"].astype(str).str.strip()
        df["to_bank"] = df["to_bank"].astype(str).str.strip()
        df["payment_currency"] = df["payment_currency"].astype(str).str.strip()
        df["receiving_currency"] = df["receiving_currency"].astype(str).str.strip()
        df["payment_format"] = df["payment_format"].astype(str).str.strip()
        df["amount_received"] = pd.to_numeric(df["amount_received"], errors="coerce").fillna(0.0)
        df["amount_paid"] = pd.to_numeric(df["amount_paid"], errors="coerce").fillna(0.0)
        df["timestamp"] = pd.to_datetime(df["timestamp"])

        # Merge account metadata
        if self.df_acc is not None:
            if "from_entity_id" not in df.columns:
                df = df.merge(
                    self.df_acc[["Account Number", "Entity ID", "entity_type"]].rename(
                        columns={"Account Number": "from_account", "Entity ID": "from_entity_id", "entity_type": "from_entity_type"}
                    ),
                    on="from_account",
                    how="left",
                )
            if "to_entity_id" not in df.columns:
                df = df.merge(
                    self.df_acc[["Account Number", "Entity ID", "entity_type"]].rename(
                        columns={"Account Number": "to_account", "Entity ID": "to_entity_id", "entity_type": "to_entity_type"}
                    ),
                    on="to_account",
                    how="left",
                )

        df["from_entity_type"] = df.get("from_entity_type", pd.Series("Unknown", index=df.index)).fillna("Unknown")
        df["to_entity_type"] = df.get("to_entity_type", pd.Series("Unknown", index=df.index)).fillna("Unknown")
        df["from_entity_id"] = df.get("from_entity_id", pd.Series("Unknown", index=df.index)).fillna("Unknown")
        df["to_entity_id"] = df.get("to_entity_id", pd.Series("Unknown", index=df.index)).fillna("Unknown")

        # Transform features
        try:
            X = self.feature_engineer.transform(df, is_realtime=is_realtime, state_cache=state_cache)
        except TypeError:
            X = self.feature_engineer.transform(df)

        # Inferences
        rf_probs = self.rf_model.predict_proba(X)[:, 1]
        rf_flags = (rf_probs >= self.threshold).astype(int)

        iso_scores = self.iso_model.decision_function(X)
        iso_risk = -iso_scores
        iso_flags = (self.iso_model.predict(X) == -1).astype(int)

        output_df = df[["timestamp", "from_account", "to_account", "amount_paid", "payment_format"]].copy()
        output_df["rf_laundering_prob"] = np.round(rf_probs, 4)
        output_df["rf_is_laundering_flag"] = rf_flags
        output_df["iso_risk_score"] = np.round(iso_risk, 4)
        output_df["iso_anomaly_flag"] = iso_flags
        output_df["final_aml_alert"] = np.where(
            (rf_flags == 1) | (iso_flags == 1), "FLAGGED", "NORMAL"
        )
        return output_df


if __name__ == "__main__":
    predictor = AMLPredictor()
    # Test on a few sample transactions
    test_sample = pd.DataFrame([{
        "Timestamp": "2022-09-01 00:21:00",
        "From Bank": "70",
        "Account": "100428660",
        "To Bank": "1124",
        "Account.1": "800825340",
        "Amount Received": 389769.39,
        "Receiving Currency": "US Dollar",
        "Amount Paid": 389769.39,
        "Payment Currency": "US Dollar",
        "Payment Format": "Cheque",
    }])
    result = predictor.predict(test_sample)
    print("\n--- Sample Transaction Inference ---")
    print(result.to_dict(orient="records"))
