"""
Feature Engineering Module for AML Transaction & Anomaly Detection.
Constructs graph/topological metrics, velocity, monetary, and entity-level risk indicators.
"""

import os
import sys
import logging
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder, StandardScaler, RobustScaler

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


class AMLFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Stateful AML Feature Extractor that builds graph topologies,
    financial ratios, temporal indicators, and categorical encodings.
    """

    def __init__(self, top_currencies=None, scale_numerical=True):
        self.top_currencies = top_currencies or [
            "US Dollar", "Euro", "Yuan", "Bitcoin", "Rupee", "Yen", "UK Pound", "Canadian Dollar"
        ]
        self.scale_numerical = scale_numerical
        
        # Learned statistics from training set
        self.from_degree_stats_ = {}
        self.to_degree_stats_ = {}
        self.bank_stats_ = {}
        self.one_hot_encoder_ = None
        self.scaler_ = None
        self.feature_names_ = []

    def _compute_graph_aggregates(self, df: pd.DataFrame):
        """Learn historical graph degree and flow statistics from data."""
        logger.info("Computing network graph aggregates (fan-in, fan-out, counterparty statistics)...")
        
        # Source account statistics (Fan-Out)
        from_grp = df.groupby("from_account")
        self.from_degree_stats_ = {
            "count": from_grp["to_account"].count().to_dict(),
            "unique_dest": from_grp["to_account"].nunique().to_dict(),
            "total_sent": from_grp["amount_paid"].sum().to_dict(),
            "mean_sent": from_grp["amount_paid"].mean().to_dict(),
            "max_sent": from_grp["amount_paid"].max().to_dict(),
        }

        # Destination account statistics (Fan-In)
        to_grp = df.groupby("to_account")
        self.to_degree_stats_ = {
            "count": to_grp["from_account"].count().to_dict(),
            "unique_src": to_grp["from_account"].nunique().to_dict(),
            "total_received": to_grp["amount_received"].sum().to_dict(),
            "mean_received": to_grp["amount_received"].mean().to_dict(),
            "max_received": to_grp["amount_received"].max().to_dict(),
        }

        # Bank activity
        self.bank_stats_ = {
            "from_bank_counts": df["from_bank"].value_counts().to_dict(),
            "to_bank_counts": df["to_bank"].value_counts().to_dict(),
        }

    def fit(self, df: pd.DataFrame, y=None):
        """Fit feature engineer on training transactions."""
        logger.info("Fitting AMLFeatureEngineer on training transactions...")
        self._compute_graph_aggregates(df)

        # Prepare categorical encoder
        cat_df = self._extract_categorical_features(df)
        self.one_hot_encoder_ = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        self.one_hot_encoder_.fit(cat_df)

        # Build feature names
        num_features = self._get_numerical_feature_names()
        cat_feature_names = list(self.one_hot_encoder_.get_feature_names_out(cat_df.columns))
        self.feature_names_ = num_features + cat_feature_names

        if self.scale_numerical:
            num_df = self._extract_numerical_features(df)
            self.scaler_ = RobustScaler()
            self.scaler_.fit(num_df)

        logger.info(f"Fitted AMLFeatureEngineer with {len(self.feature_names_)} features.")
        return self

    def _get_numerical_feature_names(self):
        return [
            # Monetary
            "amount_paid",
            "amount_received",
            "log_amount_paid",
            "log_amount_received",
            "amount_diff",
            "amount_ratio",
            "amount_to_from_mean_ratio",
            # Graph / Network (Topology)
            "from_fan_out_degree",
            "from_unique_destinations",
            "from_fan_ratio",
            "from_total_sent",
            "from_mean_sent",
            "to_fan_in_degree",
            "to_unique_sources",
            "to_fan_ratio",
            "to_total_received",
            "to_mean_received",
            "from_bank_frequency",
            "to_bank_frequency",
            # Behavioral binary flags
            "is_cross_bank",
            "is_cross_currency",
            "is_same_entity",
            "is_same_account",
            "is_cash",
            "is_crypto",
            # Temporal / Velocity
            "hour",
            "minute",
            "time_diff_prev_from",
        ]

    def _extract_categorical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        cat_df = pd.DataFrame(index=df.index)
        cat_df["payment_format"] = df["payment_format"].fillna("Unknown")
        cat_df["from_entity_type"] = df["from_entity_type"].fillna("Unknown")
        cat_df["to_entity_type"] = df["to_entity_type"].fillna("Unknown")
        
        # Payment Currency grouped
        cat_df["payment_currency_grouped"] = df["payment_currency"].apply(
            lambda x: x if x in self.top_currencies else "Other"
        )
        return cat_df

    def _extract_numerical_features(self, df: pd.DataFrame, is_realtime: bool = False, state_cache: dict = None) -> pd.DataFrame:
        res = pd.DataFrame(index=df.index)

        # 1. Monetary features
        res["amount_paid"] = df["amount_paid"].clip(lower=0)
        res["amount_received"] = df["amount_received"].clip(lower=0)
        res["log_amount_paid"] = np.log1p(res["amount_paid"])
        res["log_amount_received"] = np.log1p(res["amount_received"])
        res["amount_diff"] = df["amount_paid"] - df["amount_received"]
        res["amount_ratio"] = res["amount_received"] / (res["amount_paid"] + 1e-4)

        # 2. Graph topology features using learned dictionaries (with safe defaults for unseen accounts)
        res["from_fan_out_degree"] = df["from_account"].map(self.from_degree_stats_["count"]).fillna(1)
        res["from_unique_destinations"] = df["from_account"].map(self.from_degree_stats_["unique_dest"]).fillna(1)
        res["from_fan_ratio"] = res["from_unique_destinations"] / (res["from_fan_out_degree"] + 1e-4)
        res["from_total_sent"] = df["from_account"].map(self.from_degree_stats_["total_sent"]).fillna(res["amount_paid"])
        res["from_mean_sent"] = df["from_account"].map(self.from_degree_stats_["mean_sent"]).fillna(res["amount_paid"])

        res["amount_to_from_mean_ratio"] = res["amount_paid"] / (res["from_mean_sent"] + 1.0)

        res["to_fan_in_degree"] = df["to_account"].map(self.to_degree_stats_["count"]).fillna(1)
        res["to_unique_sources"] = df["to_account"].map(self.to_degree_stats_["unique_src"]).fillna(1)
        res["to_fan_ratio"] = res["to_unique_sources"] / (res["to_fan_in_degree"] + 1e-4)
        res["to_total_received"] = df["to_account"].map(self.to_degree_stats_["total_received"]).fillna(res["amount_received"])
        res["to_mean_received"] = df["to_account"].map(self.to_degree_stats_["mean_received"]).fillna(res["amount_received"])

        res["from_bank_frequency"] = df["from_bank"].map(self.bank_stats_["from_bank_counts"]).fillna(1)
        res["to_bank_frequency"] = df["to_bank"].map(self.bank_stats_["to_bank_counts"]).fillna(1)

        # 3. Behavioral flags
        res["is_cross_bank"] = (df["from_bank"].astype(str) != df["to_bank"].astype(str)).astype(int)
        res["is_cross_currency"] = (df["payment_currency"].astype(str) != df["receiving_currency"].astype(str)).astype(int)
        
        has_entities = (df["from_entity_id"] != "Unknown") & (df["to_entity_id"] != "Unknown")
        res["is_same_entity"] = (has_entities & (df["from_entity_id"] == df["to_entity_id"])).astype(int)
        res["is_same_account"] = (df["from_account"] == df["to_account"]).astype(int)
        
        res["is_cash"] = (df["payment_format"].astype(str).str.lower() == "cash").astype(int)
        res["is_crypto"] = (
            (df["payment_format"].astype(str).str.lower() == "bitcoin") | 
            (df["payment_currency"].astype(str).str.lower() == "bitcoin")
        ).astype(int)

        # 4. Temporal / Velocity features
        ts = pd.to_datetime(df["timestamp"])
        res["hour"] = ts.dt.hour
        res["minute"] = ts.dt.minute

        # Approximate account velocity: delta seconds between transactions from same source
        # Offline Dataset Mode vs Real-Time Mode separation (Section 11 Compliance)
        if is_realtime and state_cache is not None:
            # REAL-TIME MODE: Use stateful persisted cache of previous transaction timestamps
            time_diffs = []
            for _, row in df.iterrows():
                acc = str(row["from_account"]).strip().upper()
                cur_t = pd.to_datetime(row["timestamp"])
                if acc in state_cache:
                    prev_t = state_cache[acc]
                    diff_sec = max(0.0, (cur_t - prev_t).total_seconds())
                else:
                    diff_sec = 86400.0  # Default 24h baseline for first-seen transaction
                time_diffs.append(diff_sec)
                state_cache[acc] = cur_t
            res["time_diff_prev_from"] = time_diffs
        else:
            # OFFLINE DATASET MODE: Strictly sort by timestamp to ensure no future information is used
            df_sorted = df[["from_account", "timestamp"]].copy()
            df_sorted["timestamp"] = ts
            df_sorted["original_order"] = np.arange(len(df))
            df_sorted = df_sorted.sort_values(by=["from_account", "timestamp"], ascending=[True, True])
            df_sorted["prev_time"] = df_sorted.groupby("from_account")["timestamp"].shift(1)
            # Future transactions are never used: shift(1) strictly selects chronologically prior record
            diffs = (df_sorted["timestamp"] - df_sorted["prev_time"]).dt.total_seconds().fillna(86400.0)
            # Clip negative anomalies if any to 0
            df_sorted["time_diff_prev_from"] = np.maximum(0.0, diffs)
            df_sorted = df_sorted.sort_values(by="original_order")
            res["time_diff_prev_from"] = df_sorted["time_diff_prev_from"].values

        res["time_diff_prev_from"] = np.log1p(np.clip(res["time_diff_prev_from"], 0, 86400 * 7))

        return res

    def validate_feature_compatibility(self, df: pd.DataFrame) -> tuple[bool, str]:
        """
        Validate whether the given dataset has the necessary columns to construct
        the 58-dimensional feature vector for the pre-trained Random Forest model.
        """
        required_cols = ["from_account", "to_account", "amount_paid", "timestamp"]
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            return False, f"Supervised model unavailable for this schema: missing required columns {missing}."
        return True, "Dataset schema is compatible with pre-trained feature representation."

    def transform(self, df: pd.DataFrame, is_realtime: bool = False, state_cache: dict = None) -> pd.DataFrame:
        """
        Transform transactions DataFrame into feature matrix DataFrame.
        Supports both offline batch mode and real-time stateful streaming mode.
        """
        num_df = self._extract_numerical_features(df, is_realtime=is_realtime, state_cache=state_cache)
        cat_df = self._extract_categorical_features(df)

        # One hot encode categoricals
        cat_encoded = self.one_hot_encoder_.transform(cat_df)
        cat_cols = list(self.one_hot_encoder_.get_feature_names_out(cat_df.columns))
        cat_encoded_df = pd.DataFrame(cat_encoded, columns=cat_cols, index=df.index)

        # Scale numericals if enabled
        if self.scale_numerical and self.scaler_ is not None:
            scaled_vals = self.scaler_.transform(num_df)
            num_scaled_df = pd.DataFrame(scaled_vals, columns=num_df.columns, index=df.index)
        else:
            num_scaled_df = num_df

        # Concatenate features
        X_df = pd.concat([num_scaled_df, cat_encoded_df], axis=1)
        return X_df

    def get_feature_names(self):
        return self.feature_names_


if __name__ == "__main__":
    from src.clean import clean_and_enrich_data
    df = clean_and_enrich_data()
    fe = AMLFeatureEngineer()
    fe.fit(df)
    X = fe.transform(df.head(100))
    print(f"Engineered feature matrix shape for 100 sample records: {X.shape}")
    print("Sample feature columns:", list(X.columns[:10]))
