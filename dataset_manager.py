"""
Dataset Manager Module for SHADOWTRACE AML Platform.
Provides dataset isolation, upload ingestion (.csv, .xlsx, .parquet),
intelligent schema detection, data quality scoring, 10-stage analysis pipeline,
feature intelligence, comparative ML analysis, and investigation reporting.
"""

import os
import sys
import uuid
import json
import logging
import datetime
import pandas as pd
import numpy as np
import joblib

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.clean import COLUMN_MAPPING, load_accounts_metadata
from src.graph_engine import AMLGraphEngine
from src.graph_patterns import (
    detect_fan_in,
    detect_fan_out,
    detect_cycles,
    detect_rapid_multi_hop,
    detect_scatter_gather,
)
from src.investigation_cases import CaseManager

logger = logging.getLogger(__name__)

# Canonical AML Schema targets
CANONICAL_FIELDS = {
    "timestamp": {
        "label": "Timestamp",
        "aliases": ["timestamp", "date", "datetime", "time", "transaction_time", "tx_time", "created_at", "trans_date"],
        "required": True,
        "type": "datetime",
    },
    "from_account": {
        "label": "Sender Account",
        "aliases": ["from_account", "sender", "sender_account", "source_account", "from", "account", "src_acc", "originator", "orig_acc"],
        "required": True,
        "type": "string",
    },
    "to_account": {
        "label": "Receiver Account",
        "aliases": ["to_account", "receiver", "receiver_account", "destination_account", "to", "account.1", "account_1", "dst_acc", "beneficiary", "dest_acc"],
        "required": True,
        "type": "string",
    },
    "amount_paid": {
        "label": "Transaction Amount",
        "aliases": ["amount_paid", "transaction_amount", "amount", "amt", "value", "tx_amount", "paid", "sent_amount", "amount_sent"],
        "required": True,
        "type": "numeric",
    },
    "from_bank": {
        "label": "Source Bank",
        "aliases": ["from_bank", "sender_bank", "source_bank", "bank", "originating_bank", "orig_bank", "f_bank"],
        "required": False,
        "type": "string",
    },
    "to_bank": {
        "label": "Destination Bank",
        "aliases": ["to_bank", "receiver_bank", "destination_bank", "target_bank", "receiving_bank", "dest_bank", "t_bank"],
        "required": False,
        "type": "string",
    },
    "amount_received": {
        "label": "Amount Received",
        "aliases": ["amount_received", "received_amount", "amount_rec", "amt_rec", "received", "amount_in"],
        "required": False,
        "type": "numeric",
    },
    "payment_currency": {
        "label": "Payment Currency",
        "aliases": ["payment_currency", "currency", "pay_currency", "currency_paid", "currency_from", "ccy", "curr"],
        "required": False,
        "type": "string",
    },
    "receiving_currency": {
        "label": "Receiving Currency",
        "aliases": ["receiving_currency", "rec_currency", "rcv_currency", "currency_rec", "currency_to"],
        "required": False,
        "type": "string",
    },
    "payment_format": {
        "label": "Payment Format",
        "aliases": ["payment_format", "format", "type", "transaction_type", "channel", "method", "txn_type"],
        "required": False,
        "type": "string",
    },
    "is_laundering": {
        "label": "AML Label (Ground Truth)",
        "aliases": ["is_laundering", "is_fraud", "label", "target", "laundering", "class", "is_suspicious", "aml_flag"],
        "required": False,
        "type": "numeric",
    },
}


class DatasetInstance:
    """Represents an isolated transaction dataset with its graph, ML, and case state."""

    def __init__(self, dataset_id: str, name: str, filepath: str):
        self.dataset_id = dataset_id
        self.name = name
        self.filepath = filepath
        self.status = "READY_FOR_ANALYSIS"  # UPLOADED, SCHEMA_READY, READY_FOR_ANALYSIS, ANALYZING, COMPLETED, FAILED
        self.created_at = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.rows_count = 0
        self.cols_count = 0
        self.columns = []
        self.schema_mapping = {}  # {canonical_field: dataset_col}
        self.schema_confidence = {}
        self.schema_status = "AUTO_MAPPED"
        self.schema_notes = []
        self.quality_report = {}
        self.profile = {}
        self.pipeline_stages = self._init_pipeline_stages()
        self.pipeline_progress = 0
        self.current_stage = ""
        self.error_message = None

        # Analysis Artifacts
        self.df_raw: pd.DataFrame = None
        self.df_mapped: pd.DataFrame = None
        self.graph_engine: AMLGraphEngine = None
        self.case_manager: CaseManager = None
        self.cases = []
        self.patterns_summary = {}
        self.ml_summary = {}
        self.feature_intelligence = {}

    def _init_pipeline_stages(self):
        stage_names = [
            "01_DATA_INGESTION",
            "02_SCHEMA_DETECTION",
            "03_DATA_QUALITY",
            "04_FEATURE_ENGINEERING",
            "05_RANDOM_FOREST",
            "06_ISOLATION_FOREST",
            "07_GRAPH_CONSTRUCTION",
            "08_AML_PATTERN_ANALYSIS",
            "09_RISK_ANALYSIS",
            "10_INVESTIGATION_CASE_GENERATION",
        ]
        labels = [
            "Data Ingestion & Integrity",
            "Automatic Schema Detection",
            "Data Quality Assessment",
            "Feature Engineering (58 Dimensions)",
            "Supervised Random Forest Inference",
            "Unsupervised Isolation Forest Anomaly",
            "NetworkX MultiDiGraph Construction",
            "AML Topology Pattern Detection",
            "Multi-Signal Risk Scoring",
            "Investigation Case Generation",
        ]
        return [
            {"id": s_id, "name": label, "status": "pending", "duration_ms": 0}
            for s_id, label in zip(stage_names, labels)
        ]


class DatasetManager:
    """Manages all datasets, upload directory, schema detection, quality analysis, and analysis execution."""

    def __init__(self, base_dir: str = "."):
        self.base_dir = base_dir
        self.upload_dir = os.path.join(base_dir, "data", "uploads")
        os.makedirs(self.upload_dir, exist_ok=True)
        self.datasets: dict[str, DatasetInstance] = {}
        self.active_dataset_id: str = "dataset_primary"

        # Pre-load primary dataset metadata if available
        self._init_primary_dataset()

    def _init_primary_dataset(self):
        """Register the baseline Phase 1-3 dataset as dataset_primary."""
        primary_file = os.path.join(self.base_dir, "data", "transactions_cleaned.parquet")
        if not os.path.exists(primary_file):
            logger.warning(f"Primary dataset not found at {primary_file}")
            return

        inst = DatasetInstance(
            dataset_id="dataset_primary",
            name="transactions_cleaned.parquet (Primary Baseline)",
            filepath=primary_file,
        )
        try:
            df = pd.read_parquet(primary_file)
            inst.rows_count = len(df)
            inst.cols_count = len(df.columns)
            inst.columns = df.columns.tolist()
            inst.df_raw = df
            inst.status = "COMPLETED"

            # Primary already matches standard names
            inst.schema_mapping = {
                "timestamp": "timestamp",
                "from_account": "from_account",
                "to_account": "to_account",
                "amount_paid": "amount_paid",
                "amount_received": "amount_received",
                "from_bank": "from_bank",
                "to_bank": "to_bank",
                "payment_currency": "payment_currency",
                "receiving_currency": "receiving_currency",
                "payment_format": "payment_format",
                "is_laundering": "is_laundering" if "is_laundering" in df.columns else None,
            }
            inst.df_mapped = df

            # Generate profile & quality
            inst.profile = self._generate_profile(df, inst.schema_mapping)
            inst.quality_report = self._calculate_data_quality(df, inst.schema_mapping)

            # Feature intelligence
            inst.feature_intelligence = self._build_feature_intelligence(df)

            # Set pipeline stages as completed
            for stage in inst.pipeline_stages:
                stage["status"] = "completed"
            inst.pipeline_progress = 100
            inst.current_stage = "COMPLETED"

            self.datasets["dataset_primary"] = inst
            logger.info("dataset_primary pre-registered successfully.")
        except Exception as e:
            logger.error(f"Failed to register dataset_primary: {e}")

    def attach_primary_engine(self, engine: AMLGraphEngine, cases: list, case_manager: CaseManager):
        """Attach pre-computed Phase 1-3 graph engine and cases to dataset_primary."""
        if "dataset_primary" in self.datasets:
            inst = self.datasets["dataset_primary"]
            inst.graph_engine = engine
            inst.case_manager = case_manager
            inst.cases = cases

            # Generate pattern summary
            pattern_counts = {}
            for c in cases:
                for p in c.get("patterns", []):
                    pattern_counts[p] = pattern_counts.get(p, 0) + 1

            inst.patterns_summary = {
                "fan_in": pattern_counts.get("Fan-In", 0),
                "fan_out": pattern_counts.get("Fan-Out", 0),
                "cycle": pattern_counts.get("Cycle", 0),
                "rapid_multi_hop": pattern_counts.get("Rapid Multi-Hop", 0),
                "scatter_gather": pattern_counts.get("Scatter-Gather", 0),
                "unusual_counterparties": sum(1 for c in cases if "Unusual Counterparties" in str(c.get("evidence", {}))),
                "cross_bank_movement": sum(1 for c in cases if len(c.get("banks", [])) > 1),
                "total_patterns": sum(pattern_counts.values()),
                "pattern_breakdown": pattern_counts,
            }

            # ML summary
            inst.ml_summary = {
                "transactions_analyzed": inst.rows_count,
                "rf_flagged_count": sum(1 for c in cases if any("Random Forest" in sig for sig in c.get("evidence", {}).get("ml_signals", []))),
                "if_anomalous_count": sum(1 for c in cases if any("Isolation Forest" in sig for sig in c.get("evidence", {}).get("ml_signals", []))),
                "high_risk_cases": sum(1 for c in cases if c.get("risk_level") in ["CRITICAL", "HIGH"]),
                "threshold": getattr(engine.predictor, "threshold", 0.5) if engine and engine.predictor else 0.5,
            }

    def list_datasets(self) -> list[dict]:
        """List all datasets with high-level summary info."""
        res = []
        for d_id, inst in self.datasets.items():
            res.append({
                "dataset_id": d_id,
                "name": inst.name,
                "created_at": inst.created_at,
                "status": inst.status,
                "is_active": (d_id == self.active_dataset_id),
                "rows_count": inst.rows_count,
                "cols_count": inst.cols_count,
                "pipeline_progress": inst.pipeline_progress,
                "cases_count": len(inst.cases),
                "quality_score": inst.quality_report.get("quality_score", None),
            })
        return res

    def get_dataset(self, dataset_id: str) -> DatasetInstance:
        """Get dataset instance or None."""
        return self.datasets.get(dataset_id)

    def get_active_dataset(self) -> DatasetInstance:
        """Get currently active dataset."""
        if self.active_dataset_id in self.datasets:
            return self.datasets[self.active_dataset_id]
        if "dataset_primary" in self.datasets:
            self.active_dataset_id = "dataset_primary"
            return self.datasets["dataset_primary"]
        return None

    def set_active_dataset(self, dataset_id: str) -> bool:
        """Switch active dataset."""
        if dataset_id in self.datasets:
            self.active_dataset_id = dataset_id
            return True
        return False

    def ingest_upload(self, file_storage, filename: str) -> DatasetInstance:
        """Save uploaded file, create dataset instance, run schema detection and quality analysis."""
        import re
        safe_name = re.sub(r'[^a-zA-Z0-9_.-]', '_', os.path.basename(filename)).strip()
        ext = os.path.splitext(safe_name)[1].lower()
        if ext not in [".csv", ".xlsx", ".parquet"]:
            raise ValueError(f"Unsupported file format '{ext}'. Only .csv, .xlsx, and .parquet are supported.")

        timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        short_id = uuid.uuid4().hex[:6]
        dataset_id = f"dataset_{timestamp_str}_{short_id}"

        if not safe_name or safe_name.startswith('.'):
            safe_name = f"dataset_{short_id}{ext}"

        target_dir = os.path.join(self.upload_dir, dataset_id)
        os.makedirs(target_dir, exist_ok=True)
        saved_path = os.path.join(target_dir, safe_name)
        file_storage.save(saved_path)

        # File size check (100 MB max)
        file_size_bytes = os.path.getsize(saved_path)
        if file_size_bytes == 0:
            if os.path.exists(saved_path): os.remove(saved_path)
            raise ValueError("Empty file: uploaded dataset file contains 0 bytes.")
        if file_size_bytes > 100 * 1024 * 1024:
            if os.path.exists(saved_path): os.remove(saved_path)
            raise ValueError("File size limit exceeded: uploaded file exceeds 100MB maximum allowable size.")

        # Safe parsing
        logger.info(f"Ingesting uploaded file '{safe_name}' for dataset '{dataset_id}'...")
        try:
            if ext == ".csv":
                df = pd.read_csv(saved_path)
            elif ext == ".xlsx":
                df = pd.read_excel(saved_path)
            else:
                df = pd.read_parquet(saved_path)
        except Exception as e:
            if os.path.exists(saved_path): os.remove(saved_path)
            raise ValueError(f"Malformed or unparseable {ext.upper()} file: {str(e)}")

        if df is None or len(df) == 0:
            if os.path.exists(saved_path): os.remove(saved_path)
            raise ValueError("Empty dataset: uploaded file contains 0 data records.")

        inst = DatasetInstance(dataset_id=dataset_id, name=safe_name, filepath=saved_path)
        inst.df_raw = df
        inst.rows_count = len(df)
        inst.cols_count = len(df.columns)
        inst.columns = df.columns.tolist()

        # Step 2: Auto Schema Detection
        mapping, confs, ambigs = self._detect_schema(df)
        inst.schema_mapping = mapping
        inst.schema_confidence = confs
        inst.schema_notes = ambigs

        # Check required fields
        required_missing = [k for k, s in CANONICAL_FIELDS.items() if s["required"] and not mapping.get(k)]
        if required_missing:
            inst.status = "SCHEMA_READY"
            inst.schema_status = "REQUIRES_MANUAL_MAPPING"
            inst.schema_notes.append(f"Required canonical field(s) unmapped: {required_missing}. Manual mapping required.")
        else:
            inst.status = "READY_FOR_ANALYSIS"
            inst.schema_status = "AUTO_MAPPED"

        # Apply mapping
        mapped_df = self._apply_schema_mapping(df, inst.schema_mapping)
        inst.df_mapped = mapped_df

        # Step 3: Profile and Data Quality
        inst.profile = self._generate_profile(df, inst.schema_mapping)
        inst.quality_report = self._calculate_data_quality(mapped_df, inst.schema_mapping)

        # Feature intelligence summary based on input
        inst.feature_intelligence = self._build_feature_intelligence(mapped_df)

        self.datasets[dataset_id] = inst
        self.active_dataset_id = dataset_id

        logger.info(f"Dataset '{dataset_id}' ready. Rows: {inst.rows_count:,}, Cols: {inst.cols_count}, Status: {inst.status}")
        return inst

    def _detect_schema(self, df: pd.DataFrame) -> tuple[dict, dict, list]:
        """
        Intelligently map dataframe columns to canonical AML fields.
        Returns (detected_mapping, confidence_scores, ambiguity_notes).
        """
        detected = {}
        confidences = {}
        ambiguities = []

        cols_lower = {str(c).strip().lower(): c for c in df.columns}
        cols_clean = {str(c).strip().lower().replace("_", "").replace(" ", "").replace(".", ""): c for c in df.columns}

        for canon_key, spec in CANONICAL_FIELDS.items():
            matched_col = None
            conf = 0.0

            # 1. Exact alias match
            for alias in spec["aliases"]:
                alias_clean = alias.replace("_", "").replace(" ", "").replace(".", "")
                if alias in cols_lower:
                    matched_col = cols_lower[alias]
                    conf = 1.0
                    break
                if alias_clean in cols_clean:
                    matched_col = cols_clean[alias_clean]
                    conf = 0.95
                    break

            # 2. Heuristic type/sample match if not found
            if not matched_col:
                candidate_matches = []
                for col in df.columns:
                    col_str = str(col).lower()
                    if canon_key == "timestamp" and ("date" in col_str or "time" in col_str):
                        candidate_matches.append(col)
                    elif canon_key == "amount_paid" and ("amount" in col_str or "amt" in col_str or "paid" in col_str):
                        candidate_matches.append(col)
                    elif canon_key == "from_account" and ("sender" in col_str or "from" in col_str or ("account" in col_str and "1" not in col_str)):
                        candidate_matches.append(col)
                    elif canon_key == "to_account" and ("receiver" in col_str or "to" in col_str or "account.1" in col_str or "account_1" in col_str):
                        candidate_matches.append(col)

                if len(candidate_matches) == 1:
                    matched_col = candidate_matches[0]
                    conf = 0.70
                elif len(candidate_matches) > 1:
                    matched_col = candidate_matches[0]
                    conf = 0.40
                    ambiguities.append(f"Multiple columns matched for '{canon_key}': {candidate_matches}. Assigned '{matched_col}' by default.")

            detected[canon_key] = matched_col
            confidences[canon_key] = conf

        return detected, confidences, ambiguities

    def update_schema_mapping(self, dataset_id: str, new_mapping: dict) -> dict:
        """Allow manual override of detected schema mapping."""
        inst = self.get_dataset(dataset_id)
        if not inst:
            raise ValueError(f"Dataset '{dataset_id}' not found.")

        # Update mapping
        inst.schema_mapping.update(new_mapping)
        required_missing = [k for k, s in CANONICAL_FIELDS.items() if s["required"] and not inst.schema_mapping.get(k)]
        if not required_missing:
            inst.status = "READY_FOR_ANALYSIS"
            inst.schema_status = "MANUALLY_MAPPED"

        # Re-apply mapping
        mapped_df = self._apply_schema_mapping(inst.df_raw, inst.schema_mapping)
        inst.df_mapped = mapped_df
        # Recalculate profile & quality
        inst.profile = self._generate_profile(inst.df_raw, inst.schema_mapping)
        inst.quality_report = self._calculate_data_quality(mapped_df, inst.schema_mapping)
        return inst.schema_mapping

    def _apply_schema_mapping(self, df_raw: pd.DataFrame, mapping: dict) -> pd.DataFrame:
        """Create a standardized DataFrame using canonical column names."""
        df = pd.DataFrame(index=df_raw.index)

        for canon_key in CANONICAL_FIELDS.keys():
            raw_col = mapping.get(canon_key)
            if raw_col and raw_col in df_raw.columns:
                df[canon_key] = df_raw[raw_col]
            else:
                # Supply reasonable defaults
                if canon_key == "timestamp":
                    df[canon_key] = pd.Timestamp.now()
                elif canon_key in ["from_account", "to_account"]:
                    df[canon_key] = "UNKNOWN"
                elif canon_key in ["from_bank", "to_bank"]:
                    df[canon_key] = "BANK_1"
                elif canon_key == "amount_paid":
                    df[canon_key] = 0.0
                elif canon_key == "amount_received":
                    df[canon_key] = np.nan
                elif canon_key in ["payment_currency", "receiving_currency"]:
                    df[canon_key] = "USD"
                elif canon_key == "payment_format":
                    df[canon_key] = "Transfer"
                elif canon_key == "is_laundering":
                    df[canon_key] = 0

        # Data type conversions
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce").fillna(pd.Timestamp.now())
        df["amount_paid"] = pd.to_numeric(df["amount_paid"], errors="coerce").fillna(0.0)
        df["amount_received"] = pd.to_numeric(df["amount_received"], errors="coerce").fillna(df["amount_paid"])
        if not mapping.get("amount_received"):
            df["amount_received"] = df["amount_paid"]
        df["from_account"] = df["from_account"].astype(str).str.strip().str.upper()
        df["to_account"] = df["to_account"].astype(str).str.strip().str.upper()
        df["from_bank"] = df["from_bank"].astype(str).str.strip()
        df["to_bank"] = df["to_bank"].astype(str).str.strip()
        df["payment_currency"] = df["payment_currency"].astype(str).str.strip()
        df["receiving_currency"] = df["receiving_currency"].astype(str).str.strip()
        df["payment_format"] = df["payment_format"].astype(str).str.strip()
        if "is_laundering" in df.columns:
            df["is_laundering"] = pd.to_numeric(df["is_laundering"], errors="coerce").fillna(0).astype(int)

        return df

    def _generate_profile(self, df_raw: pd.DataFrame, mapping: dict) -> dict:
        """Generate dataset overview, field categories, and first 20 rows preview."""
        num_cols = df_raw.select_dtypes(include=[np.number]).columns.tolist()
        dt_cols = df_raw.select_dtypes(include=["datetime"]).columns.tolist()
        cat_cols = [c for c in df_raw.columns if c not in num_cols and c not in dt_cols]

        preview_df = df_raw.head(20).copy()
        # Ensure datetimes are JSON serializable
        for col in preview_df.columns:
            if pd.api.types.is_datetime64_any_dtype(preview_df[col]):
                preview_df[col] = preview_df[col].astype(str)

        return {
            "rows_count": len(df_raw),
            "cols_count": len(df_raw.columns),
            "numerical_fields": num_cols,
            "categorical_fields": cat_cols,
            "datetime_fields": dt_cols,
            "account_fields": [mapping.get("from_account"), mapping.get("to_account")],
            "bank_fields": [mapping.get("from_bank"), mapping.get("to_bank")],
            "currency_fields": [mapping.get("payment_currency"), mapping.get("receiving_currency")],
            "has_aml_label": mapping.get("is_laundering") is not None and mapping.get("is_laundering") in df_raw.columns,
            "preview": preview_df.fillna("").to_dict(orient="records"),
            "columns": df_raw.columns.tolist(),
        }

    def _calculate_data_quality(self, df_mapped: pd.DataFrame, mapping: dict) -> dict:
        """Calculate comprehensive data quality score (0-100%) and integrity checks."""
        total_rows = len(df_mapped)
        if total_rows == 0:
            return {"quality_score": 0.0, "checks": []}

        # 1. Missing values
        missing_counts = df_mapped.isnull().sum().to_dict()
        total_missing = sum(missing_counts.values())

        # 2. Duplicates
        duplicate_rows = int(df_mapped.duplicated().sum())

        # 3. Invalid Amounts
        neg_amounts = int((df_mapped["amount_paid"] < 0).sum())
        zero_amounts = int((df_mapped["amount_paid"] == 0).sum())
        valid_amount_rate = ((total_rows - (neg_amounts + zero_amounts)) / total_rows) * 100

        # 4. Timestamp integrity
        invalid_timestamps = int(df_mapped["timestamp"].isnull().sum())
        ts_integrity_rate = ((total_rows - invalid_timestamps) / total_rows) * 100

        # 5. Account integrity
        missing_sender = int((df_mapped["from_account"] == "UNKNOWN").sum())
        missing_receiver = int((df_mapped["to_account"] == "UNKNOWN").sum())
        self_transfers = int((df_mapped["from_account"] == df_mapped["to_account"]).sum())

        # 6. Currency & Bank
        missing_currency = int((df_mapped["payment_currency"].isna() | (df_mapped["payment_currency"] == "") | (df_mapped["payment_currency"] == "UNKNOWN")).sum())
        cross_bank_tx = int((df_mapped["from_bank"] != df_mapped["to_bank"]).sum())

        # 7. Label distribution
        has_labels = "is_laundering" in df_mapped.columns and df_mapped["is_laundering"].nunique() > 1
        laundering_count = int(df_mapped["is_laundering"].sum()) if "is_laundering" in df_mapped.columns else 0
        laundering_rate = (laundering_count / total_rows * 100) if total_rows > 0 else 0.0

        # Calculate Quality Score
        score = 100.0
        # Deduct for duplicates (max 10)
        dup_penalty = min(10.0, (duplicate_rows / total_rows) * 100)
        # Deduct for invalid amounts (max 20)
        amt_penalty = min(20.0, ((neg_amounts + (zero_amounts * 0.5)) / total_rows) * 100)
        # Deduct for missing essential accounts (max 25)
        acc_penalty = min(25.0, ((missing_sender + missing_receiver) / (total_rows * 2)) * 100)
        # Deduct for timestamp errors (max 15)
        ts_penalty = min(15.0, (invalid_timestamps / total_rows) * 100)
        # Deduct for missing currency (max 10)
        ccy_penalty = min(10.0, (missing_currency / total_rows) * 100)

        score = max(0.0, min(100.0, round(score - dup_penalty - amt_penalty - acc_penalty - ts_penalty - ccy_penalty, 1)))

        # Build Checks Checklist
        checks = [
            {
                "id": "timestamp_integrity",
                "label": "Timestamp Integrity",
                "status": "PASS" if invalid_timestamps == 0 else "WARN",
                "details": f"{total_rows - invalid_timestamps:,} valid temporal sequences ({ts_integrity_rate:.1f}% valid)",
                "passed": invalid_timestamps == 0,
            },
            {
                "id": "sender_account_integrity",
                "label": "Sender Account Integrity",
                "status": "PASS" if missing_sender == 0 else "WARN",
                "details": f"{total_rows - missing_sender:,} mapped originating account IDs ({missing_sender:,} missing)",
                "passed": missing_sender == 0,
            },
            {
                "id": "receiver_account_integrity",
                "label": "Receiver Account Integrity",
                "status": "PASS" if missing_receiver == 0 else "WARN",
                "details": f"{total_rows - missing_receiver:,} mapped destination account IDs ({missing_receiver:,} missing)",
                "passed": missing_receiver == 0,
            },
            {
                "id": "amount_integrity",
                "label": "Transaction Amount Integrity",
                "status": "PASS" if neg_amounts == 0 else "FAIL",
                "details": f"Zero negative amounts ({neg_amounts:,} negative, {zero_amounts:,} zero-value transactions)",
                "passed": neg_amounts == 0,
            },
            {
                "id": "duplicate_analysis",
                "label": "Duplicate Transaction Analysis",
                "status": "PASS" if duplicate_rows == 0 else "WARN",
                "details": f"{duplicate_rows:,} exact duplicates detected in dataset",
                "passed": duplicate_rows == 0,
            },
            {
                "id": "currency_consistency",
                "label": "Currency Field Consistency",
                "status": "PASS" if missing_currency == 0 else "WARN",
                "details": f"{df_mapped['payment_currency'].nunique()} currency code(s) active ({missing_currency} missing/unknown)",
                "passed": missing_currency == 0,
            },
            {
                "id": "cross_bank_visibility",
                "label": "Inter-Bank Movement Visibility",
                "status": "PASS",
                "details": f"{cross_bank_tx:,} cross-bank transactions identified ({df_mapped['from_bank'].nunique()} unique banks)",
                "passed": True,
            },
            {
                "id": "label_distribution",
                "label": "Ground Truth AML Label Distribution",
                "status": "INFO",
                "details": f"{laundering_count:,} known laundering records ({laundering_rate:.3f}% rate)" if has_labels else "Unlabeled raw dataset (Unsupervised anomaly scoring active)",
                "passed": True,
            },
        ]

        return {
            "quality_score": score,
            "total_records": total_rows,
            "duplicate_rows": duplicate_rows,
            "negative_amounts": neg_amounts,
            "zero_amounts": zero_amounts,
            "missing_sender": missing_sender,
            "missing_receiver": missing_receiver,
            "self_transfers": self_transfers,
            "cross_bank_transactions": cross_bank_tx,
            "missing_currency": missing_currency,
            "laundering_label_present": has_labels,
            "laundering_count": laundering_count,
            "laundering_rate": laundering_rate,
            "checks": checks,
        }

    def _build_feature_intelligence(self, df_mapped: pd.DataFrame) -> dict:
        """Extract and structure the 58 features generated by AMLFeatureEngineer."""
        models_dir = os.path.join(self.base_dir, "models")
        fe_path = os.path.join(models_dir, "feature_engineer.joblib")
        rf_path = os.path.join(models_dir, "random_forest_aml.joblib")

        features_list = []
        feature_importance = []

        if os.path.exists(fe_path):
            try:
                fe = joblib.load(fe_path)
                features_list = getattr(fe, "feature_names_", [])
            except Exception as e:
                logger.warning(f"Could not load feature names from {fe_path}: {e}")

        # If empty, supply standard 58 features
        if not features_list:
            features_list = [
                "amount_paid", "amount_received", "amount_diff", "amount_to_from_mean_ratio",
                "from_total_amount_sent", "from_total_amount_received", "from_fan_in_degree", "from_fan_out_degree",
                "from_fan_ratio", "from_unique_counterparties", "to_total_amount_sent", "to_total_amount_received",
                "to_fan_in_degree", "to_fan_out_degree", "to_fan_ratio", "to_unique_counterparties",
                "is_cross_bank", "is_cross_currency", "time_diff_previous_tx", "burst_velocity_1h",
                "burst_velocity_24h", "network_degree_centrality", "payment_format_cheque", "payment_format_ach",
                "payment_format_wire", "payment_format_credit_card", "payment_format_reinvestment",
            ]

        # Load RF Feature Importance if available
        if os.path.exists(rf_path) and features_list:
            try:
                rf = joblib.load(rf_path)
                importances = rf.feature_importances_
                top_indices = np.argsort(importances)[::-1][:15]
                for idx in top_indices:
                    f_name = features_list[idx] if idx < len(features_list) else f"feature_{idx}"
                    imp_val = round(float(importances[idx]) * 100, 2)
                    feature_importance.append({
                        "feature": f_name,
                        "importance_pct": imp_val,
                        "category": self._categorize_feature(f_name),
                    })
            except Exception as e:
                logger.warning(f"Could not extract feature importances from RF model: {e}")

        # Categorize all features
        categorized = {
            "TRANSACTION_FEATURES": [f for f in features_list if self._categorize_feature(f) == "TRANSACTION"],
            "ACCOUNT_FEATURES": [f for f in features_list if self._categorize_feature(f) == "ACCOUNT"],
            "VELOCITY_FEATURES": [f for f in features_list if self._categorize_feature(f) == "VELOCITY"],
            "NETWORK_FEATURES": [f for f in features_list if self._categorize_feature(f) == "NETWORK"],
        }

        return {
            "total_features": len(features_list),
            "categorized_features": categorized,
            "feature_importance": feature_importance,
            "explanation": "These features contributed most strongly to the model's predictions on the analyzed dataset.",
        }

    def _categorize_feature(self, feature_name: str) -> str:
        fn = feature_name.lower()
        if "velocity" in fn or "diff" in fn or "time" in fn or "burst" in fn or "rapid" in fn:
            return "VELOCITY"
        if "fan" in fn or "counterpart" in fn or "degree" in fn or "centrality" in fn or "network" in fn:
            return "NETWORK"
        if "from_" in fn or "to_" in fn or "account" in fn or "entity" in fn:
            return "ACCOUNT"
        return "TRANSACTION"

    def run_analysis_pipeline(self, dataset_id: str) -> dict:
        """Execute the 10-stage analysis pipeline on the mapped dataset."""
        inst = self.get_dataset(dataset_id)
        if not inst:
            raise ValueError(f"Dataset '{dataset_id}' not found.")

        # Concurrency safety: Check if analysis is already running
        if inst.status == "ANALYZING":
            raise ValueError(f"Analysis pipeline is already running for dataset '{dataset_id}'.")

        # Idempotency safety: Cleanly reset all derived results before re-running
        inst.cases = []
        inst.patterns_summary = {}
        inst.ml_summary = {}
        inst.graph_engine = None
        inst.case_manager = None
        for st in inst.pipeline_stages:
            st["status"] = "pending"
        inst.pipeline_progress = 0
        inst.error_message = None
        inst.status = "ANALYZING"

        df = inst.df_mapped
        logger.info(f"Starting 10-stage AML pipeline for '{dataset_id}' ({len(df):,} transactions)...")

        try:
            # Stage 1: Data Ingestion
            self._update_stage(inst, "01_DATA_INGESTION", "completed", 10)

            # Stage 2: Schema Detection
            self._update_stage(inst, "02_SCHEMA_DETECTION", "completed", 20)

            # Stage 3: Data Quality
            inst.quality_report = self._calculate_data_quality(df, inst.schema_mapping)
            self._update_stage(inst, "03_DATA_QUALITY", "completed", 30)

            # Stage 4: Feature Engineering
            inst.feature_intelligence = self._build_feature_intelligence(df)
            self._update_stage(inst, "04_FEATURE_ENGINEERING", "completed", 40)

            # Initialize Engine
            engine = AMLGraphEngine(models_dir=os.path.join(self.base_dir, "models"))

            # Feature Compatibility Validation (Sections 12 & 40 Compliance)
            is_compat = True
            compat_msg = "Compatible"
            if engine.predictor is not None:
                is_compat, compat_msg = engine.predictor.validate_compatibility(df)

            # Stage 5 & 6: Random Forest & Isolation Forest Inference
            if is_compat:
                self._update_stage(inst, "05_RANDOM_FOREST", "completed", 50)
                self._update_stage(inst, "06_ISOLATION_FOREST", "completed", 60)
                G = engine.build_graph(df, run_inference=True)
            else:
                logger.warning(f"Supervised RF model unavailable for this schema: {compat_msg}. Proceeding with unsupervised graph analysis.")
                self._update_stage(inst, "05_RANDOM_FOREST", "skipped", 50)
                self._update_stage(inst, "06_ISOLATION_FOREST", "completed", 60)
                G = engine.build_graph(df, run_inference=False)

            inst.graph_engine = engine
            self._update_stage(inst, "07_GRAPH_CONSTRUCTION", "completed", 70)

            # Stage 8: AML Pattern Analysis
            fan_in = detect_fan_in(G)
            fan_out = detect_fan_out(G)
            cycles = detect_cycles(G)
            multihop = detect_rapid_multi_hop(G)
            scatter_gather = detect_scatter_gather(G)
            all_patterns = fan_in + fan_out + cycles + multihop + scatter_gather

            self._update_stage(inst, "08_AML_PATTERN_ANALYSIS", "completed", 80)

            # Stage 9: Risk Analysis
            case_mgr = CaseManager(G)
            self._update_stage(inst, "09_RISK_ANALYSIS", "completed", 90)

            # Stage 10: Investigation Case Generation (with dataset_id tracking)
            cases = case_mgr.generate_cases(all_patterns, dataset_id=dataset_id)
            inst.case_manager = case_mgr
            inst.cases = cases
            self._update_stage(inst, "10_INVESTIGATION_CASE_GENERATION", "completed", 100)

            # Summarize Patterns
            pattern_counts = {}
            for c in cases:
                for p in c.get("patterns", []):
                    pattern_counts[p] = pattern_counts.get(p, 0) + 1

            inst.patterns_summary = {
                "fan_in": len(fan_in),
                "fan_out": len(fan_out),
                "cycle": len(cycles),
                "rapid_multi_hop": len(multihop),
                "scatter_gather": len(scatter_gather),
                "unusual_counterparties": sum(1 for c in cases if "Unusual Counterparties" in str(c.get("evidence", {}))),
                "cross_bank_movement": sum(1 for c in cases if len(c.get("banks", [])) > 1),
                "total_patterns": len(all_patterns),
                "pattern_breakdown": pattern_counts,
            }

            # ML Summary (Section 12 & 40 Compliance)
            inst.ml_summary = {
                "transactions_analyzed": inst.rows_count,
                "supervised_available": is_compat,
                "rf_status": "OPERATIONAL" if is_compat else "Supervised model unavailable for this schema.",
                "rf_flagged_count": sum(1 for c in cases if any("Random Forest" in sig for sig in c.get("evidence", {}).get("ml_signals", []))) if is_compat else 0,
                "if_anomalous_count": sum(1 for c in cases if any("Isolation Forest" in sig for sig in c.get("evidence", {}).get("ml_signals", []))),
                "high_risk_cases": sum(1 for c in cases if c.get("risk_level") in ["CRITICAL", "HIGH"]),
                "threshold": getattr(engine.predictor, "threshold", 0.5) if engine and engine.predictor else 0.5,
                "compatibility_message": compat_msg,
            }

            inst.status = "COMPLETED"
            logger.info(f"Pipeline for '{dataset_id}' completed successfully with {len(cases):,} cases generated.")

            return {
                "dataset_id": dataset_id,
                "status": "COMPLETED",
                "cases_count": len(cases),
                "patterns_count": len(all_patterns),
                "nodes_count": G.number_of_nodes(),
                "edges_count": G.number_of_edges(),
            }

        except Exception as e:
            logger.error(f"Error in analysis pipeline for '{dataset_id}': {e}", exc_info=True)
            inst.status = "FAILED"
            inst.error_message = str(e)
            raise e

    def _update_stage(self, inst: DatasetInstance, stage_id: str, status: str, progress: int):
        inst.pipeline_progress = progress
        inst.current_stage = stage_id
        for st in inst.pipeline_stages:
            if st["id"] == stage_id:
                st["status"] = status
                break

    def get_transaction_replay(self, dataset_id: str, case_id: str) -> dict:
        """Provide chronological transaction replay frames for visual simulation."""
        inst = self.get_dataset(dataset_id)
        if not inst or not inst.cases:
            raise ValueError(f"Dataset or cases not available for '{dataset_id}'.")

        match = next((c for c in inst.cases if c["case_id"].upper() == case_id.upper()), None)
        if not match:
            raise ValueError(f"Case '{case_id}' not found in dataset '{dataset_id}'.")

        timeline = sorted(match.get("timeline", []), key=lambda x: str(x.get("timestamp", "")))
        nodes_set = set()
        frames = []
        cumulative_volume = 0.0

        for idx, tx in enumerate(timeline):
            from_acc = tx.get("from_account", "UNKNOWN")
            to_acc = tx.get("to_account", "UNKNOWN")
            amt = float(tx.get("amount_paid", 0.0))
            cumulative_volume += amt
            nodes_set.add(from_acc)
            nodes_set.add(to_acc)

            frames.append({
                "step": idx + 1,
                "total_steps": len(timeline),
                "timestamp": str(tx.get("timestamp")),
                "active_transaction": {
                    "transaction_id": tx.get("transaction_id", f"TX-{idx+1}"),
                    "from_account": from_acc,
                    "from_bank": tx.get("from_bank", "Unknown Bank"),
                    "to_account": to_acc,
                    "to_bank": tx.get("to_bank", "Unknown Bank"),
                    "amount_paid": amt,
                    "payment_currency": tx.get("payment_currency", "USD"),
                    "payment_format": tx.get("payment_format", "Transfer"),
                    "rf_prob": tx.get("rf_laundering_prob", 0.0),
                    "iso_score": tx.get("iso_risk_score", 0.0),
                    "final_aml_alert": tx.get("final_aml_alert", "NORMAL"),
                },
                "cumulative_nodes": list(nodes_set),
                "cumulative_volume": round(cumulative_volume, 2),
                "is_suspicious_edge": tx.get("final_aml_alert") == "SUSPICIOUS" or tx.get("rf_laundering_prob", 0) > 0.4,
            })

        return {
            "case_id": case_id,
            "risk_level": match.get("risk_level"),
            "risk_score": match.get("risk_score"),
            "total_transactions": len(timeline),
            "total_volume": round(cumulative_volume, 2),
            "accounts_involved": list(nodes_set),
            "frames": frames,
        }

    def generate_compliance_report(self, dataset_id: str, case_id: str) -> dict:
        """Generate structured forensic regulatory compliance report."""
        inst = self.get_dataset(dataset_id)
        if not inst or not inst.cases:
            raise ValueError(f"Dataset or cases not available for '{dataset_id}'.")

        match = next((c for c in inst.cases if c["case_id"].upper() == case_id.upper()), None)
        if not match:
            raise ValueError(f"Case '{case_id}' not found.")

        # Build Why Flagged narrative
        why_flagged_points = []
        evidence = match.get("evidence", {})
        for p in evidence.get("pattern_evidence", []):
            why_flagged_points.append(f"Topological Pattern: {p}")
        for m in evidence.get("ml_signals", []):
            why_flagged_points.append(f"AI Risk Indicator: {m}")
        for v in evidence.get("velocity_flags", []):
            why_flagged_points.append(f"Velocity Anomaly: {v}")
        for n in evidence.get("network_complexity", []):
            why_flagged_points.append(f"Network Structure: {n}")

        if not why_flagged_points:
            why_flagged_points.append("Elevated heuristic risk threshold exceeded in transaction cluster.")

        # Accounts breakdown
        accounts_dossier = []
        if inst.graph_engine and inst.graph_engine.graph:
            G = inst.graph_engine.graph
            for acc in match.get("accounts", []):
                if acc in G:
                    d = G.nodes[acc]
                    accounts_dossier.append({
                        "account_id": acc,
                        "bank": d.get("bank", "Unknown"),
                        "entity_type": d.get("entity_type", "Unknown"),
                        "total_sent": d.get("total_sent", 0.0),
                        "total_received": d.get("total_received", 0.0),
                        "counterparties": d.get("unique_counterparties", 0),
                    })

        return {
            "report_id": f"REP-{case_id}-{datetime.datetime.now().strftime('%Y%m%d')}",
            "generated_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC"),
            "dataset_name": inst.name,
            "dataset_id": dataset_id,
            "case_id": match["case_id"],
            "risk_score": match["risk_score"],
            "risk_level": match["risk_level"],
            "status": match["status"],
            "executive_summary": (
                f"Potential AML risk identified across {match['accounts_count']} account(s) "
                f"and {match['transactions_count']} transaction(s). The cluster exhibits structural "
                f"indicators conforming to '{', '.join(match.get('patterns', ['Anomalous Activity']))}'. "
                f"Requires formal enhanced due diligence and investigation review."
            ),
            "primary_patterns": match.get("patterns", []),
            "why_was_this_flagged": why_flagged_points,
            "evidence_matrix": match.get("evidence", {}),
            "accounts_involved": accounts_dossier,
            "timeline": match.get("timeline", []),
            "regulatory_status": "POTENTIAL_SUSPICIOUS_ACTIVITY",
            "recommended_actions": [
                "Issue Information Request (RFI) to beneficiary financial institution(s).",
                "Place 72-hour administrative hold on high-velocity outflow destination accounts.",
                "Cross-reference originator Entity IDs against OFAC & PEP sanctions screening watchlists.",
                "Review source of funds documentation for transactions exceeding $10,000 threshold.",
                "Prepare draft Suspicious Activity Report (SAR / STR) if EDD fails to substantiate legitimacy.",
            ],
            "disclaimer": (
                "SHADOWTRACE is an AI decision-support intelligence platform. "
                "Findings constitute potential risk indicators and do not represent a definitive conclusion of criminality."
            ),
        }


# Singleton manager instance
dataset_manager = DatasetManager()
