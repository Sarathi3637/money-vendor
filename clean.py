"""
Data Loading, Cleaning, and Entity Enrichment Module for AML Detection Pipeline.
"""

import os
import re
import logging
import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Standard schema mapping
COLUMN_MAPPING = {
    "Timestamp": "timestamp",
    "From Bank": "from_bank",
    "Account": "from_account",
    "To Bank": "to_bank",
    "Account.1": "to_account",
    "Amount Received": "amount_received",
    "Receiving Currency": "receiving_currency",
    "Amount Paid": "amount_paid",
    "Payment Currency": "payment_currency",
    "Payment Format": "payment_format",
    "Is Laundering": "is_laundering",
}


def extract_entity_type(name: str) -> str:
    """Extract general entity class from Entity Name (e.g. 'Corporation #123' -> 'Corporation')."""
    if pd.isna(name):
        return "Unknown"
    s = str(name).strip()
    match = re.match(r"^([A-Za-z\s]+)", s)
    if match:
        extracted = match.group(1).strip()
        if "#" in extracted:
            extracted = extracted.split("#")[0].strip()
        return extracted if extracted else "Unknown"
    return "Unknown"


def load_accounts_metadata(accounts_path: str = "data/HI-Small_accounts.csv") -> pd.DataFrame:
    """Load and process account-to-entity metadata."""
    logger.info(f"Loading accounts metadata from {accounts_path}...")
    df_acc = pd.read_csv(
        accounts_path,
        usecols=["Bank ID", "Account Number", "Entity ID", "Entity Name"],
        dtype={"Account Number": str, "Bank ID": str, "Entity ID": str, "Entity Name": str},
    )
    df_acc["Account Number"] = df_acc["Account Number"].str.strip().str.upper()
    df_acc["entity_type"] = df_acc["Entity Name"].apply(extract_entity_type)
    
    # Drop duplicates if any on Account Number
    df_acc = df_acc.drop_duplicates(subset=["Account Number"])
    logger.info(f"Loaded {len(df_acc):,} unique account records.")
    return df_acc


def load_transactions(data_path: str = "data/Dataset 1.xlsx", cache_path: str = "data/transactions_cleaned.parquet") -> pd.DataFrame:
    """Load transactions from Excel or cache."""
    if cache_path and os.path.exists(cache_path):
        logger.info(f"Loading cached cleaned transactions from {cache_path}...")
        try:
            return pd.read_parquet(cache_path)
        except Exception as e:
            logger.warning(f"Could not read parquet cache ({e}), reading raw excel...")

    logger.info(f"Reading raw Excel from {data_path} (this may take a few moments)...")
    df = pd.read_excel(data_path)
    return df


def clean_and_enrich_data(
    data_path: str = "data/Dataset 1.xlsx",
    accounts_path: str = "data/HI-Small_accounts.csv",
    save_cache: bool = True,
    cache_path: str = "data/transactions_cleaned.parquet"
) -> pd.DataFrame:
    """
    Complete cleaning, schema standardization, and account-entity enrichment.
    """
    if os.path.exists(cache_path):
        logger.info(f"Cleaned dataset already cached at {cache_path}. Reading directly.")
        try:
            return pd.read_parquet(cache_path)
        except Exception as e:
            logger.warning(f"Cache load failed: {e}. Reprocessing from raw files.")

    df = load_transactions(data_path, cache_path=None)
    
    # 1. Standardize column names
    df = df.rename(columns=COLUMN_MAPPING)

    # 2. String cleaning and type conversion
    df["from_account"] = df["from_account"].astype(str).str.strip().str.upper()
    df["to_account"] = df["to_account"].astype(str).str.strip().str.upper()
    df["from_bank"] = df["from_bank"].astype(str).str.strip()
    df["to_bank"] = df["to_bank"].astype(str).str.strip()
    
    df["payment_currency"] = df["payment_currency"].astype(str).str.strip()
    df["receiving_currency"] = df["receiving_currency"].astype(str).str.strip()
    df["payment_format"] = df["payment_format"].astype(str).str.strip()

    df["amount_received"] = pd.to_numeric(df["amount_received"], errors="coerce").fillna(0.0)
    df["amount_paid"] = pd.to_numeric(df["amount_paid"], errors="coerce").fillna(0.0)
    
    if "is_laundering" in df.columns:
        df["is_laundering"] = pd.to_numeric(df["is_laundering"], errors="coerce").fillna(0).astype(int)
    else:
        df["is_laundering"] = 0

    # 3. Parse timestamp and sort
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values(by="timestamp").reset_index(drop=True)

    # 4. Join account metadata
    if os.path.exists(accounts_path):
        df_acc = load_accounts_metadata(accounts_path)
        
        # Source account metadata
        df = df.merge(
            df_acc[["Account Number", "Entity ID", "entity_type"]].rename(
                columns={
                    "Account Number": "from_account",
                    "Entity ID": "from_entity_id",
                    "entity_type": "from_entity_type",
                }
            ),
            on="from_account",
            how="left",
        )
        
        # Destination account metadata
        df = df.merge(
            df_acc[["Account Number", "Entity ID", "entity_type"]].rename(
                columns={
                    "Account Number": "to_account",
                    "Entity ID": "to_entity_id",
                    "entity_type": "to_entity_type",
                }
            ),
            on="to_account",
            how="left",
        )
    else:
        logger.warning(f"Accounts file {accounts_path} not found. Proceeding without entity metadata.")
        df["from_entity_id"] = "Unknown"
        df["from_entity_type"] = "Unknown"
        df["to_entity_id"] = "Unknown"
        df["to_entity_type"] = "Unknown"

    df["from_entity_type"] = df["from_entity_type"].fillna("Unknown")
    df["to_entity_type"] = df["to_entity_type"].fillna("Unknown")
    df["from_entity_id"] = df["from_entity_id"].fillna("Unknown")
    df["to_entity_id"] = df["to_entity_id"].fillna("Unknown")

    logger.info(f"Successfully cleaned and enriched {len(df):,} transactions.")
    
    if save_cache:
        try:
            df.to_parquet(cache_path, index=False)
            logger.info(f"Saved cleaned parquet cache to {cache_path}.")
        except Exception as e:
            logger.warning(f"Failed to save parquet cache: {e}")

    return df


def inspect_cleaned_data(df: pd.DataFrame) -> dict:
    """Generate statistical inspection summary."""
    inspection = {
        "total_transactions": len(df),
        "total_columns": len(df.columns),
        "missing_values": df.isnull().sum().to_dict(),
        "laundering_counts": df["is_laundering"].value_counts().to_dict() if "is_laundering" in df.columns else {},
        "unique_from_accounts": df["from_account"].nunique(),
        "unique_to_accounts": df["to_account"].nunique(),
        "unique_currencies": df["payment_currency"].nunique(),
        "payment_formats": df["payment_format"].value_counts().to_dict(),
        "time_span": {
            "start": str(df["timestamp"].min()),
            "end": str(df["timestamp"].max())
        }
    }
    return inspection


if __name__ == "__main__":
    df = clean_and_enrich_data()
    summary = inspect_cleaned_data(df)
    print("\n--- Cleaned Data Inspection Summary ---")
    for k, v in summary.items():
        print(f"{k}: {v}")
