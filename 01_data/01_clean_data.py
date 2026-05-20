import pandas as pd
import numpy as np
import os

os.makedirs('data', exist_ok=True)

print("="*55)
print("  UPI PULSE 2024 — DATA CLEANING")
print("="*55)

# ── 1. Load raw data ──────────────────────────────────────────
print("\nLoading raw data...")
df = pd.read_csv("upi_transactions_2024.csv")  
print(f"Original shape: {df.shape}")

print("\nOriginal columns:")
print(df.columns.tolist())

# ── 2. Clean column names ─────────────────────────────────────
# regex=False is critical — ( and ) are special regex characters
# without it, pandas may misinterpret them
df.columns = (
    df.columns
    .str.strip()
    .str.lower()
    .str.replace(" ", "_",  regex=False)
    .str.replace("(", "",   regex=False)
    .str.replace(")", "",   regex=False)
)

print("\nCleaned columns:")
print(df.columns.tolist())

# ── 3. Data quality checks ────────────────────────────────────
print("\nNULL VALUES:")
print(df.isnull().sum())

print(f"\nDUPLICATE ROWS: {df.duplicated().sum()}")

print(f"\nFRAUD COUNT: {df['fraud_flag'].sum()}")
print(f"FRAUD RATE:  {round(df['fraud_flag'].mean() * 100, 3)} %")

print(f"\nTRANSACTION STATUS values: {df['transaction_status'].unique()}")

# ── 4. Fix timestamp ──────────────────────────────────────────
# errors='coerce' means: if a timestamp can't be parsed,
# put NaT (Not a Time) instead of crashing
df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

# ── 5. Extract time features from timestamp ───────────────────
df["month"]     = df["timestamp"].dt.month_name()
# month_name() gives: January, February, etc.
# We also need month as a number for sorting in SQL and ML
df["month_num"] = df["timestamp"].dt.month
# month gives: 1, 2, 3 ... 12

df["hour"]      = df["timestamp"].dt.hour
# Same as hour_of_day — keeping for compatibility

# ── 6. Feature engineering (new derived columns) ──────────────

# Amount buckets — group amounts into ranges
df["amount_bucket"] = pd.cut(
    df["amount_inr"],
    bins=[0, 100, 500, 1000, 5000, 100000],
    labels=["micro", "small", "medium", "large", "very_large"]
)

# Success flag — 1 if success, 0 if failed
df["is_success"] = (df["transaction_status"] == "SUCCESS").astype(int)

# Same bank flag — 1 if sender and receiver use same bank
# Useful for ML model — inter-bank vs intra-bank transfers
df["is_same_bank"] = (df["sender_bank"] == df["receiver_bank"]).astype(int)

# Age group as ordered number — ML needs numbers not text
# 18-25=1, 26-35=2, 36-45=3, 46-55=4, 56+=5
age_map = {"18-25": 1, "26-35": 2, "36-45": 3, "46-55": 4, "56+": 5}
df["sender_age_order"] = df["sender_age_group"].map(age_map)

# ── 7. Final checks ───────────────────────────────────────────
print(f"\nFINAL SHAPE: {df.shape}")
print(f"Columns added: month, month_num, hour, amount_bucket,")
print(f"               is_success, is_same_bank, sender_age_order")

print("\nSample of new columns:")
print(df[["amount_inr","amount_bucket","is_success",
          "is_same_bank","sender_age_order","month","month_num"]].head(3))

# ── 8. Save ───────────────────────────────────────────────────
df.to_csv("data/upi_clean.csv", index=False)
print("\n✅ DONE — Saved: data/upi_clean.csv")

import pandas as pd
from sqlalchemy import create_engine

print("Loading cleaned CSV...")

df = pd.read_csv("data/upi_clean.csv")

print("Rows:", len(df))
print("Columns:", df.columns)

# IMPORTANT: replace password
engine = create_engine(
    "postgresql+psycopg2://postgres:actualpassword@localhost:5432/upi_transaction_analysis"
)

print("Uploading to PostgreSQL...")

df.to_sql(
    "transactions",
    engine,
    if_exists="replace",
    index=False
)

print("DONE: Table 'transactions' created successfully")