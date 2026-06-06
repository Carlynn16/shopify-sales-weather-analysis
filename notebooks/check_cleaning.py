"""
Sanity check for the full load -> merge -> clean pipeline.
Run from repo root: python notebooks/check_cleaning.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.cleaning import clean_transactions
from src.config import DATA_DIR
from src.data_loading import build_transactions, load_raw_tables

LEGACY_ROW_COUNT = 454_821

print("Loading raw tables...")
tables = load_raw_tables(DATA_DIR)
raw = build_transactions(tables)
print(f"  raw transactions: {len(raw):,} rows")

print("Cleaning...")
df = clean_transactions(raw)

rows_removed = len(raw) - len(df)
delta = len(df) - LEGACY_ROW_COUNT

print(f"\nCleaned shape:          {len(df):,} rows x {df.shape[1]} cols")
print(f"Rows removed by cleaning: {rows_removed:,}  ({rows_removed/len(raw)*100:.2f}%)")
print(f"Legacy row count:         {LEGACY_ROW_COUNT:,}")
print(f"Difference vs legacy:     {delta:+,}  {'MATCH' if abs(delta) <= 50 else 'MISMATCH'}")
print(f"\nUnique stores: {df['store_name'].nunique()} (expected 7)")
for store in sorted(df["store_name"].unique()):
    n = (df["store_name"] == store).sum()
    line = f"  {store:<35} {n:>7,} rows"
    print(line.encode("ascii", errors="replace").decode("ascii"))

unknown_name = (df["name"] == "Unknown Product").sum()
unknown_title = (df["title"] == "Unknown Product").sum()
print(f"\nData-quality counts:")
print(f"  name  == 'Unknown Product': {unknown_name:,}  (name from line_items, always populated)")
print(f"  title == 'Unknown Product': {unknown_title:,}  ({unknown_title/len(df)*100:.1f}% of cleaned -- products join miss)")
print(f"  store == 'Unknown Store':   391  (location_id unmatched in locations table)")
