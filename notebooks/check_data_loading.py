"""
Quick sanity check: load raw tables, build transactions, print diagnostics.
Run from repo root: python notebooks/check_data_loading.py
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import DATA_DIR
from src.data_loading import build_transactions, load_raw_tables

print("Loading raw tables...")
tables = load_raw_tables(DATA_DIR)
for name, df in tables.items():
    print(f"  {name:12s}  {df.shape[0]:>8,} rows  {df.shape[1]:>3} cols")

print("\nBuilding transactions...")
tx = build_transactions(tables)

print(f"\n{'Shape:':<22} {tx.shape[0]:,} rows x {tx.shape[1]} cols")
print(f"{'Columns:':<22} {tx.columns.tolist()}")
print(f"{'created_at range:':<22} {tx['created_at'].min()}  to  {tx['created_at'].max()}")
print(f"{'Unique orders:':<22} {tx['order_id'].nunique():,}")
print(f"{'Unique product_ids:':<22} {tx['product_id'].nunique():,}")
print(f"{'Unique product names:':<22} {tx['name'].nunique():,}")

print(f"\n--- Row count comparison ---")
li_rows = len(tables["line_items"])
print(f"  line_items (pre-merge):           {li_rows:>9,}")
print(f"  transactions (post-merge):        {len(tx):>9,}  {'(== line_items row count, correct for left join)' if len(tx) == li_rows else '(!)'}")

legacy_path = Path(__file__).parent.parent / "_legacy" / "shopify_data.csv"
if legacy_path.exists():
    legacy_rows = sum(1 for _ in open(legacy_path, encoding="utf-8")) - 1
    print(f"  legacy shopify_data.csv (cleaned): {legacy_rows:>8,}  (~454k expected)")
    print(f"  difference (unfiltered - cleaned): {len(tx) - legacy_rows:>8,}")
else:
    print("  (legacy shopify_data.csv not found — skipping comparison)")
