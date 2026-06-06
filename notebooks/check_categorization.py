"""
Sanity check: full load -> merge -> clean -> categorize pipeline.
Run from repo root: python notebooks/check_categorization.py
"""
import sys
from pathlib import Path

import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.categorization import categorize
from src.cleaning import clean_transactions
from src.config import DATA_DIR
from src.data_loading import build_transactions, load_raw_tables

print("Running pipeline...")
tables = load_raw_tables(DATA_DIR)
df = categorize(clean_transactions(build_transactions(tables)))

print(f"\nFinal shape: {df.shape[0]:,} rows x {df.shape[1]} cols\n")

# Revenue breakdown by category
rev = (
    df.groupby("name_category")
    .agg(row_count=("revenue", "count"), total_revenue=("revenue", "sum"))
    .assign(pct_revenue=lambda x: x["total_revenue"] / x["total_revenue"].sum() * 100)
    .sort_values("total_revenue", ascending=False)
)

print(f"{'Category':<20} {'Rows':>8}  {'Revenue (DKK)':>16}  {'% Rev':>7}")
print("-" * 58)
for cat, row in rev.iterrows():
    print(f"{cat:<20} {row['row_count']:>8,}  {row['total_revenue']:>16,.0f}  {row['pct_revenue']:>6.1f}%")

unc_share = (df["name_category"] == "Uncategorized").mean()
print(f"\nUncategorized: {(df['name_category'] == 'Uncategorized').sum():,} rows  ({unc_share:.2%})")

print("\nExpected ranking: Ice Cream #1, Chocolate #2, Buns & Bakery #3")
top3 = rev.index[:3].tolist()
print(f"Actual top 3:    {top3}")
match = top3 == ["Ice Cream", "Chocolate", "Buns & Bakery"]
print(f"Ranking match:   {'YES' if match else 'NO - check categorization'}")
