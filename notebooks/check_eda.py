"""
Terminal equivalent of 01_sales_eda.ipynb — shows tables and saved figure paths.
Run from repo root: python notebooks/check_eda.py
"""
import sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.categorization import categorize
from src.cleaning import clean_transactions
from src.config import DATA_DIR, FIGURES_DIR
from src.data_loading import build_transactions, load_raw_tables
from src.eda import (
    pareto_analysis, plot_category_breakdown, plot_pareto,
    plot_top_products, revenue_by_category, top_products,
)
import matplotlib.pyplot as plt

print("Building pipeline...")
tables = load_raw_tables(DATA_DIR)
df = categorize(clean_transactions(build_transactions(tables)))
print(f"  {len(df):,} rows | {df['name'].nunique()} products | {df['name_category'].nunique()} categories\n")

# ── Top products by revenue ──────────────────────────────────────────────────
print("=== Top 20 Products by Revenue ===")
tbl = top_products(df, by="revenue", n=20)
print(f"  {'Rank':<5} {'Name':<45} {'Units':>8}  {'Revenue (DKK)':>14}  {'% Rev':>7}")
print("  " + "-" * 82)
for rank, row in tbl.iterrows():
    print(f"  {rank:<5} {row['name']:<45} {row['units']:>8,.0f}  {row['revenue']:>14,.0f}  {row['pct_revenue']:>6.1f}%")

# ── Pareto ────────────────────────────────────────────────────────────────────
print()
prod, n80 = pareto_analysis(df)
total_rev = df["revenue"].sum()
top20_share = prod.head(20)["revenue"].sum() / total_rev * 100
print(f"=== Pareto ===")
print(f"  Products to reach 80% of revenue: {n80}")
print(f"  Top-20 products share:            {top20_share:.1f}%  (anchor: ~80%)")

# ── Category breakdown ────────────────────────────────────────────────────────
print()
print("=== Revenue by Category ===")
cat = revenue_by_category(df)
print(f"  {'#':<4} {'Category':<22} {'Units':>8}  {'Revenue (DKK)':>14}  {'% Rev':>7}")
print("  " + "-" * 60)
for rank, row in cat.iterrows():
    print(f"  {rank:<4} {row['name_category']:<22} {row['units']:>8,.0f}  {row['revenue']:>14,.0f}  {row['pct_revenue']:>6.1f}%")

# ── Validation anchors ────────────────────────────────────────────────────────
print()
top1 = tbl.iloc[0]
print("=== Validation Anchors ===")
print(f"  Top product:  {top1['name']}  —  {top1['revenue']:,.0f} DKK  ({top1['pct_revenue']:.1f}%)  [anchor: ~5.3M, ~20%]")
print(f"  n80:          {n80}  [anchor: ~20]")
print(f"  #1 category:  {cat.iloc[0]['name_category']}  [anchor: Ice Cream]")
print(f"  #2 category:  {cat.iloc[1]['name_category']}  [anchor: Chocolate]")
print(f"  #3 category:  {cat.iloc[2]['name_category']}  [anchor: Buns & Bakery]")

# ── Save figures ──────────────────────────────────────────────────────────────
print()
print("Saving figures...")
fig = plot_top_products(df, by="revenue", n=15); plt.close(fig)
fig = plot_top_products(df, by="units",   n=15); plt.close(fig)
fig, _ = plot_pareto(df);                        plt.close(fig)
fig = plot_category_breakdown(df);               plt.close(fig)

for p in sorted(FIGURES_DIR.glob("*.png")):
    size_kb = p.stat().st_size // 1024
    print(f"  figures/{p.name}  ({size_kb} KB)")
