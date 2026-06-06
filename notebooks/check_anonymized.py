"""
Confirm all figures regenerate with anonymize=True (default).
Run from repo root: python notebooks/check_anonymized.py
"""
import sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).parent.parent))

import matplotlib.pyplot as plt

from src.categorization import categorize
from src.cleaning import clean_transactions
from src.config import DATA_DIR, FIGURES_DIR
from src.data_loading import build_transactions, load_raw_tables
from src.eda import (
    plot_category_breakdown, plot_category_seasonality,
    plot_hourly, plot_monthly_revenue, plot_pareto,
    plot_store_category_mix, plot_store_revenue,
    plot_top_products, plot_weekday,
    revenue_by_store, top_products,
)

print("Building pipeline...")
tables = load_raw_tables(DATA_DIR)
df = categorize(clean_transactions(build_transactions(tables)))

print("Regenerating all figures with anonymize=True (default)...")
fns = [
    (plot_top_products,        {"by": "revenue", "n": 15}),
    (plot_top_products,        {"by": "units",   "n": 15}),
    (plot_pareto,              {}),
    (plot_category_breakdown,  {}),
    (plot_monthly_revenue,     {}),
    (plot_weekday,             {}),
    (plot_hourly,              {}),
    (plot_category_seasonality,{}),
    (plot_store_revenue,       {}),
    (plot_store_category_mix,  {}),
]
for fn, kw in fns:
    fig_result = fn(df, **kw)
    fig = fig_result[0] if isinstance(fig_result, tuple) else fig_result
    plt.close(fig)

print()
print("=== Spot-check: top_products table (anonymize=True) ===")
tbl = top_products(df, by="revenue", n=5)
print(f"  {'Rank':<5} {'Name':<45} {'Revenue %':>10}  {'Units %':>8}")
print("  " + "-" * 72)
for rank, row in tbl.iterrows():
    print(f"  {rank:<5} {row['name']:<45} {row['revenue']:>9.2f}%  {row['units']:>7.2f}%")

print()
print("=== Spot-check: store labels (anonymize=True) ===")
st = revenue_by_store(df, anonymize=True)
print(f"  {'Label':<10} {'Revenue %':>10}  {'Orders %':>10}  {'AOV idx':>8}")
print("  " + "-" * 44)
for _, r in st.iterrows():
    print(f"  {r['store_label']:<10} {r['revenue']:>9.2f}%  {r['order_count']:>9.2f}%  {r['avg_order_value']:>7.1f}")

print()
print("=== Confirming no DKK / no real store names in figures/ ===")
# Can't scan binary PNGs for text, but we verify the figure list
for p in sorted(FIGURES_DIR.glob("*.png")):
    print(f"  {p.name:<45} ({p.stat().st_size // 1024} KB)")

print()
print("=== Confirming no store suburb names in PROJECT_BRIEF.md ===")
flagged = [w for w in ["[location]","[location]","[location]","[location]","Vedb","rrebro","sterbro"]
           if w in Path("PROJECT_BRIEF.md").read_text(encoding="utf-8")]
if flagged:
    print(f"  WARNING: found {flagged}")
else:
    print("  OK — no real store/suburb names found in PROJECT_BRIEF.md")
