"""
Terminal check for the data quality analysis.
Run from repo root: python notebooks/check_data_quality.py
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
from src.data_quality import (
    plot_unknown_by_store, plot_unknown_over_time, plot_unknown_store_over_time,
    unknown_by_store, unknown_over_time, unknown_product_summary, unknown_store_over_time,
)

print("Building pipeline...")
tables = load_raw_tables(DATA_DIR)
df = categorize(clean_transactions(build_transactions(tables)))

# ── Overall ───────────────────────────────────────────────────────────────────
print("\n=== Overall unknown-product rate ===")
summary = unknown_product_summary(df)
for _, r in summary.iterrows():
    print(f"  {r['metric']:<30}  {int(r['unknown_count']):>7,} / {int(r['total']):>7,}  =  {r['rate_pct']:>6.2f}%")

print("\n  >> Row rate vs unit rate diverge because unknown lines carry bulk quantities.")
print(f"  >> Avg units/line for unknown rows: "
      f"{int(summary.loc[1,'unknown_count']) / int(summary.loc[0,'unknown_count']):.0f}"
      f"  vs ~{int(summary.loc[1,'total'] - summary.loc[1,'unknown_count']) / (int(summary.loc[0,'total'] - summary.loc[0,'unknown_count'])):.0f} for known rows")

# ── By store ──────────────────────────────────────────────────────────────────
print("\n=== Unknown-product rate by store ===")
by_store = unknown_by_store(df)
print(f"  {'Store':<10} {'Unk rows':>9} {'Tot rows':>9} {'Row%':>7}  {'Unk units':>10} {'Tot units':>10} {'Unit%':>7}")
print("  " + "-" * 68)
for _, r in by_store.iterrows():
    print(f"  {r['store_label']:<10} {int(r['unknown_rows']):>9,} {int(r['total_rows']):>9,} {r['row_rate']*100:>6.1f}%"
          f"  {int(r['unknown_units']):>10,} {int(r['total_units']):>10,} {r['unit_rate']*100:>6.1f}%")

worst = by_store.iloc[0]
print(f"\n  Most affected: {worst['store_label']}  (unit rate {worst['unit_rate']*100:.1f}%)")
print(f"  Next highest:  {by_store.iloc[1]['store_label']}  (unit rate {by_store.iloc[1]['unit_rate']*100:.1f}%)")

# ── Over time ─────────────────────────────────────────────────────────────────
print("\n=== Monthly unknown rate (all stores) ===")
monthly = unknown_over_time(df)
print(f"  {'Month':<10}  {'Row%':>6}  {'Unit%':>6}")
print("  " + "-" * 28)
for _, r in monthly.iterrows():
    print(f"  {r['month_label']:<10}  {r['row_rate']*100:>5.1f}%  {r['unit_rate']*100:>5.1f}%")

# ── Worst store over time ─────────────────────────────────────────────────────
store_monthly, worst_label = unknown_store_over_time(df)
print(f"\n=== Monthly unknown rate — {worst_label} ===")
print(f"  {'Month':<10}  {'Row%':>6}  {'Unit%':>6}")
print("  " + "-" * 28)
for _, r in store_monthly.iterrows():
    print(f"  {r['month_label']:<10}  {r['row_rate']*100:>5.1f}%  {r['unit_rate']*100:>5.1f}%")

drop_month = store_monthly.loc[store_monthly["unit_rate"] < 0.01, "month_label"]
drop_from  = drop_month.iloc[0] if len(drop_month) else "not yet"
print(f"\n  Unit rate drops below 1% from: {drop_from}")

# ── Save figures ──────────────────────────────────────────────────────────────
print("\nSaving figures...")
for fn in (plot_unknown_by_store, plot_unknown_over_time, plot_unknown_store_over_time):
    result = fn(df)
    fig = result[0] if isinstance(result, tuple) else result
    plt.close(fig)

for p in sorted(FIGURES_DIR.glob("dq_*.png")):
    print(f"  figures/{p.name}  ({p.stat().st_size // 1024} KB)")
