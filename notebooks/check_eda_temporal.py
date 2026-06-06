"""
Terminal check for the temporal + store EDA functions.
Run from repo root: python notebooks/check_eda_temporal.py
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
    category_seasonality, plot_category_seasonality,
    plot_hourly, plot_monthly_revenue,
    plot_store_category_mix, plot_store_revenue,
    plot_weekday, revenue_by_hour,
    revenue_by_month, revenue_by_store,
    revenue_by_weekday, store_category_mix,
)

print("Building pipeline...")
tables = load_raw_tables(DATA_DIR)
df = categorize(clean_transactions(build_transactions(tables)))
print(f"  {len(df):,} rows ready\n")

# ── Monthly ───────────────────────────────────────────────────────────────────
print("=== Monthly Revenue ===")
m = revenue_by_month(df)
print(f"  {'Month':<10} {'Revenue (DKK)':>14}  {'Orders':>8}")
print("  " + "-" * 38)
for _, r in m.iterrows():
    peak = " <-- PEAK" if r["revenue"] == m["revenue"].max() else ""
    print(f"  {r['month_label']:<10} {r['revenue']:>14,.0f}  {r['order_count']:>8,}{peak}")

# ── Weekday ───────────────────────────────────────────────────────────────────
print()
print("=== Revenue by Day of Week (local time) ===")
wd = revenue_by_weekday(df)
print(f"  {'Day':<12} {'Revenue (DKK)':>14}  {'Orders':>7}  {'AOV (DKK)':>10}")
print("  " + "-" * 50)
for _, r in wd.iterrows():
    peak = " <--" if r["revenue"] == wd["revenue"].max() else ""
    print(f"  {r['weekday_name']:<12} {r['revenue']:>14,.0f}  {r['order_count']:>7,}  {r['avg_order_value']:>10,.0f}{peak}")

# ── Hourly ────────────────────────────────────────────────────────────────────
print()
print("=== Top 8 Revenue Hours (local time) ===")
hr = revenue_by_hour(df)
print(f"  {'Hour':<8} {'Revenue (DKK)':>14}  {'Orders':>7}")
print("  " + "-" * 35)
for _, r in hr.nlargest(8, "revenue").sort_values("hour").iterrows():
    print(f"  {int(r['hour']):02d}:00    {r['revenue']:>14,.0f}  {r['order_count']:>7,}")

# ── Store ─────────────────────────────────────────────────────────────────────
print()
print("=== Store Performance ===")
st = revenue_by_store(df)
print(f"  {'Store':<22} {'Revenue (DKK)':>14}  {'Orders':>7}  {'AOV (DKK)':>10}")
print("  " + "-" * 60)
for _, r in st.iterrows():
    print(f"  {r['store_short']:<22} {r['revenue']:>14,.0f}  {r['order_count']:>7,}  {r['avg_order_value']:>10,.0f}")

# ── Category seasonality summary ──────────────────────────────────────────────
print()
print("=== Category Seasonality (peak month per category) ===")
pivot = category_seasonality(df)
for cat in pivot.index:
    peak_col = pivot.loc[cat].idxmax()
    peak_val = pivot.loc[cat].max()
    print(f"  {cat:<22}  peak: {peak_col}  ({peak_val:>10,.0f} DKK)")

# ── Validation anchors ────────────────────────────────────────────────────────
print()
print("=== Validation Anchors ===")
peak_month = m.loc[m["revenue"].idxmax(), "month_label"]
top2_days  = wd.nlargest(2, "revenue")["weekday_name"].tolist()
top3_hours = hr.nlargest(3, "revenue")["hour"].tolist()
print(f"  Peak month:  {peak_month}  [anchor: May-Jun]")
print(f"  Top-2 days:  {top2_days}  [anchor: Fri/Sat]")
print(f"  Top-3 hours: {[f'{h}:00' for h in sorted(top3_hours)]}  [anchor: 12-15]")

# ── Save all figures ──────────────────────────────────────────────────────────
print()
print("Saving figures...")
for fn, kwargs in [
    (plot_monthly_revenue,      {}),
    (plot_weekday,              {}),
    (plot_hourly,               {}),
    (plot_category_seasonality, {}),
    (plot_store_revenue,        {}),
    (plot_store_category_mix,   {}),
]:
    fig = fn(df, **kwargs)
    plt.close(fig)

print()
for p in sorted(FIGURES_DIR.glob("*.png")):
    print(f"  figures/{p.name}  ({p.stat().st_size // 1024} KB)")
