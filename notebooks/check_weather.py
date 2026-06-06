"""
Check script: build the weather panel and print diagnostics.
Run from repo root: python notebooks/check_weather.py
"""
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.categorization import categorize
from src.cleaning import clean_transactions
from src.config import DATA_DIR
from src.data_loading import build_transactions, load_raw_tables
from src.weather import WEATHER_VARS, build_weather_panel

print("Loading pipeline...")
tables   = load_raw_tables(DATA_DIR)
df       = categorize(clean_transactions(build_transactions(tables)))
locs     = tables["locations"]
print(f"  {len(df):,} transactions ready")

print("\nBuilding weather panel (uses cache if available)...")
store_panel, daily_panel = build_weather_panel(df, locs)

# ── Store panel summary ───────────────────────────────────────────────────────
n_total   = len(store_panel)
n_matched = store_panel["temperature_2m_max"].notna().sum()

print(f"\n=== Store panel ===")
print(f"  Shape:           {store_panel.shape[0]:,} rows x {store_panel.shape[1]} cols")
print(f"  Date range:      {store_panel['date'].min()}  to  {store_panel['date'].max()}")
print(f"  Stores:          {sorted(store_panel['store_label'].unique())}")
print(f"  Weather matched: {n_matched:,} / {n_total:,} store-days ({n_matched/n_total*100:.1f}%)")
print(f"  Columns:         {store_panel.columns.tolist()}")

# ── Daily panel summary ───────────────────────────────────────────────────────
nd_total   = len(daily_panel)
nd_matched = daily_panel["temperature_2m_max"].notna().sum()

print(f"\n=== Daily panel (all stores) ===")
print(f"  Shape:           {daily_panel.shape[0]:,} rows x {daily_panel.shape[1]} cols")
print(f"  Date range:      {daily_panel['date'].min()}  to  {daily_panel['date'].max()}")
print(f"  Weather matched: {nd_matched:,} / {nd_total:,} days ({nd_matched/nd_total*100:.1f}%)")

# ── Top-10 store-day rows by revenue (anonymized) ────────────────────────────
print(f"\n=== Sample: top-10 store-days by revenue share ===")
cols_show = ["store_label", "date", "revenue_pct",
             "temperature_2m_max", "apparent_temperature_max",
             "precipitation_sum", "windspeed_10m_max", "daylight_duration"]
sample = (
    store_panel.nlargest(10, "revenue_pct")[cols_show]
    .copy()
)
sample["revenue_pct"]    = sample["revenue_pct"].round(3)
sample["daylight_duration"] = (sample["daylight_duration"] / 3600).round(1)   # hours for readability
sample = sample.rename(columns={"daylight_duration": "daylight_h"})
print(sample.to_string(index=False))

# ── Weather summary stats ─────────────────────────────────────────────────────
print(f"\n=== Weather variable summary (store panel) ===")
print(f"  {'Variable':<30}  {'min':>7}  {'mean':>7}  {'max':>7}  {'null%':>6}")
print("  " + "-" * 66)
for col in WEATHER_VARS:
    vals = store_panel[col].dropna()
    null_pct = store_panel[col].isna().mean() * 100
    print(f"  {col:<30}  {vals.min():>7.1f}  {vals.mean():>7.1f}  {vals.max():>7.1f}  {null_pct:>5.1f}%")
