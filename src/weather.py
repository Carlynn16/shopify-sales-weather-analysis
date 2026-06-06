"""
src/weather.py — weather data acquisition and panel construction for Block B.

Store cities are read at runtime from the (git-ignored) locations table.
No real city names are hardcoded in this file.
All fetched data is cached under data/ (git-ignored).
"""
import json
import time
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import requests

from src.anonymize import store_labels
from src.config import DATA_DIR

_TZ = ZoneInfo("Europe/Copenhagen")

_GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"
_ARCHIVE_URL   = "https://archive-api.open-meteo.com/v1/archive"
_COORDS_CACHE  = DATA_DIR / "store_coords.cache.json"

WEATHER_VARS: list[str] = [
    "temperature_2m_max",
    "temperature_2m_min",
    "apparent_temperature_max",
    "precipitation_sum",
    "rain_sum",
    "snowfall_sum",
    "windspeed_10m_max",
    "daylight_duration",
    "sunshine_duration",
]


def _weather_cache_path(lat: float, lon: float) -> Path:
    return DATA_DIR / f"weather_{lat:.4f}_{lon:.4f}.json"


# ── Geocoding ─────────────────────────────────────────────────────────────────

def get_store_coords(locations: pd.DataFrame) -> dict[str, tuple[float, float]]:
    """
    Geocode each unique non-null city in *locations* to (lat, lon).

    Caches to data/store_coords.cache.json (git-ignored) so the API is
    only called once per city.  *locations* must have a 'city' column.

    Returns {city: (lat, lon)}.
    """
    # Load existing cache
    cache: dict[str, list[float]] = {}
    if _COORDS_CACHE.exists():
        cache = json.loads(_COORDS_CACHE.read_text(encoding="utf-8"))

    cities  = locations["city"].dropna().unique().tolist()
    updated = False

    for city in cities:
        if city in cache:
            continue

        # Try progressively shorter name variants if the full name fails
        # (e.g. "Frederiksberg C" → "Frederiksberg"). Fully generic — no
        # real names hardcoded here.
        parts     = city.split()
        names_to_try = [" ".join(parts[:k]) for k in range(len(parts), 0, -1)]
        matched   = None

        for candidate in names_to_try:
            resp = requests.get(
                _GEOCODING_URL,
                params={"name": candidate, "count": 1, "language": "en", "format": "json"},
                timeout=10,
            )
            resp.raise_for_status()
            results = resp.json().get("results", [])
            time.sleep(0.15)
            if results:
                matched = results[0]
                break

        if matched is None:
            raise ValueError(f"Geocoding failed for city: {city!r}")

        cache[city] = [matched["latitude"], matched["longitude"]]
        updated = True

    if updated:
        _COORDS_CACHE.parent.mkdir(parents=True, exist_ok=True)
        _COORDS_CACHE.write_text(json.dumps(cache, indent=2), encoding="utf-8")

    return {city: (float(ll[0]), float(ll[1])) for city, ll in cache.items()
            if city in cities}


# ── Weather fetching ──────────────────────────────────────────────────────────

def fetch_weather(lat: float, lon: float, start: str, end: str) -> pd.DataFrame:
    """
    Fetch daily weather from the Open-Meteo archive API.

    start / end: 'YYYY-MM-DD'.
    Caches per (lat, lon) to data/weather_{lat}_{lon}.json (git-ignored).
    If the cache already covers the requested range, no API call is made.

    Returns a DataFrame with columns: date (datetime.date) + WEATHER_VARS.
    """
    cache_path = _weather_cache_path(lat, lon)
    req_start  = pd.to_datetime(start).date()
    req_end    = pd.to_datetime(end).date()

    # Try serving from cache
    if cache_path.exists():
        raw       = json.loads(cache_path.read_text(encoding="utf-8"))
        cached_df = pd.DataFrame(raw)
        cached_df["date"] = pd.to_datetime(cached_df["date"]).dt.date
        if cached_df["date"].min() <= req_start and cached_df["date"].max() >= req_end:
            mask = (cached_df["date"] >= req_start) & (cached_df["date"] <= req_end)
            return cached_df[mask].reset_index(drop=True)
        existing = cached_df
    else:
        existing = None

    # Fetch from API
    resp = requests.get(
        _ARCHIVE_URL,
        params={
            "latitude":   lat,
            "longitude":  lon,
            "start_date": start,
            "end_date":   end,
            "daily":      ",".join(WEATHER_VARS),
            "timezone":   "Europe/Copenhagen",
        },
        timeout=30,
    )
    resp.raise_for_status()
    daily = resp.json()["daily"]

    fresh = pd.DataFrame({"date": pd.to_datetime(daily["time"]).date})
    for var in WEATHER_VARS:
        fresh[var] = daily.get(var)

    # Merge with existing cache, deduplicate, and save
    if existing is not None:
        combined = (
            pd.concat([existing, fresh])
            .drop_duplicates("date")
            .sort_values("date")
            .reset_index(drop=True)
        )
    else:
        combined = fresh.sort_values("date").reset_index(drop=True)

    saveable = combined.copy()
    saveable["date"] = saveable["date"].astype(str)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(
        json.dumps(saveable.to_dict(orient="list"), indent=2), encoding="utf-8"
    )

    mask = (combined["date"] >= req_start) & (combined["date"] <= req_end)
    return combined[mask].reset_index(drop=True)


# ── Panel builder ─────────────────────────────────────────────────────────────

def build_weather_panel(
    transactions: pd.DataFrame,
    locations: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Join daily aggregated store revenue with daily weather data.

    Parameters
    ----------
    transactions : cleaned transaction-level DataFrame (from the full pipeline)
    locations    : raw locations table (columns include 'id', 'city')

    Returns
    -------
    store_panel : pd.DataFrame
        One row per (store_label × local date).
        Columns: store_label, date, revenue_pct, orders, <WEATHER_VARS>.
        Store labels are anonymized (Store A–G); no city names in output.

    daily_panel : pd.DataFrame
        One row per local date (all stores combined).
        Columns: date, revenue_pct, orders, <WEATHER_VARS>.
        Weather is the area-average of all store-city values for that date.
    """
    # ── 1. Build location lookup for geocoding: location_id → city ──────────
    locs = (
        locations
        .rename(columns={"id": "location_id"})
        [["location_id", "city"]]
        .dropna(subset=["city"])
    )

    # ── 2. Add local date to transactions ─────────────────────────────────────
    # city and store_name are already present from build_transactions()
    tx = transactions.copy()
    tx["date"] = tx["created_at"].dt.tz_convert(_TZ).dt.date

    # Ensure city column exists (join if it wasn't in the pipeline output)
    if "city" not in tx.columns:
        tx = tx.merge(locs, on="location_id", how="left")

    # ── 3. Geocode only cities that appear in the transactions ────────────────
    active_cities = tx["city"].dropna().unique()
    locs_active   = locs[locs["city"].isin(active_cities)].drop_duplicates("city")
    coords        = get_store_coords(locs_active)      # {city: (lat, lon)}

    # ── 4. Determine date range and fetch weather per city ────────────────────
    start_str = str(tx["date"].min())
    end_str   = str(tx["date"].max())

    city_wx: dict[str, pd.DataFrame] = {}
    for city in active_cities:
        if city not in coords:
            continue
        lat, lon = coords[city]
        wx = fetch_weather(lat, lon, start_str, end_str)
        wx["city"] = city
        city_wx[city] = wx

    wx_all = pd.concat(city_wx.values(), ignore_index=True) if city_wx else pd.DataFrame()

    # ── 5. Aggregate transactions to store × date (known stores only) ─────────
    labels   = store_labels(transactions)
    tx["store_label"] = tx["store_name"].map(labels)
    tx_known = tx[tx["city"].notna() & tx["store_label"].notna()]

    store_agg = (
        tx_known
        .groupby(["store_label", "date", "city"], as_index=False)
        .agg(revenue=("revenue", "sum"), orders=("order_id", "nunique"))
    )

    # ── 6. Join weather to store_agg ──────────────────────────────────────────
    if not wx_all.empty:
        store_panel = store_agg.merge(wx_all, on=["city", "date"], how="left")
    else:
        store_panel = store_agg.copy()
        for var in WEATHER_VARS:
            store_panel[var] = float("nan")

    # Drop city — it is a real location name, must not appear in committed output
    store_panel = store_panel.drop(columns=["city"])

    total_rev = transactions["revenue"].sum()
    store_panel["revenue_pct"] = store_panel["revenue"] / total_rev * 100
    store_panel = (
        store_panel
        .drop(columns=["revenue"])
        [["store_label", "date", "revenue_pct", "orders"] + WEATHER_VARS]
        .sort_values(["store_label", "date"])
        .reset_index(drop=True)
    )

    # ── 7. All-stores daily panel with area-averaged weather ──────────────────
    daily_rev = (
        tx
        .groupby("date", as_index=False)
        .agg(revenue=("revenue", "sum"), orders=("order_id", "nunique"))
    )
    daily_rev["revenue_pct"] = daily_rev["revenue"] / total_rev * 100
    daily_rev = daily_rev.drop(columns=["revenue"])

    if not wx_all.empty:
        wx_avg = wx_all.groupby("date", as_index=False)[WEATHER_VARS].mean()
        daily_panel = daily_rev.merge(wx_avg, on="date", how="left")
    else:
        daily_panel = daily_rev.copy()
        for var in WEATHER_VARS:
            daily_panel[var] = float("nan")

    daily_panel = (
        daily_panel
        [["date", "revenue_pct", "orders"] + WEATHER_VARS]
        .sort_values("date")
        .reset_index(drop=True)
    )

    return store_panel, daily_panel
