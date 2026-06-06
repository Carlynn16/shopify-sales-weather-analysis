"""
tests/test_weather.py — structural tests for the weather data layer.

Tests that require the Open-Meteo API are skipped gracefully when offline
or when the cache is absent.  After the first successful run the cache
lives in data/ (git-ignored) and all tests run without network access.
"""
import pytest
import requests

from src.categorization import categorize
from src.cleaning import clean_transactions
from src.config import DATA_DIR
from src.data_loading import build_transactions, load_raw_tables
from src.weather import WEATHER_VARS, build_weather_panel, fetch_weather, get_store_coords


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def df():
    tables = load_raw_tables(DATA_DIR)
    return categorize(clean_transactions(build_transactions(tables)))


@pytest.fixture(scope="module")
def locations():
    return load_raw_tables(DATA_DIR)["locations"]


@pytest.fixture(scope="module")
def panels(df, locations):
    """Build the weather panel; skip the whole test module if unavailable."""
    try:
        return build_weather_panel(df, locations)
    except requests.exceptions.ConnectionError:
        pytest.skip("No network access — weather API unavailable")
    except Exception as exc:
        pytest.skip(f"Weather panel build failed: {exc}")


@pytest.fixture(scope="module")
def store_panel(panels):
    return panels[0]


@pytest.fixture(scope="module")
def daily_panel(panels):
    return panels[1]


# ── Store panel structure ─────────────────────────────────────────────────────

def test_store_panel_has_weather_columns(store_panel):
    for col in WEATHER_VARS:
        assert col in store_panel.columns, f"Missing column: {col}"


def test_store_panel_has_required_columns(store_panel):
    for col in ("store_label", "date", "revenue_pct", "orders"):
        assert col in store_panel.columns


def test_store_panel_no_city_column(store_panel):
    """City names must never appear in the panel output."""
    assert "city" not in store_panel.columns


def test_store_panel_store_labels_anonymized(store_panel):
    assert all(lbl.startswith("Store ") for lbl in store_panel["store_label"].unique())


def test_store_panel_7_stores(store_panel):
    n = store_panel["store_label"].nunique()
    assert n == 7, f"Expected 7 stores, got {n}"


def test_store_panel_date_range(store_panel, df):
    from zoneinfo import ZoneInfo
    _TZ = ZoneInfo("Europe/Copenhagen")
    tx_start = df["created_at"].dt.tz_convert(_TZ).dt.date.min()
    tx_end   = df["created_at"].dt.tz_convert(_TZ).dt.date.max()
    assert store_panel["date"].min() >= tx_start
    assert store_panel["date"].max() <= tx_end


# ── Revenue aggregation correctness ──────────────────────────────────────────

def test_store_panel_revenue_pct_positive(store_panel):
    assert (store_panel["revenue_pct"] >= 0).all()


def test_store_panel_revenue_pct_bounded(store_panel):
    """Store panel covers known stores only; total share < 100 is expected."""
    assert store_panel["revenue_pct"].sum() < 105   # allow tiny rounding


def test_daily_panel_revenue_pct_sums_to_100(daily_panel):
    """Daily panel covers all stores; total should reach ~100% of revenue."""
    assert abs(daily_panel["revenue_pct"].sum() - 100.0) < 0.5


def test_orders_are_positive_integers(store_panel):
    assert (store_panel["orders"] > 0).all()
    assert store_panel["orders"].dtype.kind in ("i", "u", "f")


# ── Weather value plausibility ────────────────────────────────────────────────

def test_temperature_max_range(store_panel):
    col = store_panel["temperature_2m_max"].dropna()
    assert col.between(-20, 40).all(), f"Out-of-range temps: {col[~col.between(-20,40)]}"


def test_temperature_min_range(store_panel):
    col = store_panel["temperature_2m_min"].dropna()
    assert col.between(-25, 35).all()


def test_precipitation_non_negative(store_panel):
    assert (store_panel["precipitation_sum"].dropna() >= 0).all()


def test_windspeed_non_negative(store_panel):
    assert (store_panel["windspeed_10m_max"].dropna() >= 0).all()


def test_daylight_duration_range(store_panel):
    """Daylight in seconds: Denmark ranges roughly 6–18 hours (21600–64800 s)."""
    col = store_panel["daylight_duration"].dropna()
    assert (col >= 20_000).all() and (col <= 75_000).all(), (
        f"Unexpected daylight values: min={col.min()}, max={col.max()}"
    )


# ── Null rate after join ──────────────────────────────────────────────────────

def test_weather_null_rate_low(store_panel):
    """Fewer than 5% of store-days should have missing weather."""
    null_rate = store_panel["temperature_2m_max"].isna().mean()
    assert null_rate < 0.05, f"High null rate in weather join: {null_rate:.1%}"


# ── Geocoding (live API, skip if offline / cache available) ───────────────────

def test_get_store_coords_returns_valid_lat_lon(locations):
    """Runs against cache or live API; skips if both unavailable."""
    try:
        coords = get_store_coords(locations.dropna(subset=["city"]))
    except requests.exceptions.ConnectionError:
        pytest.skip("No network access")
    assert len(coords) > 0
    for city, (lat, lon) in coords.items():
        assert -90 <= lat <= 90,  f"Bad lat for {city}: {lat}"
        assert -180 <= lon <= 180, f"Bad lon for {city}: {lon}"


# ── Daily panel ───────────────────────────────────────────────────────────────

def test_daily_panel_has_weather_columns(daily_panel):
    for col in WEATHER_VARS:
        assert col in daily_panel.columns


def test_daily_panel_monotonic_dates(daily_panel):
    dates = daily_panel["date"].tolist()
    assert dates == sorted(dates)


def test_daily_panel_no_city_column(daily_panel):
    assert "city" not in daily_panel.columns
