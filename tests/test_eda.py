import matplotlib.pyplot as plt
import pytest

from src.anonymize import store_labels
from src.categorization import categorize
from src.cleaning import clean_transactions
from src.config import DATA_DIR, FIGURES_DIR
from src.data_loading import build_transactions, load_raw_tables
from src.eda import (
    category_seasonality,
    pareto_analysis,
    plot_category_breakdown,
    plot_category_seasonality,
    plot_hourly,
    plot_monthly_revenue,
    plot_pareto,
    plot_store_category_mix,
    plot_store_revenue,
    plot_top_products,
    plot_weekday,
    revenue_by_category,
    revenue_by_hour,
    revenue_by_month,
    revenue_by_store,
    revenue_by_weekday,
    store_category_mix,
    top_products,
)


@pytest.fixture(scope="module")
def df():
    tables = load_raw_tables(DATA_DIR)
    raw = build_transactions(tables)
    cleaned = clean_transactions(raw)
    return categorize(cleaned)


# ── Pareto ────────────────────────────────────────────────────────────────────

def test_pareto_cumulative_monotonic(df):
    prod, _ = pareto_analysis(df)
    shares = prod["cumulative_share"].values
    assert (shares[1:] >= shares[:-1]).all(), "cumulative_share must be non-decreasing"


def test_pareto_cumulative_reaches_100(df):
    prod, _ = pareto_analysis(df)
    assert abs(prod["cumulative_share"].iloc[-1] - 100.0) < 0.01


def test_pareto_n80_plausible(df):
    _, n80 = pareto_analysis(df)
    # Brief states ~20 products = 80%; allow ±10 tolerance
    assert 10 <= n80 <= 30, f"Expected n80 ~20, got {n80}"


# ── Top products ──────────────────────────────────────────────────────────────

def test_top_products_sorted_by_revenue(df):
    tbl = top_products(df, by="revenue", n=20)
    revs = tbl["revenue"].values
    assert (revs[:-1] >= revs[1:]).all(), "Revenue column must be descending"


def test_top_products_sorted_by_units(df):
    tbl = top_products(df, by="units", n=20)
    units = tbl["units"].values
    assert (units[:-1] >= units[1:]).all(), "Units column must be descending"


def test_top_products_revenue_matches_source(df):
    n = 50
    tbl = top_products(df, by="revenue", n=n, anonymize=False)
    expected = (
        df.groupby("name")["revenue"].sum()
        .sort_values(ascending=False)
        .head(n)
        .sum()
    )
    assert abs(tbl["revenue"].sum() - expected) < 1.0


def test_top_products_index_one_based(df):
    tbl = top_products(df, by="revenue", n=10)
    assert tbl.index[0] == 1 and tbl.index[-1] == 10


def test_top_products_invalid_by_raises(df):
    with pytest.raises(ValueError):
        top_products(df, by="price")


# ── Plot functions write files ────────────────────────────────────────────────

def test_plot_top_products_revenue_saves_file(df):
    fig = plot_top_products(df, by="revenue", n=15)
    plt.close(fig)
    assert (FIGURES_DIR / "top_15_products_by_revenue.png").exists()


def test_plot_top_products_units_saves_file(df):
    fig = plot_top_products(df, by="units", n=15)
    plt.close(fig)
    assert (FIGURES_DIR / "top_15_products_by_units.png").exists()


def test_plot_pareto_saves_file(df):
    fig, n80 = plot_pareto(df)
    plt.close(fig)
    assert (FIGURES_DIR / "pareto_revenue.png").exists()
    assert isinstance(n80, int) and n80 > 0


def test_plot_category_breakdown_saves_file(df):
    fig = plot_category_breakdown(df)
    plt.close(fig)
    assert (FIGURES_DIR / "category_revenue_breakdown.png").exists()


# ── Revenue by category ───────────────────────────────────────────────────────

def test_category_table_sorted_by_revenue(df):
    tbl = revenue_by_category(df)
    revs = tbl["revenue"].values
    assert (revs[:-1] >= revs[1:]).all()


def test_category_pct_sums_to_100(df):
    tbl = revenue_by_category(df)
    assert abs(tbl["pct_revenue"].sum() - 100.0) < 0.01


def test_ice_cream_top_category(df):
    tbl = revenue_by_category(df)
    assert tbl["name_category"].iloc[0] == "Ice Cream"


# ── Monthly revenue ───────────────────────────────────────────────────────────

def test_monthly_revenue_sums_to_total(df):
    tbl = revenue_by_month(df, anonymize=False)
    assert abs(tbl["revenue"].sum() - df["revenue"].sum()) < 1.0


def test_monthly_revenue_chronological(df):
    tbl = revenue_by_month(df)
    periods = tbl["month_period"].tolist()
    assert periods == sorted(periods)


def test_monthly_revenue_14_to_15_months(df):
    # Data runs Jan 2024 – Mar 2025: 15 calendar months (brief quotes ~14 as approximation)
    tbl = revenue_by_month(df)
    assert 14 <= len(tbl) <= 16, f"Expected 14-16 months, got {len(tbl)}"


def test_plot_monthly_revenue_saves_file(df):
    fig = plot_monthly_revenue(df)
    plt.close(fig)
    assert (FIGURES_DIR / "monthly_revenue.png").exists()


# ── Weekday revenue ───────────────────────────────────────────────────────────

def test_weekday_has_7_buckets(df):
    tbl = revenue_by_weekday(df)
    assert len(tbl) == 7


def test_weekday_sums_to_total(df):
    tbl = revenue_by_weekday(df, anonymize=False)
    assert abs(tbl["revenue"].sum() - df["revenue"].sum()) < 1.0


def test_weekday_ordered_mon_to_sun(df):
    tbl = revenue_by_weekday(df)
    assert tbl["weekday"].tolist() == list(range(7))


def test_plot_weekday_saves_file(df):
    fig = plot_weekday(df)
    plt.close(fig)
    assert (FIGURES_DIR / "weekday_revenue.png").exists()


# ── Hourly revenue ────────────────────────────────────────────────────────────

def test_hourly_sums_to_total(df):
    tbl = revenue_by_hour(df, anonymize=False)
    assert abs(tbl["revenue"].sum() - df["revenue"].sum()) < 1.0


def test_hourly_hours_in_range(df):
    tbl = revenue_by_hour(df)
    assert tbl["hour"].between(0, 23).all()


def test_plot_hourly_saves_file(df):
    fig = plot_hourly(df)
    plt.close(fig)
    assert (FIGURES_DIR / "hourly_revenue.png").exists()


# ── Category seasonality ──────────────────────────────────────────────────────

def test_category_seasonality_shape(df):
    pivot = category_seasonality(df)
    assert pivot.shape[0] == 8              # 8 categories
    assert 14 <= pivot.shape[1] <= 16       # 14-15 calendar months in this dataset


def test_category_seasonality_sums_to_total(df):
    pivot = category_seasonality(df, anonymize=False)
    # Uncategorized rows are intentionally excluded from the seasonality pivot
    known_rev = df[df["name_category"] != "Uncategorized"]["revenue"].sum()
    assert abs(pivot.values.sum() - known_rev) < 1.0


def test_plot_category_seasonality_saves_file(df):
    fig = plot_category_seasonality(df)
    plt.close(fig)
    assert (FIGURES_DIR / "category_seasonality.png").exists()


# ── Store revenue ─────────────────────────────────────────────────────────────

def test_store_revenue_7_stores(df):
    tbl = revenue_by_store(df)
    assert len(tbl) == 7


def test_store_revenue_sums_to_known_total(df):
    tbl = revenue_by_store(df, anonymize=False)
    known_total = df[df["store_name"] != "Unknown Store"]["revenue"].sum()
    assert abs(tbl["revenue"].sum() - known_total) < 1.0


def test_store_avg_order_value_positive(df):
    tbl = revenue_by_store(df)
    assert (tbl["avg_order_value"] > 0).all()


def test_plot_store_revenue_saves_file(df):
    fig = plot_store_revenue(df)
    plt.close(fig)
    assert (FIGURES_DIR / "store_revenue.png").exists()


# ── Store category mix ────────────────────────────────────────────────────────

def test_store_category_mix_rows_sum_to_100(df):
    pivot = store_category_mix(df)
    row_sums = pivot.sum(axis=1)
    assert (abs(row_sums - 100) < 0.01).all()


def test_store_category_mix_no_unknown_store(df):
    pivot = store_category_mix(df)
    assert "Unknown Store" not in pivot.index
    assert not any("Unknown Store" in s for s in pivot.index)


def test_plot_store_category_mix_saves_file(df):
    fig = plot_store_category_mix(df)
    plt.close(fig)
    assert (FIGURES_DIR / "store_category_mix.png").exists()


# ── Anonymization behaviour ───────────────────────────────────────────────────

def test_anonymized_top_products_revenue_is_pct(df):
    """With anonymize=True, revenue values are %-of-total (all ≤ 100, sum ≈ 100 for all products)."""
    tbl = top_products(df, by="revenue", n=320, anonymize=True)
    assert tbl["revenue"].max() <= 100.0
    assert abs(tbl["revenue"].sum() - 100.0) < 0.01


def test_anonymized_monthly_max_is_100(df):
    tbl = revenue_by_month(df, anonymize=True)
    assert abs(tbl["revenue"].max() - 100.0) < 0.01


def test_anonymized_weekday_max_is_100(df):
    tbl = revenue_by_weekday(df, anonymize=True)
    assert abs(tbl["revenue"].max() - 100.0) < 0.01


def test_anonymized_hourly_max_is_100(df):
    tbl = revenue_by_hour(df, anonymize=True)
    assert abs(tbl["revenue"].max() - 100.0) < 0.01


def test_anonymized_store_labels_are_a_to_g(df):
    tbl = revenue_by_store(df, anonymize=True)
    expected = {f"Store {chr(65 + i)}" for i in range(7)}
    assert set(tbl["store_label"]) == expected


def test_anonymized_store_no_real_names(df):
    # Derive forbidden terms at runtime from the git-ignored data — no brand string hardcoded
    real_names = {n for n in df["store_name"].dropna().unique() if n != "Unknown Store"}
    tbl = revenue_by_store(df, anonymize=True)
    for real in real_names:
        assert real not in tbl["store_label"].values, f"Real store name leaked: {real!r}"
    # anonymized output must not expose a store_name column
    assert "store_name" not in tbl.columns


def test_anonymized_category_mix_store_labels(df):
    real_names = {n for n in df["store_name"].dropna().unique() if n != "Unknown Store"}
    pivot = store_category_mix(df, anonymize=True)
    for real in real_names:
        assert real not in pivot.index, f"Real store name in mix index: {real!r}"
    assert all(idx.startswith("Store ") for idx in pivot.index)


def test_store_labels_a_is_highest_revenue(df):
    labels = store_labels(df)
    # Store A must be the highest-revenue store
    tbl = revenue_by_store(df, anonymize=False)
    top_store = tbl["store_name"].iloc[0]
    assert labels[top_store] == "Store A"


def test_anonymized_category_seasonality_rows_sum_to_100(df):
    pivot = category_seasonality(df, anonymize=True)
    row_sums = pivot.sum(axis=1)
    assert (abs(row_sums - 100) < 0.01).all()
