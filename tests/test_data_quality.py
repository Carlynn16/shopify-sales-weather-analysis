import matplotlib.pyplot as plt
import pytest

from src.categorization import categorize
from src.cleaning import clean_transactions
from src.config import DATA_DIR, FIGURES_DIR
from src.data_loading import build_transactions, load_raw_tables
from src.data_quality import (
    plot_unknown_by_store,
    plot_unknown_over_time,
    plot_unknown_store_over_time,
    unknown_by_store,
    unknown_over_time,
    unknown_product_summary,
    unknown_store_over_time,
)


@pytest.fixture(scope="module")
def df():
    tables = load_raw_tables(DATA_DIR)
    raw = build_transactions(tables)
    cleaned = clean_transactions(raw)
    return categorize(cleaned)


# ── Summary ───────────────────────────────────────────────────────────────────

def test_summary_rates_in_range(df):
    s = unknown_product_summary(df)
    assert (s["rate_pct"] >= 0).all()
    assert (s["rate_pct"] <= 100).all()


def test_summary_unit_rate_exceeds_row_rate(df):
    """Unit rate >> row rate because unknown lines carry large quantities."""
    s = unknown_product_summary(df)
    row_rate  = s.loc[s["metric"] == "rows (line items)",           "rate_pct"].iloc[0]
    unit_rate = s.loc[s["metric"] == "units (quantity-weighted)", "rate_pct"].iloc[0]
    assert unit_rate > row_rate


def test_summary_counts_consistent(df):
    s = unknown_product_summary(df)
    assert int(s.loc[0, "unknown_count"]) <= int(s.loc[0, "total"])
    assert int(s.loc[1, "unknown_count"]) <= int(s.loc[1, "total"])


# ── By store ──────────────────────────────────────────────────────────────────

def test_store_rates_in_range(df):
    t = unknown_by_store(df)
    assert (t["row_rate"]  >= 0).all() and (t["row_rate"]  <= 1).all()
    assert (t["unit_rate"] >= 0).all() and (t["unit_rate"] <= 1).all()


def test_store_labels_anonymized(df):
    t = unknown_by_store(df)
    assert all(lbl.startswith("Store ") for lbl in t["store_label"])


def test_store_counts_consistent_with_overall(df):
    """Per-store unknown_rows sum ≤ overall (Unknown Store rows excluded from per-store)."""
    summary  = unknown_product_summary(df)
    by_store = unknown_by_store(df)
    overall  = int(summary.loc[0, "unknown_count"])   # row-weighted entry
    assert int(by_store["unknown_rows"].sum()) <= overall + 1


def test_store_sorted_by_unit_rate(df):
    t = unknown_by_store(df)
    rates = t["unit_rate"].values
    assert (rates[:-1] >= rates[1:]).all()


def test_worst_store_unit_rate_substantially_higher(df):
    """The most-affected store should have a unit rate well above the median."""
    t = unknown_by_store(df)
    assert t.iloc[0]["unit_rate"] > t["unit_rate"].median() * 2


# ── Over time ─────────────────────────────────────────────────────────────────

def test_monthly_rates_in_range(df):
    m = unknown_over_time(df)
    for col in ("row_rate", "unit_rate"):
        assert (m[col] >= 0).all() and (m[col] <= 1).all()


def test_monthly_covers_full_window(df):
    m = unknown_over_time(df)
    assert 14 <= len(m) <= 16


def test_monthly_chronological(df):
    m = unknown_over_time(df)
    periods = m["month_period"].tolist()
    assert periods == sorted(periods)


def test_worst_store_monthly_rates_in_range(df):
    tbl, label = unknown_store_over_time(df)
    assert label.startswith("Store ")
    assert (tbl["row_rate"]  >= 0).all() and (tbl["row_rate"]  <= 1).all()
    assert (tbl["unit_rate"] >= 0).all() and (tbl["unit_rate"] <= 1).all()


# ── Plots write files ─────────────────────────────────────────────────────────

def test_plot_by_store_saves_file(df):
    fig = plot_unknown_by_store(df)
    plt.close(fig)
    assert (FIGURES_DIR / "dq_unknown_by_store.png").exists()


def test_plot_over_time_saves_file(df):
    fig = plot_unknown_over_time(df)
    plt.close(fig)
    assert (FIGURES_DIR / "dq_unknown_over_time.png").exists()


def test_plot_store_over_time_saves_file(df):
    fig, label = plot_unknown_store_over_time(df)
    plt.close(fig)
    assert (FIGURES_DIR / "dq_unknown_store_over_time.png").exists()
    assert label.startswith("Store ")
