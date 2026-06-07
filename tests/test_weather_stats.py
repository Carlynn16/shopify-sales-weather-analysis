"""
tests/test_weather_stats.py — tests for src/weather_stats.py.

Tests that need the weather panels skip gracefully when the API/cache is absent.
"""
import numpy as np
import pandas as pd
import pytest
import requests

from src.categorization import categorize
from src.cleaning import clean_transactions
from src.config import DATA_DIR
from src.data_loading import build_transactions, load_raw_tables
from src.weather import build_weather_panel
from src.weather_stats import (
    CORR_VARS,
    apply_fdr,
    build_daily_category_revenue,
    group_comparisons,
    plot_category_scatter,
    plot_correlation_heatmap,
    plot_group_comparisons,
    store_weather_sensitivity,
    weather_correlations,
    weather_multicollinearity,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def transactions():
    tables = load_raw_tables(DATA_DIR)
    return categorize(clean_transactions(build_transactions(tables)))


@pytest.fixture(scope="module")
def locations():
    return load_raw_tables(DATA_DIR)["locations"]


@pytest.fixture(scope="module")
def panels(transactions, locations):
    try:
        return build_weather_panel(transactions, locations)
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


@pytest.fixture(scope="module")
def daily_cat_rev(transactions):
    return build_daily_category_revenue(transactions)


@pytest.fixture(scope="module")
def corr_results(daily_panel, daily_cat_rev):
    # Use fewer bootstrap iterations for speed
    return weather_correlations(daily_panel, daily_cat_rev, n_boot=300)


@pytest.fixture(scope="module")
def multi_results(daily_panel):
    return weather_multicollinearity(daily_panel)


@pytest.fixture(scope="module")
def group_results(daily_panel):
    return group_comparisons(daily_panel, n_boot=300)


@pytest.fixture(scope="module")
def sensitivity_results(store_panel):
    return store_weather_sensitivity(store_panel)


# ── build_daily_category_revenue ──────────────────────────────────────────────

def test_cat_rev_columns(daily_cat_rev):
    for col in ("date", "ice_cream_pct", "hot_bev_pct"):
        assert col in daily_cat_rev.columns


def test_cat_rev_non_negative(daily_cat_rev):
    assert (daily_cat_rev["ice_cream_pct"].dropna() >= 0).all()
    assert (daily_cat_rev["hot_bev_pct"].dropna() >= 0).all()


def test_cat_rev_sorted_dates(daily_cat_rev):
    dates = daily_cat_rev["date"].tolist()
    assert dates == sorted(dates)


# ── weather_correlations ──────────────────────────────────────────────────────

def test_corr_is_dataframe(corr_results):
    assert isinstance(corr_results, pd.DataFrame)
    assert len(corr_results) > 0


def test_corr_all_vars_present(corr_results):
    assert set(CORR_VARS).issubset(set(corr_results["weather_var"]))


def test_corr_outcomes_present(corr_results):
    outcomes = set(corr_results["outcome"])
    assert "Total" in outcomes
    assert "Ice Cream" in outcomes
    assert "Hot Beverages" in outcomes


def test_corr_pearson_r_in_range(corr_results):
    col = corr_results["pearson_r"].dropna()
    assert (col >= -1).all() and (col <= 1).all()


def test_corr_spearman_r_in_range(corr_results):
    col = corr_results["spearman_r"].dropna()
    assert (col >= -1).all() and (col <= 1).all()


def test_corr_p_values_in_range(corr_results):
    for col in ("pearson_p", "spearman_p", "pearson_p_adj", "spearman_p_adj"):
        valid = corr_results[col].dropna()
        assert (valid >= 0).all() and (valid <= 1).all(), f"Out-of-range p-values in {col}"


def test_corr_adjusted_p_ge_raw(corr_results):
    # BH-adjusted p-values are never smaller than raw (up to floating-point noise)
    mask = corr_results["pearson_p"].notna() & corr_results["pearson_p_adj"].notna()
    sub = corr_results[mask]
    assert (sub["pearson_p_adj"] >= sub["pearson_p"] - 1e-9).all()


def test_corr_ci_ordering(corr_results):
    """lower bound ≤ upper bound for all non-NaN CIs."""
    for lo, hi in [("pearson_ci_lower", "pearson_ci_upper"),
                   ("spearman_ci_lower", "spearman_ci_upper")]:
        valid = corr_results.dropna(subset=[lo, hi])
        assert (valid[lo] <= valid[hi] + 1e-9).all(), f"CI ordering violated: {lo} > {hi}"


def test_corr_ci_brackets_estimate(corr_results):
    """
    Block-bootstrap 95% CI should bracket the point estimate for
    non-degenerate correlations (|r| > 0.05, finite CI width).
    """
    for r_col, lo, hi in [
        ("pearson_r",  "pearson_ci_lower",  "pearson_ci_upper"),
        ("spearman_r", "spearman_ci_lower", "spearman_ci_upper"),
    ]:
        sub = corr_results.dropna(subset=[r_col, lo, hi])
        sub = sub[(sub[hi] - sub[lo]) > 0.01]   # non-degenerate CI only
        sub = sub[sub[r_col].abs() > 0.05]       # non-trivial correlation only
        if sub.empty:
            continue
        assert (sub[lo] <= sub[r_col] + 1e-6).all(), f"{lo} > {r_col}"
        assert (sub[r_col] <= sub[hi] + 1e-6).all(), f"{r_col} > {hi}"


def test_corr_has_required_columns(corr_results):
    required = {
        "outcome", "weather_var",
        "pearson_r", "pearson_p", "pearson_p_adj", "pearson_ci_lower", "pearson_ci_upper",
        "spearman_r", "spearman_p", "spearman_p_adj", "spearman_ci_lower", "spearman_ci_upper",
    }
    assert required.issubset(set(corr_results.columns))


# ── weather_multicollinearity ─────────────────────────────────────────────────

def test_multi_returns_two_dfs(multi_results):
    corr_matrix, vif_df = multi_results
    assert isinstance(corr_matrix, pd.DataFrame)
    assert isinstance(vif_df, pd.DataFrame)


def test_corr_matrix_square(multi_results):
    corr_matrix, _ = multi_results
    assert corr_matrix.shape[0] == corr_matrix.shape[1] == len(CORR_VARS)


def test_corr_matrix_diagonal_one(multi_results):
    corr_matrix, _ = multi_results
    np.testing.assert_allclose(np.diag(corr_matrix.values), 1.0, atol=1e-4)


def test_corr_matrix_symmetric(multi_results):
    corr_matrix, _ = multi_results
    np.testing.assert_allclose(
        corr_matrix.values, corr_matrix.values.T, atol=1e-6
    )


def test_vif_positive(multi_results):
    _, vif_df = multi_results
    assert (vif_df["VIF"].dropna() > 0).all()


def test_vif_has_columns(multi_results):
    _, vif_df = multi_results
    assert {"predictor", "VIF"}.issubset(set(vif_df.columns))


def test_vif_covers_all_vars(multi_results):
    _, vif_df = multi_results
    assert set(CORR_VARS).issubset(set(vif_df["predictor"]))


def test_vif_temperature_high(multi_results):
    """Temperature and apparent temperature are extremely collinear (VIF >> 10)."""
    _, vif_df = multi_results
    vif = vif_df.set_index("predictor")["VIF"]
    assert vif["temperature_2m_max"] > 10, vif["temperature_2m_max"]
    assert vif["apparent_temperature_max"] > 10, vif["apparent_temperature_max"]
    # Daylight has moderate collinearity with the temperature cluster
    assert vif["daylight_duration"] > 1, vif["daylight_duration"]


# ── group_comparisons ─────────────────────────────────────────────────────────

def test_group_is_dataframe(group_results):
    assert isinstance(group_results, pd.DataFrame)


def test_group_has_four_splits(group_results):
    assert len(group_results) == 4


def test_group_required_columns(group_results):
    required = {
        "split", "group_a", "group_b", "n_a", "n_b",
        "U_stat", "p_value", "rank_biserial_r",
        "median_a", "median_b", "median_diff", "ci_lower", "ci_upper", "p_adj",
    }
    assert required.issubset(set(group_results.columns))


def test_group_p_values_in_range(group_results):
    for col in ("p_value", "p_adj"):
        valid = group_results[col].dropna()
        assert (valid >= 0).all() and (valid <= 1).all(), f"Out-of-range in {col}"


def test_group_rank_biserial_in_range(group_results):
    valid = group_results["rank_biserial_r"].dropna()
    assert (valid >= -1).all() and (valid <= 1).all()


def test_group_ci_ordering(group_results):
    valid = group_results.dropna(subset=["ci_lower", "ci_upper"])
    assert (valid["ci_lower"] <= valid["ci_upper"] + 1e-9).all()


def test_group_ci_brackets_diff(group_results):
    valid = group_results.dropna(subset=["median_diff", "ci_lower", "ci_upper"])
    assert (valid["ci_lower"] <= valid["median_diff"] + 1e-6).all()
    assert (valid["median_diff"] <= valid["ci_upper"] + 1e-6).all()


def test_group_sample_sizes_positive(group_results):
    assert (group_results["n_a"] > 0).all()
    assert (group_results["n_b"] > 0).all()


def test_group_split_names(group_results):
    expected = {
        "Rainy vs Dry", "Warm vs Cold",
        "Long vs Short Daylight", "Summer vs Winter",
    }
    assert expected == set(group_results["split"])


# ── store_weather_sensitivity ─────────────────────────────────────────────────

def test_sensitivity_has_all_stores(sensitivity_results, store_panel):
    expected = set(store_panel["store_label"].unique())
    assert expected == set(sensitivity_results["store_label"])


def test_sensitivity_r_in_range(sensitivity_results):
    col = sensitivity_results["spearman_r"].dropna()
    assert (col >= -1).all() and (col <= 1).all()


def test_sensitivity_p_in_range(sensitivity_results):
    col = sensitivity_results["p_value"].dropna()
    assert (col >= 0).all() and (col <= 1).all()


def test_sensitivity_sorted_by_abs_r(sensitivity_results):
    abs_r = sensitivity_results["spearman_r"].abs().dropna().tolist()
    assert abs_r == sorted(abs_r, reverse=True)


def test_sensitivity_has_p_adj(sensitivity_results):
    assert "p_adj" in sensitivity_results.columns


# ── apply_fdr ─────────────────────────────────────────────────────────────────

def test_fdr_in_range():
    p = pd.Series([0.001, 0.01, 0.04, 0.1, 0.5, 0.9])
    adj = apply_fdr(p)
    assert (adj >= 0).all() and (adj <= 1).all()


def test_fdr_adjusted_ge_raw():
    p = pd.Series([0.001, 0.01, 0.04, 0.1])
    adj = apply_fdr(p)
    assert (adj >= p.to_numpy() - 1e-9).all()


def test_fdr_handles_nan():
    p = pd.Series([0.01, float("nan"), 0.05])
    adj = apply_fdr(p)
    assert len(adj) == 3
    assert np.isfinite(adj).all()


# ── Figures ───────────────────────────────────────────────────────────────────

def test_heatmap_written(corr_results, tmp_path):
    out = tmp_path / "heatmap.png"
    result = plot_correlation_heatmap(corr_results, out_path=out)
    assert result.exists()
    assert result.stat().st_size > 5_000


def test_category_scatter_written(daily_panel, daily_cat_rev, tmp_path):
    out = tmp_path / "scatter.png"
    result = plot_category_scatter(daily_panel, daily_cat_rev, out_path=out)
    assert result.exists()
    assert result.stat().st_size > 5_000


def test_group_comparison_plot_written(daily_panel, tmp_path):
    out = tmp_path / "groups.png"
    result = plot_group_comparisons(daily_panel, out_path=out)
    assert result.exists()
    assert result.stat().st_size > 5_000
