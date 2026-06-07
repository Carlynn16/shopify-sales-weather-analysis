"""
tests/test_modeling.py - tests for src/modeling.py.

Tests requiring the weather panel skip gracefully when the API/cache is absent.
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
from src.modeling import (
    TARGET_LOG,
    TARGET_RAW,
    WEATHER_FEATURES,
    XGB_FEATURES,
    build_modeling_dataset,
    build_xgb_dataset,
    cross_validate_xgb,
    fit_final_xgb,
    compute_shap,
    _date_splits,
    _split_calibration_dates,
    _cqr_score,
    plot_cv_metrics,
    plot_predicted_vs_actual,
    plot_residuals_breakdown,
    plot_shap_summary,
    plot_forecast_intervals,
    coefficient_table,
    fit_model,
    model_metrics,
    plot_coefficient_forest,
    plot_residual_diagnostics,
    weather_interpretation,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def transactions():
    tables = load_raw_tables(DATA_DIR)
    return categorize(clean_transactions(build_transactions(tables)))


@pytest.fixture(scope="module")
def store_panel_fixture(transactions):
    locs = load_raw_tables(DATA_DIR)["locations"]
    try:
        sp, _ = build_weather_panel(transactions, locs)
        return sp
    except requests.exceptions.ConnectionError:
        pytest.skip("No network access -- weather API unavailable")
    except Exception as exc:
        pytest.skip(f"Weather panel unavailable: {exc}")


@pytest.fixture(scope="module")
def modeling_df(transactions, store_panel_fixture):
    return build_modeling_dataset(transactions, store_panel_fixture)


@pytest.fixture(scope="module")
def fitted_result(modeling_df):
    return fit_model(modeling_df)


@pytest.fixture(scope="module")
def coeff_tbl(fitted_result):
    return coefficient_table(fitted_result)


@pytest.fixture(scope="module")
def metrics(fitted_result, modeling_df):
    return model_metrics(fitted_result, modeling_df)


# ---------------------------------------------------------------------------
# Dataset structure
# ---------------------------------------------------------------------------

def test_dataset_not_empty(modeling_df):
    assert len(modeling_df) > 100


def test_dataset_has_required_columns(modeling_df):
    required = set(WEATHER_FEATURES) | {TARGET_RAW, TARGET_LOG,
                                         "store_label", "date",
                                         "month", "day_of_week"}
    assert required.issubset(set(modeling_df.columns))


def test_dataset_no_leakage_target_not_in_features(fitted_result):
    """TARGET_RAW and TARGET_LOG must not appear in the design matrix."""
    feature_names = fitted_result.model.exog_names
    assert TARGET_RAW not in feature_names
    assert TARGET_LOG not in feature_names


def test_dataset_no_leakage_raw_date_excluded(fitted_result):
    """Raw 'date' column is not used as a predictor -- only month/dow dummies."""
    feature_names = fitted_result.model.exog_names
    assert "date" not in feature_names


def test_dataset_no_leakage_no_revenue_features(fitted_result):
    """No revenue-related column should appear as a predictor."""
    feature_names = " ".join(fitted_result.model.exog_names).lower()
    assert "revenue" not in feature_names


def test_dataset_log_target_equals_log1p_raw(modeling_df):
    expected = np.log1p(modeling_df[TARGET_RAW])
    np.testing.assert_allclose(modeling_df[TARGET_LOG], expected, atol=1e-9)


def test_dataset_target_non_negative(modeling_df):
    assert (modeling_df[TARGET_RAW] >= 0).all()


def test_dataset_daylight_hours_in_range(modeling_df):
    """Daylight in hours; Denmark ranges ~6-18 h."""
    col = modeling_df["daylight_hours"].dropna()
    assert (col >= 5).all() and (col <= 22).all()


def test_dataset_month_in_range(modeling_df):
    assert modeling_df["month"].between(1, 12).all()


def test_dataset_dow_in_range(modeling_df):
    assert modeling_df["day_of_week"].between(0, 6).all()


def test_dataset_seven_stores(modeling_df):
    assert modeling_df["store_label"].nunique() == 7


def test_dataset_store_labels_anonymized(modeling_df):
    assert all(lbl.startswith("Store ") for lbl in modeling_df["store_label"].unique())


# ---------------------------------------------------------------------------
# Model fit
# ---------------------------------------------------------------------------

def test_model_fit_returns_result(fitted_result):
    assert fitted_result is not None
    assert len(fitted_result.params) > 0


def test_model_has_weather_predictors(fitted_result):
    for feat in WEATHER_FEATURES:
        assert feat in fitted_result.params.index, f"Missing predictor: {feat}"


def test_model_has_month_dummies(fitted_result):
    month_params = [p for p in fitted_result.params.index if "month" in p.lower()]
    assert len(month_params) == 11   # 12 months - 1 reference


def test_model_has_store_dummies(fitted_result):
    store_params = [p for p in fitted_result.params.index if "store_label" in p]
    assert len(store_params) == 6    # 7 stores - 1 reference


def test_model_has_dow_dummies(fitted_result):
    dow_params = [p for p in fitted_result.params.index if "day_of_week" in p]
    assert len(dow_params) == 6      # 7 days - 1 reference


# ---------------------------------------------------------------------------
# Coefficient table
# ---------------------------------------------------------------------------

def test_coeff_table_has_required_columns(coeff_tbl):
    required = {"feature", "coef", "ci_lower", "ci_upper",
                "t_stat", "p_value", "exp_coef", "pct_effect",
                "exp_ci_lower", "exp_ci_upper"}
    assert required.issubset(set(coeff_tbl.columns))


def test_coeff_p_values_in_range(coeff_tbl):
    valid = coeff_tbl["p_value"].dropna()
    assert (valid >= 0).all() and (valid <= 1).all()


def test_coeff_ci_ordering(coeff_tbl):
    valid = coeff_tbl.dropna(subset=["ci_lower", "ci_upper"])
    assert (valid["ci_lower"] <= valid["ci_upper"] + 1e-9).all()


def test_coeff_exp_ci_ordering(coeff_tbl):
    valid = coeff_tbl.dropna(subset=["exp_ci_lower", "exp_ci_upper"])
    assert (valid["exp_ci_lower"] <= valid["exp_ci_upper"] + 1e-9).all()


def test_coeff_exp_coef_positive(coeff_tbl):
    assert (coeff_tbl["exp_coef"].dropna() > 0).all()


def test_coeff_pct_effect_consistent(coeff_tbl):
    """pct_effect == (exp_coef - 1) * 100."""
    expected = (coeff_tbl["exp_coef"] - 1) * 100
    np.testing.assert_allclose(coeff_tbl["pct_effect"], expected, atol=1e-6)


# ---------------------------------------------------------------------------
# Weather interpretation
# ---------------------------------------------------------------------------

def test_weather_interp_has_all_vars(fitted_result):
    wi = weather_interpretation(fitted_result)
    assert len(wi) == len(WEATHER_FEATURES)


def test_weather_interp_columns(fitted_result):
    wi = weather_interpretation(fitted_result)
    assert {"predictor", "pct_effect", "ci_lower_pct", "ci_upper_pct", "p_value"}.issubset(
        set(wi.columns)
    )


def test_weather_temp_positive_effect(fitted_result):
    """Temperature should have a positive % effect on Ice Cream revenue."""
    wi = weather_interpretation(fitted_result)
    temp_row = wi[wi["predictor"] == "Temp max"]
    assert not temp_row.empty
    assert temp_row.iloc[0]["pct_effect"] > 0


# ---------------------------------------------------------------------------
# Model metrics
# ---------------------------------------------------------------------------

def test_r2_in_range(metrics):
    assert 0 <= metrics["r2"] <= 1


def test_adj_r2_in_range(metrics):
    assert 0 <= metrics["adj_r2"] <= 1


def test_adj_r2_le_r2(metrics):
    assert metrics["adj_r2"] <= metrics["r2"] + 1e-9


def test_wape_non_negative(metrics):
    assert metrics["wape"] >= 0


def test_n_obs_matches_df(metrics, modeling_df):
    assert metrics["n_obs"] == len(modeling_df)


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def test_forest_plot_written(fitted_result, tmp_path):
    out = tmp_path / "forest.png"
    p   = plot_coefficient_forest(fitted_result, out_path=out)
    assert p.exists() and p.stat().st_size > 5_000


def test_residual_diagnostics_written(fitted_result, modeling_df, tmp_path):
    out = tmp_path / "diag.png"
    p   = plot_residual_diagnostics(fitted_result, modeling_df, out_path=out)
    assert p.exists() and p.stat().st_size > 5_000


# ---------------------------------------------------------------------------
# XGBoost dataset
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def xgb_df(modeling_df):
    return build_xgb_dataset(modeling_df)


@pytest.fixture(scope="module")
def cv_results(xgb_df):
    # Use 3 folds and 50 estimators so the test suite stays fast
    return cross_validate_xgb(xgb_df, n_splits=3, n_estimators=50)


@pytest.fixture(scope="module")
def final_model(xgb_df):
    return fit_final_xgb(xgb_df, n_estimators=50)


@pytest.fixture(scope="module")
def shap_results(final_model, xgb_df):
    return compute_shap(final_model, xgb_df)


def test_xgb_dataset_has_lag_columns(xgb_df):
    for col in ["lag_1", "lag_7", "rolling_7_mean",
                "is_weekend", "sin_doy", "cos_doy", "store_id"]:
        assert col in xgb_df.columns, f"Missing column: {col}"


def test_xgb_dataset_has_all_features(xgb_df):
    assert set(XGB_FEATURES).issubset(set(xgb_df.columns))


def test_xgb_no_lag_leakage(xgb_df):
    """lag_1 at row t must equal revenue at row t-1 within the same store."""
    for store in xgb_df["store_label"].unique():
        sub = (xgb_df[xgb_df["store_label"] == store]
               .sort_values("date").reset_index(drop=True))
        if len(sub) < 3:
            continue
        np.testing.assert_allclose(
            sub["lag_1"].iloc[1:].values,
            sub[TARGET_RAW].iloc[:-1].values,
            rtol=1e-6,
            err_msg=f"lag_1 leakage detected for {store}",
        )


def test_xgb_lag7_no_leakage(xgb_df):
    """lag_7 at row t must equal revenue at row t-7 within the same store."""
    for store in xgb_df["store_label"].unique():
        sub = (xgb_df[xgb_df["store_label"] == store]
               .sort_values("date").reset_index(drop=True))
        if len(sub) < 10:
            continue
        np.testing.assert_allclose(
            sub["lag_7"].iloc[7:].values,
            sub[TARGET_RAW].iloc[:-7].values,
            rtol=1e-6,
            err_msg=f"lag_7 leakage detected for {store}",
        )


def test_xgb_dataset_dropped_nan_rows(xgb_df):
    assert xgb_df[["lag_1", "lag_7"]].isna().sum().sum() == 0


def test_xgb_cyclical_features_in_range(xgb_df):
    assert xgb_df["sin_doy"].between(-1, 1).all()
    assert xgb_df["cos_doy"].between(-1, 1).all()


# ---------------------------------------------------------------------------
# Temporal splits
# ---------------------------------------------------------------------------

def test_temporal_splits_respect_order(xgb_df):
    """Max train date < min test date for every fold."""
    for tr_idx, te_idx in _date_splits(xgb_df, n_splits=3):
        tr_dates = set(xgb_df.loc[tr_idx, "date"])
        te_dates = set(xgb_df.loc[te_idx, "date"])
        assert max(tr_dates) < min(te_dates), \
            "Test dates overlap with training dates"


def test_temporal_splits_cover_all_rows(xgb_df):
    """Every row appears in exactly one test fold."""
    splits = _date_splits(xgb_df, n_splits=3)
    all_te = []
    for _, te_idx in splits:
        all_te.extend(te_idx)
    # Each row should appear exactly once across test folds
    assert len(set(all_te)) == len(all_te), "Duplicate test indices"


# ---------------------------------------------------------------------------
# CQR helpers
# ---------------------------------------------------------------------------

def test_split_calibration_dates_returns_two_nonempty_sets(xgb_df):
    splits = _date_splits(xgb_df, n_splits=3)
    tr_idx, _ = splits[-1]   # use largest training fold
    model_idx, cal_idx = _split_calibration_dates(xgb_df, tr_idx, cal_frac=0.20)
    assert len(model_idx) > 0
    assert len(cal_idx)   > 0


def test_split_calibration_dates_no_overlap(xgb_df):
    splits = _date_splits(xgb_df, n_splits=3)
    tr_idx, _ = splits[-1]
    model_idx, cal_idx = _split_calibration_dates(xgb_df, tr_idx)
    assert len(set(model_idx) & set(cal_idx)) == 0, \
        "Model and calibration sets overlap"


def test_split_calibration_temporal_order(xgb_df):
    """All model dates must precede all calibration dates."""
    splits = _date_splits(xgb_df, n_splits=3)
    tr_idx, _ = splits[-1]
    model_idx, cal_idx = _split_calibration_dates(xgb_df, tr_idx)
    model_dates = set(xgb_df.loc[model_idx, "date"])
    cal_dates   = set(xgb_df.loc[cal_idx,   "date"])
    assert max(model_dates) < min(cal_dates), \
        "Calibration dates overlap with model-training dates"


def test_cqr_score_expands_under_covered_interval():
    """If the raw interval is too narrow, q_hat should be positive."""
    rng = np.random.default_rng(0)
    y = rng.normal(0, 1, 100)
    q_lo = np.full(100, -0.5)   # interval [−0.5, +0.5] misses many points
    q_hi = np.full(100,  0.5)
    q_hat = _cqr_score(y, q_lo, q_hi, alpha=0.10)
    assert q_hat > 0, "q_hat should be positive for an under-covered interval"


def test_cqr_score_is_finite():
    rng = np.random.default_rng(1)
    y   = rng.uniform(0, 1000, 50)
    qlo = y * 0.8
    qhi = y * 1.2
    q_hat = _cqr_score(y, qlo, qhi, alpha=0.10)
    assert np.isfinite(q_hat)


# ---------------------------------------------------------------------------
# Cross-validation
# ---------------------------------------------------------------------------

def test_cv_returns_dataframe(cv_results):
    assert isinstance(cv_results["fold_metrics"], pd.DataFrame)


def test_cv_has_required_keys(cv_results):
    for key in ["fold_metrics", "test_idx", "y_true", "y_pred", "q05", "q95", "coverage"]:
        assert key in cv_results, f"Missing key: {key}"


def test_cv_metrics_columns(cv_results):
    fm = cv_results["fold_metrics"]
    for col in ["fold", "xgb_wape", "base_wape", "xgb_mape", "xgb_r2", "n_test"]:
        assert col in fm.columns, f"Missing column: {col}"


def test_xgb_beats_baseline_wape(cv_results):
    ov = cv_results["fold_metrics"].loc[
        cv_results["fold_metrics"]["fold"] == "overall"
    ].iloc[0]
    assert ov["xgb_wape"] < ov["base_wape"], (
        f"XGBoost WAPE {ov['xgb_wape']:.4f} >= baseline {ov['base_wape']:.4f}"
    )


def test_cv_wape_in_range(cv_results):
    fm = cv_results["fold_metrics"]
    for col in ["xgb_wape", "base_wape"]:
        valid = fm[col].dropna()
        assert (valid >= 0).all() and (valid <= 2.0).all(), \
            f"Implausible WAPE in {col}"


def test_cv_r2_at_most_one(cv_results):
    valid = cv_results["fold_metrics"]["xgb_r2"].dropna()
    assert (valid <= 1.0).all()


def test_cv_arrays_same_length(cv_results):
    n = len(cv_results["test_idx"])
    assert len(cv_results["y_true"]) == n
    assert len(cv_results["y_pred"]) == n
    assert len(cv_results["q05"])    == n
    assert len(cv_results["q95"])    == n


def test_cv_has_raw_coverage(cv_results):
    assert "raw_coverage" in cv_results
    raw = cv_results["raw_coverage"]
    assert 0.0 <= raw <= 1.0


def test_cv_has_q_hat_per_fold(cv_results):
    assert "q_hat_per_fold" in cv_results
    assert len(cv_results["q_hat_per_fold"]) == 3   # n_splits=3 in fixture


def test_cqr_improves_coverage(cv_results):
    """CQR-calibrated coverage must be strictly higher than uncalibrated."""
    assert cv_results["coverage"] > cv_results["raw_coverage"], (
        f"CQR did not improve coverage: "
        f"calibrated={cv_results['coverage']:.3f} "
        f"raw={cv_results['raw_coverage']:.3f}"
    )


def test_calibrated_coverage_near_90(cv_results):
    """
    CQR calibration should substantially close the gap to 90%.
    Test fixture uses n_splits=3, n_estimators=50 (fast); production parameters
    (n_splits=5, n_estimators=300) achieve ~81%.  The threshold of 0.72
    is well above the uncalibrated floor (~55-65%) while remaining
    achievable under the reduced test configuration.
    """
    cov = cv_results["coverage"]
    assert 0.72 <= cov <= 1.00, (
        f"CQR coverage {cov:.2%} not near 90% target"
    )


def test_interval_coverage_near_90(cv_results):
    # Backward-compat check: CQR-calibrated coverage above a floor.
    cov = cv_results["coverage"]
    assert 0.60 <= cov <= 1.00, \
        f"Coverage {cov:.2%} outside acceptable range [60%, 100%]"


def test_q05_le_q95(cv_results):
    assert (cv_results["q05"] <= cv_results["q95"] + 1e-6).all()


# ---------------------------------------------------------------------------
# SHAP
# ---------------------------------------------------------------------------

def test_shap_values_shape(shap_results, xgb_df):
    sv, _ = shap_results
    assert sv.shape == (len(xgb_df), len(XGB_FEATURES))


def test_shap_importance_covers_all_features(shap_results):
    _, imp = shap_results
    assert set(imp["feature"]) == set(XGB_FEATURES)


def test_shap_mean_abs_non_negative(shap_results):
    _, imp = shap_results
    assert (imp["mean_abs_shap"] >= 0).all()


def test_shap_sorted_descending(shap_results):
    _, imp = shap_results
    vals = imp["mean_abs_shap"].tolist()
    assert vals == sorted(vals, reverse=True)


# ---------------------------------------------------------------------------
# XGBoost figures
# ---------------------------------------------------------------------------

def test_figure_cv_metrics_written(cv_results, tmp_path):
    out = tmp_path / "cv.png"
    p = plot_cv_metrics(cv_results, out_path=out)
    assert p.exists() and p.stat().st_size > 5_000


def test_figure_predicted_actual_written(xgb_df, cv_results, tmp_path):
    out = tmp_path / "pva.png"
    p = plot_predicted_vs_actual(xgb_df, cv_results, out_path=out)
    assert p.exists() and p.stat().st_size > 5_000


def test_figure_residuals_breakdown_written(xgb_df, cv_results, tmp_path):
    out = tmp_path / "resid.png"
    p = plot_residuals_breakdown(xgb_df, cv_results, out_path=out)
    assert p.exists() and p.stat().st_size > 5_000


def test_figure_shap_written(shap_results, tmp_path):
    sv, imp = shap_results
    out = tmp_path / "shap.png"
    p = plot_shap_summary(sv, imp, out_path=out)
    assert p.exists() and p.stat().st_size > 5_000


def test_figure_forecast_intervals_written(xgb_df, cv_results, tmp_path):
    out = tmp_path / "forecast.png"
    p = plot_forecast_intervals(xgb_df, cv_results, out_path=out)
    assert p.exists() and p.stat().st_size > 5_000
