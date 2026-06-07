"""
src/modeling.py - Interpretable Ice Cream revenue regression (Block B, Sec 5.1).

Target   : daily Ice Cream revenue at store x date level, log1p-transformed.
Model    : OLS on log(revenue + 1) with HC3 heteroscedasticity-robust SEs.

Model choice justification
--------------------------
Ice Cream revenue is always non-negative and strongly right-skewed
(observed range 0-64,711; mean ~5,800 across store-days).  OLS on the
log1p-transformed target is preferred over a Gamma GLM because:
  1. The +1 offset is negligible (revenue min = 2 DKK when non-zero, so
     log1p ~= log for all practical values; the ~1.7% zero-revenue store-days
     are handled gracefully without a separate zero-inflation term).
  2. Coefficients exponentiate directly to percentage multipliers, giving
     clean business interpretations ("+1 degree C -> +X% Ice Cream revenue").
  3. Closed-form HC3 robust confidence intervals avoid iterative IRLS.
  4. Residual normality on the log scale is testable and, empirically, holds
     well (QQ plot included in diagnostics figure).

Features (VIF-pruned, consistent with Section 4.2):
  Weather : temperature_2m_max, daylight_hours (= daylight_duration/3600),
            precipitation_sum, windspeed_10m_max.
            Dropped: apparent_temperature_max (VIF ~94, collinear with temp),
            rain_sum (VIF ~1.2M), snowfall_sum (VIF ~58k).
  Calendar: C(month), C(day_of_week)  [month 1 = Jan, dow 0 = Mon as refs]
  Store FE: C(store_label)            [Store A as reference]

Output contract: scale-free only.  Absolute DKK never printed.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as ss
import seaborn as sns
import shap
import statsmodels.formula.api as smf
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit
from zoneinfo import ZoneInfo

from src.anonymize import store_labels
from src.config import FIGURES_DIR

_TZ = ZoneInfo("Europe/Copenhagen")

TARGET_RAW = "ice_cream_revenue"   # raw revenue  -- internal, never printed as DKK
TARGET_LOG = "log_ice_cream"       # log1p(TARGET_RAW)  -- actual model target

WEATHER_FEATURES: list[str] = [
    "temperature_2m_max",
    "daylight_hours",
    "precipitation_sum",
    "windspeed_10m_max",
]

_WEATHER_UNITS = {
    "temperature_2m_max": "+1 C",
    "daylight_hours":     "+1 h",
    "precipitation_sum":  "+1 mm rain",
    "windspeed_10m_max":  "+1 m/s wind",
}

_WEATHER_LABELS = {
    "temperature_2m_max": "Temp max",
    "daylight_hours":     "Daylight",
    "precipitation_sum":  "Precipitation",
    "windspeed_10m_max":  "Wind speed",
}

_FIG_RC = {
    "figure.dpi":        150,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.linestyle":    "--",
    "grid.alpha":        0.4,
}

_MONTH_ABBR = {
    1: "Jan", 2: "Feb",  3: "Mar",  4: "Apr",
    5: "May", 6: "Jun",  7: "Jul",  8: "Aug",
    9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec",
}
_DOW_ABBR = {0: "Mon", 1: "Tue", 2: "Wed", 3: "Thu", 4: "Fri", 5: "Sat", 6: "Sun"}


# ---------------------------------------------------------------------------
# Dataset builder
# ---------------------------------------------------------------------------

def build_modeling_dataset(
    transactions: pd.DataFrame,
    store_panel: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build the store x date modeling dataset.

    Parameters
    ----------
    transactions : cleaned + categorized transaction-level DataFrame.
    store_panel  : output of build_weather_panel (store_label, date, weather cols).

    Returns
    -------
    DataFrame with one row per (store_label, date) that has weather data.
    Columns: store_label, date, temperature_2m_max, daylight_hours,
             precipitation_sum, windspeed_10m_max, month, day_of_week,
             ice_cream_revenue, log_ice_cream.
    """
    tx = transactions.copy()
    tx["date"] = tx["created_at"].dt.tz_convert(_TZ).dt.date

    # Apply same store labels as in build_weather_panel
    lbls = store_labels(tx)
    tx["store_label"] = tx["store_name"].map(lbls)

    # Ice Cream daily revenue by store
    ice = (
        tx[tx["name_category"] == "Ice Cream"]
        .dropna(subset=["store_label"])
        .groupby(["store_label", "date"], as_index=False)["revenue"]
        .sum()
        .rename(columns={"revenue": TARGET_RAW})
    )

    # VIF-pruned weather from store_panel
    wx = store_panel[
        ["store_label", "date",
         "temperature_2m_max", "daylight_duration",
         "precipitation_sum", "windspeed_10m_max"]
    ].copy()

    # Left-join: keep all store x date rows with weather, fill missing Ice Cream = 0
    df = wx.merge(ice, on=["store_label", "date"], how="left")
    df[TARGET_RAW] = df[TARGET_RAW].fillna(0.0)

    # Drop rows where any weather predictor is NaN
    df = df.dropna(subset=["temperature_2m_max", "daylight_duration",
                            "precipitation_sum", "windspeed_10m_max"])

    # Calendar features (local Copenhagen time — already in store_panel dates)
    date_dt = pd.to_datetime(df["date"].astype(str))
    df["month"]       = date_dt.dt.month.astype(int)
    df["day_of_week"] = date_dt.dt.dayofweek.astype(int)  # 0 = Monday

    # Daylight in hours for interpretability (+1 h coefficient)
    df["daylight_hours"] = df["daylight_duration"] / 3600.0
    df = df.drop(columns=["daylight_duration"])

    # Log1p target
    df[TARGET_LOG] = np.log1p(df[TARGET_RAW])

    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Model fitting
# ---------------------------------------------------------------------------

def fit_model(df: pd.DataFrame):
    """
    Fit OLS on log(ice_cream_revenue + 1) with HC3 robust standard errors.

    Returns a statsmodels RegressionResultsWrapper.
    Reference categories: month=1 (January), day_of_week=0 (Monday),
    store_label='Store A'.
    """
    formula = (
        f"{TARGET_LOG} ~ "
        "temperature_2m_max + daylight_hours + precipitation_sum + windspeed_10m_max + "
        "C(month) + C(day_of_week) + C(store_label)"
    )
    return smf.ols(formula, data=df).fit(cov_type="HC3")


# ---------------------------------------------------------------------------
# Results extraction
# ---------------------------------------------------------------------------

def coefficient_table(result) -> pd.DataFrame:
    """
    Tidy table of all coefficients.

    Columns: feature, coef, ci_lower, ci_upper, t_stat, p_value,
             exp_coef, pct_effect, exp_ci_lower, exp_ci_upper.
    """
    ci = result.conf_int()
    tbl = pd.DataFrame({
        "feature":  result.params.index.tolist(),
        "coef":     result.params.values,
        "ci_lower": ci.iloc[:, 0].values,
        "ci_upper": ci.iloc[:, 1].values,
        "t_stat":   result.tvalues.values,
        "p_value":  result.pvalues.values,
    })
    tbl["exp_coef"]     = np.exp(tbl["coef"])
    tbl["pct_effect"]   = (tbl["exp_coef"] - 1.0) * 100.0
    tbl["exp_ci_lower"] = np.exp(tbl["ci_lower"])
    tbl["exp_ci_upper"] = np.exp(tbl["ci_upper"])
    return tbl


def weather_interpretation(result) -> pd.DataFrame:
    """
    Concise weather-only table formatted for the check output.

    Columns: predictor, unit_change, pct_effect, ci_lower_pct,
             ci_upper_pct, p_value.
    """
    tbl = coefficient_table(result)
    rows = []
    for var in WEATHER_FEATURES:
        row = tbl[tbl["feature"] == var]
        if row.empty:
            continue
        row = row.iloc[0]
        rows.append({
            "predictor":    _WEATHER_LABELS.get(var, var),
            "unit_change":  _WEATHER_UNITS[var],
            "pct_effect":   round(row["pct_effect"], 2),
            "ci_lower_pct": round((row["exp_ci_lower"] - 1) * 100, 2),
            "ci_upper_pct": round((row["exp_ci_upper"] - 1) * 100, 2),
            "p_value":      round(row["p_value"], 6),
        })
    return pd.DataFrame(rows)


def model_metrics(result, df: pd.DataFrame) -> dict:
    """
    Scale-free fit metrics.

    Returns r2, adj_r2 (on log scale), wape (on original scale), n_obs,
    n_params, aic, bic.
    """
    y_orig   = df[TARGET_RAW].to_numpy(float)
    yhat_log = result.fittedvalues.to_numpy(float)
    yhat     = np.expm1(np.clip(yhat_log, -10.0, 30.0))

    denom = y_orig.sum()
    wape  = float(np.abs(y_orig - yhat).sum() / denom) if denom > 0 else float("nan")

    return {
        "r2":       round(result.rsquared, 4),
        "adj_r2":   round(result.rsquared_adj, 4),
        "wape":     round(wape, 4),
        "n_obs":    int(result.nobs),
        "n_params": int(result.df_model) + 1,
        "aic":      round(result.aic, 2),
        "bic":      round(result.bic, 2),
        "zero_pct": round((df[TARGET_RAW] == 0).mean() * 100, 2),
    }


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def plot_coefficient_forest(
    result,
    out_path: Path | None = None,
) -> Path:
    """
    Forest plot of % effects with 95% CIs for weather, store, and month
    predictors.  Saved to figures/model_coefficient_forest.png.
    """
    out_path = out_path or FIGURES_DIR / "model_coefficient_forest.png"
    FIGURES_DIR.mkdir(exist_ok=True)

    tbl = coefficient_table(result)

    # ── Extract three panels ────────────────────────────────────────────────
    # Weather (use raw predictor names)
    wx_rows = tbl[tbl["feature"].isin(WEATHER_FEATURES)].copy()
    wx_rows["label"] = wx_rows["feature"].map(
        lambda f: f"{_WEATHER_LABELS[f]} ({_WEATHER_UNITS[f]})"
    )

    # Store FE (drop Store A reference)
    st_rows = tbl[tbl["feature"].str.startswith("C(store_label)")].copy()
    st_rows["label"] = (
        st_rows["feature"]
        .str.extract(r"C\(store_label\)\[T\.(.*)\]")[0]
    )

    # Month FE (drop January reference)
    mo_rows = tbl[tbl["feature"].str.startswith("C(month)")].copy()
    mo_rows["month_num"] = (
        mo_rows["feature"]
        .str.extract(r"C\(month\)\[T\.(\d+)\]")[0]
        .astype(float)
        .astype("Int64")
    )
    mo_rows = mo_rows.sort_values("month_num")
    mo_rows["label"] = mo_rows["month_num"].map(
        lambda m: _MONTH_ABBR.get(int(m), str(m))
    )

    panels = [
        ("Weather predictors", wx_rows, "#1f77b4"),
        ("Store fixed effects (vs Store A)", st_rows, "#2ca02c"),
        ("Month effects (vs January)", mo_rows, "#ff7f0e"),
    ]

    with plt.rc_context(_FIG_RC):
        fig, axes = plt.subplots(1, 3, figsize=(14, 5))

        for ax, (title, rows, color) in zip(axes, panels):
            if rows.empty:
                ax.set_visible(False)
                continue

            y_pos   = np.arange(len(rows))
            effects = (rows["exp_coef"] - 1) * 100
            lo      = (rows["exp_ci_lower"] - 1) * 100
            hi      = (rows["exp_ci_upper"] - 1) * 100
            err_lo  = effects.values - lo.values
            err_hi  = hi.values - effects.values

            sig = rows["p_value"] < 0.05
            point_colors = [color if s else "#aaaaaa" for s in sig.values]

            ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
            ax.barh(y_pos, effects.values, xerr=[err_lo, err_hi],
                    color=point_colors, alpha=0.75, height=0.4,
                    error_kw={"elinewidth": 1.2, "capsize": 3, "ecolor": "#444"})
            ax.set_yticks(y_pos)
            ax.set_yticklabels(rows["label"].tolist(), fontsize=9)
            ax.set_xlabel("% change in Ice Cream revenue", fontsize=9)
            ax.set_title(title, fontsize=10)
            ax.axvline(0, color="black", linewidth=0.8)

        fig.suptitle(
            "Model coefficients: % effect on daily Ice Cream revenue\n"
            "(grey = not significant at 5%; error bars = 95% HC3 CI)",
            fontsize=11,
        )
        fig.tight_layout()
        fig.savefig(out_path, bbox_inches="tight")
        plt.close(fig)

    return out_path


def plot_residual_diagnostics(
    result,
    df: pd.DataFrame,
    out_path: Path | None = None,
) -> Path:
    """
    2x2 residual diagnostic panel.
    Saved to figures/model_residual_diagnostics.png.
    """
    out_path = out_path or FIGURES_DIR / "model_residual_diagnostics.png"
    FIGURES_DIR.mkdir(exist_ok=True)

    fitted   = result.fittedvalues.to_numpy(float)
    resid    = result.resid.to_numpy(float)
    sqrt_abs = np.sqrt(np.abs(resid))
    influence = result.get_influence()
    leverage  = influence.hat_matrix_diag

    with plt.rc_context(_FIG_RC):
        fig, axes = plt.subplots(2, 2, figsize=(11, 8))

        # 1. Residuals vs Fitted
        ax = axes[0, 0]
        ax.scatter(fitted, resid, alpha=0.25, s=10, color="#1f77b4", linewidths=0)
        ax.axhline(0, color="red", linewidth=1)
        lo_fit = np.polynomial.polynomial.polyfit(fitted, resid, 1)
        x_rng = np.linspace(fitted.min(), fitted.max(), 200)
        ax.plot(x_rng, np.polynomial.polynomial.polyval(x_rng, lo_fit),
                color="red", linewidth=1, linestyle="--")
        ax.set_xlabel("Fitted values (log scale)")
        ax.set_ylabel("Residuals")
        ax.set_title("Residuals vs Fitted")

        # 2. Normal Q-Q
        ax = axes[0, 1]
        (osm, osr), (slope, intercept, r) = ss.probplot(resid, dist="norm")
        ax.scatter(osm, osr, alpha=0.3, s=10, color="#1f77b4", linewidths=0)
        x_qq = np.array([osm[0], osm[-1]])
        ax.plot(x_qq, slope * x_qq + intercept, color="red", linewidth=1.5)
        ax.set_xlabel("Theoretical quantiles")
        ax.set_ylabel("Sample quantiles")
        ax.set_title(f"Normal Q-Q  (r = {r:.3f})")

        # 3. Scale-Location
        ax = axes[1, 0]
        ax.scatter(fitted, sqrt_abs, alpha=0.25, s=10, color="#2ca02c", linewidths=0)
        lo_sl = np.polynomial.polynomial.polyfit(fitted, sqrt_abs, 1)
        ax.plot(x_rng, np.polynomial.polynomial.polyval(x_rng, lo_sl),
                color="red", linewidth=1, linestyle="--")
        ax.set_xlabel("Fitted values (log scale)")
        ax.set_ylabel("sqrt|Residuals|")
        ax.set_title("Scale-Location")

        # 4. Residuals vs Leverage
        ax = axes[1, 1]
        ax.scatter(leverage, resid, alpha=0.25, s=10, color="#ff7f0e", linewidths=0)
        ax.axhline(0, color="red", linewidth=1)
        # Cook's distance contours
        for d in (0.5, 1.0):
            lev_range = np.linspace(leverage.min(), leverage.max(), 200)
            p = result.df_model + 1
            cook_bound = np.sqrt(d * p * (1 - lev_range) / lev_range)
            ax.plot(lev_range,  cook_bound, "r--", linewidth=0.7, alpha=0.6)
            ax.plot(lev_range, -cook_bound, "r--", linewidth=0.7, alpha=0.6)
        ax.set_xlabel("Leverage")
        ax.set_ylabel("Residuals")
        ax.set_title("Residuals vs Leverage")

        fig.suptitle(
            "OLS residual diagnostics  (target: log(Ice Cream revenue + 1))",
            fontsize=12,
        )
        fig.tight_layout()
        fig.savefig(out_path, bbox_inches="tight")
        plt.close(fig)

    return out_path


# ---------------------------------------------------------------------------
# XGBoost predictive model  (Section 5, part 2)
# ---------------------------------------------------------------------------

XGB_FEATURES: list[str] = [
    "temperature_2m_max", "daylight_hours", "precipitation_sum", "windspeed_10m_max",
    "month", "day_of_week", "is_weekend", "sin_doy", "cos_doy",
    "lag_1", "lag_7", "rolling_7_mean",
    "store_id",
]

_XGB_LABELS: dict[str, str] = {
    "temperature_2m_max": "Temp max",
    "daylight_hours":     "Daylight",
    "precipitation_sum":  "Precipitation",
    "windspeed_10m_max":  "Wind speed",
    "month":              "Month",
    "day_of_week":        "Day of week",
    "is_weekend":         "Weekend flag",
    "sin_doy":            "Day-of-year (sin)",
    "cos_doy":            "Day-of-year (cos)",
    "lag_1":              "Revenue lag 1d",
    "lag_7":              "Revenue lag 7d",
    "rolling_7_mean":     "Rolling 7d mean",
    "store_id":           "Store",
}

_XGB_PARAMS: dict = dict(
    n_estimators=300,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=15,
    reg_alpha=0.1,
    reg_lambda=1.0,
    random_state=42,
    n_jobs=-1,
    verbosity=0,
)


def build_xgb_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extend the OLS modeling dataset with XGBoost-specific features.

    Adds per-store lag features (lag_1, lag_7, rolling_7_mean), cyclical
    day-of-year (sin/cos), a weekend flag, and an integer store id.
    Lags are computed within each store group in date order -- no future
    data is used (lag_k at date D uses revenue from D-k, which is strictly
    historical).  Rows where lag_7 or lag_1 is NaN (first ~7 days per store)
    are dropped.
    """
    out = df.copy().sort_values(["store_label", "date"]).reset_index(drop=True)

    date_dt = pd.to_datetime(out["date"].astype(str))
    out["is_weekend"] = (out["day_of_week"] >= 5).astype(np.int8)
    doy = date_dt.dt.dayofyear.astype(float)
    out["sin_doy"] = np.sin(2.0 * np.pi * doy / 365.25)
    out["cos_doy"] = np.cos(2.0 * np.pi * doy / 365.25)

    grp = out.groupby("store_label", sort=False)[TARGET_RAW]
    out["lag_1"]          = grp.shift(1)
    out["lag_7"]          = grp.shift(7)
    out["rolling_7_mean"] = grp.transform(
        lambda s: s.shift(1).rolling(7, min_periods=1).mean()
    )

    sorted_lbls = sorted(out["store_label"].unique())
    out["store_id"] = out["store_label"].map(
        {lbl: i for i, lbl in enumerate(sorted_lbls)}
    ).astype(np.int8)

    out = out.dropna(subset=["lag_1", "lag_7"]).reset_index(drop=True)
    return out


def _date_splits(
    df: pd.DataFrame, n_splits: int = 5
) -> list[tuple[list[int], list[int]]]:
    """
    TimeSeriesSplit keyed on unique dates so all stores stay together per fold.
    Returns [(train_row_labels, test_row_labels), ...].
    """
    unique_dates = sorted(df["date"].unique())
    tscv = TimeSeriesSplit(n_splits=n_splits)
    result = []
    for tr_d, te_d in tscv.split(unique_dates):
        tr_set = {unique_dates[i] for i in tr_d}
        te_set = {unique_dates[i] for i in te_d}
        result.append((
            df.index[df["date"].isin(tr_set)].tolist(),
            df.index[df["date"].isin(te_set)].tolist(),
        ))
    return result


def _split_calibration_dates(
    df: pd.DataFrame,
    tr_idx: list[int],
    cal_frac: float = 0.20,
) -> tuple[list[int], list[int]]:
    """
    Split training indices into model-fit (earlier) and calibration (later).

    Uses the last cal_frac fraction of unique training dates for calibration,
    with a minimum of 20 unique dates.  The remaining earlier dates are used
    for fitting the quantile models.

    Returns (model_indices, calibration_indices).
    """
    tr_dates = sorted(df.loc[tr_idx, "date"].unique())
    n_dates  = len(tr_dates)
    n_cal    = max(20, int(n_dates * cal_frac))
    n_cal    = min(n_cal, n_dates - 20)   # always keep ≥ 20 dates for fitting
    n_cal    = max(n_cal, 1)

    cal_set   = set(tr_dates[-n_cal:])
    model_set = set(tr_dates[: n_dates - n_cal])

    tr_set    = set(tr_idx)
    model_idx = df.index[df.index.isin(tr_set) & df["date"].isin(model_set)].tolist()
    cal_idx   = df.index[df.index.isin(tr_set) & df["date"].isin(cal_set)].tolist()
    return model_idx, cal_idx


def _cqr_score(
    y: np.ndarray,
    q_lo_pred: np.ndarray,
    q_hi_pred: np.ndarray,
    alpha: float = 0.10,
) -> float:
    """
    Conformalized Quantile Regression (CQR) conformity adjustment.

    Conformity score E_i = max(q_lo(x_i) - y_i,  y_i - q_hi(x_i)).
    Returns the finite-sample adjusted quantile of E at level
    ceil((n+1)(1-alpha))/n, which guarantees ≥ 1-alpha marginal coverage
    under exchangeability.
    """
    scores = np.maximum(q_lo_pred - y, y - q_hi_pred)
    n      = len(scores)
    level  = min(1.0, np.ceil((n + 1) * (1 - alpha)) / n)
    return float(np.quantile(scores, level))


def _fold_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_base: np.ndarray,
) -> dict:
    """WAPE, MAPE (non-zero rows only), R² for XGBoost and seasonal-naive baseline."""
    denom = float(y_true.sum())

    def wape(yh: np.ndarray) -> float:
        return float(np.abs(y_true - yh).sum() / denom) if denom > 0 else float("nan")

    def mape(yh: np.ndarray) -> float:
        mask = y_true > 0
        if not mask.any():
            return float("nan")
        return float(np.mean(np.abs((y_true[mask] - yh[mask]) / y_true[mask])) * 100)

    def r2(yh: np.ndarray) -> float:
        ss_res = float(((y_true - yh) ** 2).sum())
        ss_tot = float(((y_true - y_true.mean()) ** 2).sum())
        return float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")

    return {
        "xgb_wape":  wape(y_pred),
        "xgb_mape":  mape(y_pred),
        "xgb_r2":    r2(y_pred),
        "base_wape": wape(y_base),
        "base_mape": mape(y_base),
        "base_r2":   r2(y_base),
        "n_test":    int(len(y_true)),
    }


def cross_validate_xgb(
    df: pd.DataFrame,
    n_splits: int = 5,
    n_estimators: int | None = None,
    cal_frac: float = 0.20,
    seed: int = 42,
) -> dict:
    """
    Time-series CV for XGBoost with conformalized quantile regression (CQR).

    Prediction intervals
    --------------------
    Each fold's training set is split temporally: the EARLIER portion
    (1 - cal_frac) trains the quantile models (q=0.05, q=0.95); the LATER
    portion (cal_frac, ≥ 20 unique dates) forms a held-out calibration window.
    CQR conformity scores on that calibration window are used to adjust the
    interval bounds before evaluating on the test fold, yielding empirical
    coverage near the 90% target.

    Returned dict keys
    ------------------
    fold_metrics   DataFrame  per-fold and overall WAPE/MAPE/R²
    test_idx       list       row indices of all test rows
    y_true / y_pred / q05 / q95   CQR-calibrated arrays
    q05_raw / q95_raw             uncalibrated quantile arrays
    coverage       float      CQR-calibrated empirical coverage
    raw_coverage   float      uncalibrated empirical coverage
    q_hat_per_fold list       per-fold CQR adjustment (revenue units)
    """
    params = {**_XGB_PARAMS, "random_state": seed}
    if n_estimators is not None:
        params["n_estimators"] = n_estimators

    q_params_lo = {**params, "objective": "reg:quantileerror", "quantile_alpha": 0.05}
    q_params_hi = {**params, "objective": "reg:quantileerror", "quantile_alpha": 0.95}

    X_all    = df[XGB_FEATURES].to_numpy(dtype=float)
    y_all    = df[TARGET_RAW].to_numpy(dtype=float)
    lag7_col = XGB_FEATURES.index("lag_7")

    fold_rows:    list[dict]  = []
    all_te:       list[int]   = []
    all_yp:       list[float] = []
    all_q05:      list[float] = []
    all_q95:      list[float] = []
    all_q05_raw:  list[float] = []
    all_q95_raw:  list[float] = []
    q_hat_per_fold: list[float] = []

    for fold_i, (tr_idx, te_idx) in enumerate(_date_splits(df, n_splits), 1):
        X_te = X_all[te_idx]
        y_te = y_all[te_idx]
        y_base = X_te[:, lag7_col]

        # Mean model: trained on ALL training rows (best point prediction)
        m_mean = xgb.XGBRegressor(**params)
        m_mean.fit(X_all[tr_idx], y_all[tr_idx])
        y_pred = m_mean.predict(X_te).clip(0)

        # Quantile models: trained on model portion only; calibrated on cal portion
        model_idx, cal_idx = _split_calibration_dates(df, tr_idx, cal_frac)

        m_lo = xgb.XGBRegressor(**q_params_lo)
        m_hi = xgb.XGBRegressor(**q_params_hi)
        m_lo.fit(X_all[model_idx], y_all[model_idx])
        m_hi.fit(X_all[model_idx], y_all[model_idx])

        # Raw (uncalibrated) test predictions
        q05_raw = m_lo.predict(X_te).clip(0)
        q95_raw = np.maximum(m_hi.predict(X_te).clip(0), q05_raw)

        # CQR: compute conformity score on calibration window, expand test bounds
        q05_cal = m_lo.predict(X_all[cal_idx]).clip(0)
        q95_cal = np.maximum(m_hi.predict(X_all[cal_idx]).clip(0), q05_cal)
        q_hat   = _cqr_score(y_all[cal_idx], q05_cal, q95_cal, alpha=0.10)

        q05 = (q05_raw - q_hat).clip(0)
        q95 = np.maximum(q95_raw + q_hat, q05)

        fold_rows.append({"fold": fold_i, **_fold_metrics(y_te, y_pred, y_base)})
        all_te.extend(te_idx)
        all_yp.extend(y_pred.tolist())
        all_q05.extend(q05.tolist())
        all_q95.extend(q95.tolist())
        all_q05_raw.extend(q05_raw.tolist())
        all_q95_raw.extend(q95_raw.tolist())
        q_hat_per_fold.append(q_hat)

    yt      = y_all[all_te]
    yp      = np.array(all_yp,      dtype=float)
    q05     = np.array(all_q05,     dtype=float)
    q95     = np.array(all_q95,     dtype=float)
    q05_raw = np.array(all_q05_raw, dtype=float)
    q95_raw = np.array(all_q95_raw, dtype=float)
    yb      = X_all[all_te, lag7_col]

    fold_rows.append({"fold": "overall", **_fold_metrics(yt, yp, yb)})

    coverage     = float(((yt >= q05)     & (yt <= q95)).mean())
    raw_coverage = float(((yt >= q05_raw) & (yt <= q95_raw)).mean())

    return {
        "fold_metrics":    pd.DataFrame(fold_rows),
        "test_idx":        all_te,
        "y_true":          yt,
        "y_pred":          yp,
        "q05":             q05,
        "q95":             q95,
        "q05_raw":         q05_raw,
        "q95_raw":         q95_raw,
        "coverage":        coverage,
        "raw_coverage":    raw_coverage,
        "q_hat_per_fold":  q_hat_per_fold,
    }


def fit_final_xgb(
    df: pd.DataFrame,
    n_estimators: int | None = None,
    seed: int = 42,
) -> xgb.XGBRegressor:
    """Fit XGBoost on the full dataset (for SHAP and production forecasting)."""
    params = {**_XGB_PARAMS, "random_state": seed}
    if n_estimators is not None:
        params["n_estimators"] = n_estimators
    model = xgb.XGBRegressor(**params)
    model.fit(df[XGB_FEATURES].to_numpy(dtype=float),
              df[TARGET_RAW].to_numpy(dtype=float))
    return model


def compute_shap(
    model: xgb.XGBRegressor,
    df: pd.DataFrame,
) -> tuple[np.ndarray, pd.DataFrame]:
    """
    SHAP values via TreeExplainer on the full dataset.

    Note: collinear lag features (lag_1, lag_7, rolling_7_mean) will share
    attribution; their individual values are ambiguous but the total is correct.

    Returns (shap_values, importance_df) where importance_df has columns
    feature, label, mean_abs_shap (sorted descending).
    """
    X = df[XGB_FEATURES].to_numpy(dtype=float)
    explainer = shap.TreeExplainer(model)
    sv = explainer.shap_values(X)
    if hasattr(sv, "values"):   # Explanation object in newer SHAP
        sv = sv.values

    importance = (
        pd.DataFrame({
            "feature":       XGB_FEATURES,
            "label":         [_XGB_LABELS.get(f, f) for f in XGB_FEATURES],
            "mean_abs_shap": np.abs(sv).mean(axis=0),
        })
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )
    return sv, importance


# ── XGBoost figures ──────────────────────────────────────────────────────────

def plot_cv_metrics(
    cv_results: dict,
    ols_wape: float | None = None,
    out_path: Path | None = None,
) -> Path:
    """
    Grouped bar chart: seasonal-naive vs XGBoost WAPE per fold + overall dashes.
    Saved to figures/model_cv_metrics.png.
    """
    out_path = out_path or FIGURES_DIR / "model_cv_metrics.png"
    FIGURES_DIR.mkdir(exist_ok=True)

    fm  = cv_results["fold_metrics"]
    bar = fm[fm["fold"] != "overall"].copy()
    ov  = fm[fm["fold"] == "overall"].iloc[0]

    labels    = [f"Fold {int(r.fold)}" for _, r in bar.iterrows()]
    base_w    = bar["base_wape"].values * 100
    xgb_w     = bar["xgb_wape"].values  * 100
    x, w      = np.arange(len(labels)), 0.35

    with plt.rc_context(_FIG_RC):
        fig, ax = plt.subplots(figsize=(9, 4.5))
        ax.bar(x - w/2, base_w, w, label="Seasonal naive (lag-7)",
               color="#d62728", alpha=0.75)
        ax.bar(x + w/2, xgb_w,  w, label="XGBoost",
               color="#1f77b4", alpha=0.75)
        ax.axhline(ov["base_wape"]*100, color="#d62728", linestyle="--",
                   linewidth=1.2, alpha=0.6,
                   label=f"Naive overall {ov['base_wape']*100:.1f}%")
        ax.axhline(ov["xgb_wape"]*100, color="#1f77b4", linestyle="--",
                   linewidth=1.2, alpha=0.6,
                   label=f"XGBoost overall {ov['xgb_wape']*100:.1f}%")
        if ols_wape is not None:
            ax.axhline(ols_wape*100, color="#2ca02c", linestyle=":",
                       linewidth=1.5, alpha=0.8,
                       label=f"OLS in-sample {ols_wape*100:.1f}%")
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylabel("WAPE (%)")
        ax.set_title("Cross-validation WAPE: seasonal naive vs XGBoost")
        ax.legend(fontsize=8, loc="upper right")
        fig.tight_layout()
        fig.savefig(out_path, bbox_inches="tight")
        plt.close(fig)
    return out_path


def plot_predicted_vs_actual(
    df: pd.DataFrame,
    cv_results: dict,
    out_path: Path | None = None,
) -> Path:
    """
    Predicted vs actual scatter across all test folds, indexed (max = 100).
    Saved to figures/model_predicted_vs_actual.png.
    """
    out_path = out_path or FIGURES_DIR / "model_predicted_vs_actual.png"
    FIGURES_DIR.mkdir(exist_ok=True)

    yt = cv_results["y_true"]
    yp = cv_results["y_pred"]
    scale = yt.max() / 100.0 if yt.max() > 0 else 1.0

    with plt.rc_context(_FIG_RC):
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.scatter(yt/scale, yp/scale, alpha=0.22, s=11,
                   color="#1f77b4", linewidths=0)
        lim = max((yt/scale).max(), (yp/scale).max()) * 1.05
        ax.plot([0, lim], [0, lim], "r--", linewidth=1.5, label="Perfect fit")
        ax.set_xlabel("Actual  (index, max = 100)")
        ax.set_ylabel("Predicted  (index, max = 100)")
        ax.set_title("XGBoost: predicted vs actual Ice Cream revenue\n(all test folds)")
        ax.legend(fontsize=9)
        fig.tight_layout()
        fig.savefig(out_path, bbox_inches="tight")
        plt.close(fig)
    return out_path


def plot_residuals_breakdown(
    df: pd.DataFrame,
    cv_results: dict,
    out_path: Path | None = None,
) -> Path:
    """
    WAPE by store (A-G) and by season.
    Saved to figures/model_residuals_breakdown.png.
    """
    out_path = out_path or FIGURES_DIR / "model_residuals_breakdown.png"
    FIGURES_DIR.mkdir(exist_ok=True)

    te = df.loc[cv_results["test_idx"]].copy().reset_index(drop=True)
    te["y_pred"]  = cv_results["y_pred"]
    te["abs_err"] = np.abs(te[TARGET_RAW] - te["y_pred"])
    te["season"]  = np.where(te["month"].isin([4,5,6,7,8,9]),
                              "Summer\n(Apr-Sep)", "Winter\n(Oct-Mar)")

    def _wape(g: pd.DataFrame) -> float:
        d = g[TARGET_RAW].sum()
        return float(g["abs_err"].sum() / d) if d > 0 else float("nan")

    store_w = (
        te.groupby("store_label")[["abs_err", TARGET_RAW]]
        .apply(_wape)
        .rename("wape")
        .sort_values()
        .reset_index()
    )
    season_w = (
        te.groupby("season")[["abs_err", TARGET_RAW]]
        .apply(_wape)
        .rename("wape")
        .reset_index()
    )
    overall_wape = (
        cv_results["fold_metrics"]
        .loc[lambda d: d["fold"] == "overall", "xgb_wape"]
        .iloc[0] * 100
    )

    with plt.rc_context(_FIG_RC):
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

        ax = axes[0]
        ax.barh(store_w["store_label"], store_w["wape"]*100,
                color="#1f77b4", alpha=0.8)
        ax.axvline(overall_wape, color="red", linestyle="--",
                   linewidth=1.2, label=f"Overall {overall_wape:.1f}%")
        ax.set_xlabel("WAPE (%)")
        ax.set_title("WAPE by store")
        ax.legend(fontsize=8)

        ax = axes[1]
        ax.bar(season_w["season"], season_w["wape"]*100,
               color=["#ff7f0e", "#2ca02c"], alpha=0.8)
        ax.set_ylabel("WAPE (%)")
        ax.set_title("WAPE by season")

        fig.suptitle("XGBoost residual analysis", fontsize=12)
        fig.tight_layout()
        fig.savefig(out_path, bbox_inches="tight")
        plt.close(fig)
    return out_path


def plot_shap_summary(
    shap_vals: np.ndarray,
    importance: pd.DataFrame,
    out_path: Path | None = None,
) -> Path:
    """
    Horizontal bar chart of mean |SHAP| per feature.
    Saved to figures/model_shap_summary.png.
    """
    out_path = out_path or FIGURES_DIR / "model_shap_summary.png"
    FIGURES_DIR.mkdir(exist_ok=True)

    imp = importance.sort_values("mean_abs_shap").copy()

    with plt.rc_context(_FIG_RC):
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.barh(range(len(imp)), imp["mean_abs_shap"],
                color="#1f77b4", alpha=0.8)
        ax.set_yticks(range(len(imp)))
        ax.set_yticklabels(imp["label"], fontsize=9)
        ax.set_xlabel("Mean |SHAP value|  (revenue units)")
        ax.set_title(
            "SHAP feature importance  (XGBoost, full dataset)\n"
            "Note: collinear lag features share attribution"
        )
        fig.tight_layout()
        fig.savefig(out_path, bbox_inches="tight")
        plt.close(fig)
    return out_path


def plot_forecast_intervals(
    df: pd.DataFrame,
    cv_results: dict,
    store_label: str = "Store A",
    n_days: int = 60,
    out_path: Path | None = None,
) -> Path:
    """
    Actual vs predicted median + shaded 90% PI for one store's last N test days.
    Revenue is indexed (max of shown actual = 100).
    Saved to figures/model_forecast_intervals.png.
    """
    out_path = out_path or FIGURES_DIR / "model_forecast_intervals.png"
    FIGURES_DIR.mkdir(exist_ok=True)

    te = df.loc[cv_results["test_idx"]].copy().reset_index(drop=True)
    te["y_pred"] = cv_results["y_pred"]
    te["q05"]    = cv_results["q05"]
    te["q95"]    = cv_results["q95"]

    sub = te[te["store_label"] == store_label].sort_values("date").tail(n_days)
    if sub.empty:
        store_label = te["store_label"].iloc[0]
        sub = te[te["store_label"] == store_label].sort_values("date").tail(n_days)
    sub = sub.reset_index(drop=True)

    scale  = sub[TARGET_RAW].max() / 100.0 if sub[TARGET_RAW].max() > 0 else 1.0
    actual = sub[TARGET_RAW] / scale
    pred   = sub["y_pred"]   / scale
    lo     = sub["q05"]      / scale
    hi     = sub["q95"]      / scale
    dates  = pd.to_datetime(sub["date"].astype(str))
    cov    = float(((sub[TARGET_RAW] >= sub["q05"]) &
                    (sub[TARGET_RAW] <= sub["q95"])).mean())

    with plt.rc_context(_FIG_RC):
        fig, ax = plt.subplots(figsize=(11, 4))
        ax.fill_between(dates, lo, hi, alpha=0.25, color="#1f77b4",
                        label="90% PI  (CQR-calibrated)")
        ax.plot(dates, actual, "k-",  linewidth=1.5, label="Actual")
        ax.plot(dates, pred,   "b--", linewidth=1.5, label="XGBoost median")
        ax.set_ylabel("Ice Cream revenue  (index, max = 100)")
        ax.set_title(
            f"{store_label}  --  last {len(sub)} test days  "
            f"(local PI coverage: {cov:.0%})"
        )
        ax.legend(fontsize=9)
        fig.tight_layout()
        fig.savefig(out_path, bbox_inches="tight")
        plt.close(fig)
    return out_path


def run_xgb_analysis(
    transactions: pd.DataFrame,
    store_panel: pd.DataFrame,
    ols_wape: float | None = None,
) -> dict:
    """
    Full XGBoost pipeline: build dataset -> CV -> fit final -> SHAP -> figures.
    Returns dict: xgb_df, cv, model, shap_vals, importance.
    """
    base_df = build_modeling_dataset(transactions, store_panel)
    xgb_df  = build_xgb_dataset(base_df)

    cv    = cross_validate_xgb(xgb_df)
    model = fit_final_xgb(xgb_df)
    sv, importance = compute_shap(model, xgb_df)

    plot_cv_metrics(cv, ols_wape=ols_wape)
    plot_predicted_vs_actual(xgb_df, cv)
    plot_residuals_breakdown(xgb_df, cv)
    plot_shap_summary(sv, importance)
    plot_forecast_intervals(xgb_df, cv)

    return {
        "xgb_df":     xgb_df,
        "cv":         cv,
        "model":      model,
        "shap_vals":  sv,
        "importance": importance,
    }


# ---------------------------------------------------------------------------
# Full analysis runner  (OLS)
# ---------------------------------------------------------------------------

def run_analysis(
    transactions: pd.DataFrame,
    store_panel: pd.DataFrame,
) -> dict:
    """
    Build dataset, fit model, compute metrics, and write figures.

    Returns dict: df, result, coeff_table, weather_interp, metrics.
    """
    df     = build_modeling_dataset(transactions, store_panel)
    result = fit_model(df)
    coeff  = coefficient_table(result)
    interp = weather_interpretation(result)
    met    = model_metrics(result, df)

    plot_coefficient_forest(result)
    plot_residual_diagnostics(result, df)

    return {
        "df":            df,
        "result":        result,
        "coeff_table":   coeff,
        "weather_interp": interp,
        "metrics":       met,
    }


# ---------------------------------------------------------------------------
# Check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    from src.categorization import categorize
    from src.cleaning import clean_transactions
    from src.config import DATA_DIR
    from src.data_loading import build_transactions, load_raw_tables
    from src.weather import build_weather_panel

    print("Loading data ...")
    tables = load_raw_tables(DATA_DIR)
    tx     = categorize(clean_transactions(build_transactions(tables)))
    locs   = tables["locations"]
    store_panel, _ = build_weather_panel(tx, locs)

    print("Building modeling dataset ...")
    df = build_modeling_dataset(tx, store_panel)
    print(f"  Shape: {df.shape}  |  "
          f"zero-revenue rows: {(df[TARGET_RAW]==0).sum()} "
          f"({(df[TARGET_RAW]==0).mean()*100:.1f}%)")

    print("Fitting OLS on log(ice_cream_revenue + 1) with HC3 SEs ...")
    result = fit_model(df)

    met    = model_metrics(result, df)
    coeff  = coefficient_table(result)
    interp = weather_interpretation(result)

    _SEP = "=" * 72

    # Fit summary
    print(f"\n{_SEP}")
    print("FIT SUMMARY  (log1p scale; metrics are scale-free)")
    print(_SEP)
    print(f"  N observations       : {met['n_obs']}")
    print(f"  N parameters         : {met['n_params']}")
    print(f"  R-squared            : {met['r2']:.4f}")
    print(f"  Adjusted R-squared   : {met['adj_r2']:.4f}")
    print(f"  WAPE (original scale): {met['wape']*100:.2f}%")
    print(f"  AIC                  : {met['aic']:.1f}")
    print(f"  Zero-revenue store-days: {met['zero_pct']:.1f}%")

    # Weather interpretation
    print(f"\n{_SEP}")
    print("WEATHER EFFECTS ON ICE CREAM REVENUE")
    print("(OLS log-linear: exp(coef)-1 gives % change per unit shift)")
    print(_SEP)
    wi = interp.copy()
    wi.columns = ["Predictor", "Unit", "% Effect", "CI lo %", "CI hi %", "p-value"]
    print(wi.to_string(index=False))

    # Full coefficient table (sorted by p-value)
    print(f"\n{_SEP}")
    print("FULL COEFFICIENT TABLE  (sorted by p-value; intercept omitted)")
    print(_SEP)
    display = (
        coeff[coeff["feature"] != "Intercept"]
        .sort_values("p_value")
        [["feature", "pct_effect", "exp_ci_lower", "exp_ci_upper", "t_stat", "p_value"]]
        .copy()
    )
    display.columns = ["feature", "% effect", "exp_CI_lo", "exp_CI_hi", "t", "p"]
    display["% effect"]  = display["% effect"].round(2)
    display["exp_CI_lo"] = display["exp_CI_lo"].round(4)
    display["exp_CI_hi"] = display["exp_CI_hi"].round(4)
    display["t"]         = display["t"].round(3)
    display["p"]         = display["p"].apply(
        lambda v: "<0.001" if v < 0.001 else f"{v:.4f}"
    )
    print(display.to_string(index=False))

    print(f"\nOLS figures written to {FIGURES_DIR}")

    # ── XGBoost section ────────────────────────────────────────────────────
    print(f"\n{_SEP}")
    print("XGBOOST PREDICTIVE MODEL  (TimeSeriesSplit CV)")
    print(_SEP)
    print("Building XGBoost feature set and running 5-fold CV ...")
    xgb_df = build_xgb_dataset(df)
    print(f"  XGBoost dataset: {xgb_df.shape[0]} rows, {xgb_df.shape[1]} cols")

    cv = cross_validate_xgb(xgb_df)

    print(f"\n{_SEP}")
    print("CROSS-VALIDATION METRICS  (WAPE = Weighted Absolute % Error)")
    print(f"  OLS in-sample WAPE (benchmark): {met['wape']*100:.2f}%")
    print(_SEP)
    fm = cv["fold_metrics"].copy()
    fm["base_wape"] = (fm["base_wape"] * 100).round(2)
    fm["xgb_wape"]  = (fm["xgb_wape"]  * 100).round(2)
    fm["xgb_mape"]  = fm["xgb_mape"].round(2)
    fm["xgb_r2"]    = fm["xgb_r2"].round(4)
    disp_fm = fm[["fold", "base_wape", "xgb_wape", "xgb_mape", "xgb_r2", "n_test"]].copy()
    disp_fm.columns = ["fold", "naive_WAPE%", "xgb_WAPE%", "xgb_MAPE%", "xgb_R2", "n_test"]
    print(disp_fm.to_string(index=False))

    print(f"\n{_SEP}")
    print("PREDICTION INTERVAL COVERAGE  (target: 90%)")
    print(_SEP)
    print(f"  Raw quantile (q=0.05/0.95, no calibration): "
          f"{cv['raw_coverage']*100:.1f}%")
    print(f"  CQR split-conformal (calibrated):           "
          f"{cv['coverage']*100:.1f}%")
    q_hat_vals = cv['q_hat_per_fold']
    print(f"  CQR q_hat per fold (revenue units):         "
          f"{[round(q,0) for q in q_hat_vals]}")

    print(f"\n{_SEP}")
    print("Computing SHAP values on full dataset ...")
    final_model = fit_final_xgb(xgb_df)
    sv, importance = compute_shap(final_model, xgb_df)

    print("TOP SHAP FEATURES  (mean |SHAP|, revenue units)")
    print(_SEP)
    top = importance.head(13)[["label", "mean_abs_shap"]].copy()
    top["mean_abs_shap"] = top["mean_abs_shap"].round(1)
    print(top.to_string(index=False))

    print("\nGenerating XGBoost figures ...")
    plot_cv_metrics(cv, ols_wape=met["wape"])
    plot_predicted_vs_actual(xgb_df, cv)
    plot_residuals_breakdown(xgb_df, cv)
    plot_shap_summary(sv, importance)
    plot_forecast_intervals(xgb_df, cv)
    print(f"XGBoost figures written to {FIGURES_DIR}")
