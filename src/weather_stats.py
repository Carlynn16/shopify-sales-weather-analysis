"""
src/weather_stats.py - inferential statistics for Block B (weather x revenue).

Uses daily_panel and store_panel from src/weather.build_weather_panel.
All monetary output is revenue_pct (% of grand-total revenue), no DKK.
Store names are anonymized (Store A-G).
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as ss
import seaborn as sns
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.outliers_influence import variance_inflation_factor
from zoneinfo import ZoneInfo

from src.config import FIGURES_DIR
from src.weather import WEATHER_VARS

_TZ = ZoneInfo("Europe/Copenhagen")

# Weather variables used as predictors (correlation analysis + VIF)
CORR_VARS: list[str] = [
    "temperature_2m_max",
    "apparent_temperature_max",
    "daylight_duration",
    "precipitation_sum",
    "rain_sum",
    "snowfall_sum",
    "windspeed_10m_max",
]

_VAR_LABELS: dict[str, str] = {
    "temperature_2m_max":       "Temp max (C)",
    "apparent_temperature_max": "App. temp max (C)",
    "daylight_duration":        "Daylight (s)",
    "precipitation_sum":        "Precip (mm)",
    "rain_sum":                 "Rain (mm)",
    "snowfall_sum":             "Snowfall (mm)",
    "windspeed_10m_max":        "Wind max (m/s)",
}

_N_BOOT = 1000
_BLOCK_SIZE = 7   # 7-day blocks for autocorrelation-robust bootstrap


# -- Internal helpers ----------------------------------------------------------

def _block_bootstrap_ci(
    x: np.ndarray,
    y: np.ndarray,
    func: Callable[[np.ndarray, np.ndarray], float],
    n_boot: int = _N_BOOT,
    block_size: int = _BLOCK_SIZE,
    seed: int = 42,
) -> tuple[float, float]:
    """
    Block-bootstrap 95% CI for func(x, y) on a time-ordered pair of arrays.
    Consecutive blocks of length block_size are resampled with replacement.
    """
    rng = np.random.default_rng(seed)
    n = len(x)
    n_blocks = int(np.ceil(n / block_size))
    max_start = max(1, n - block_size + 1)
    vals: list[float] = []
    for _ in range(n_boot):
        starts = rng.integers(0, max_start, size=n_blocks)
        idx = np.concatenate([np.arange(s, min(s + block_size, n)) for s in starts])[:n]
        try:
            v = func(x[idx], y[idx])
            if np.isfinite(v):
                vals.append(v)
        except Exception:
            pass
    if len(vals) < 10:
        return float("nan"), float("nan")
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def _two_sample_bootstrap_ci(
    a: np.ndarray,
    b: np.ndarray,
    n_boot: int = _N_BOOT,
    seed: int = 42,
) -> tuple[float, float]:
    """Bootstrap 95% CI for the difference of medians (a - b)."""
    rng = np.random.default_rng(seed)
    diffs = [
        float(
            np.median(rng.choice(a, len(a), replace=True))
            - np.median(rng.choice(b, len(b), replace=True))
        )
        for _ in range(n_boot)
    ]
    return float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))


# -- FDR correction ------------------------------------------------------------

def apply_fdr(p_values: pd.Series, alpha: float = 0.05) -> np.ndarray:
    """Benjamini-Hochberg FDR correction. NaN p-values are treated as 1.0."""
    p_clean = p_values.fillna(1.0).to_numpy(dtype=float)
    _, p_adj, _, _ = multipletests(p_clean, method="fdr_bh", alpha=alpha)
    return p_adj


# -- Category revenue builder --------------------------------------------------

def build_daily_category_revenue(transactions: pd.DataFrame) -> pd.DataFrame:
    """
    Daily revenue_pct for Ice Cream and Hot Beverages.

    Returns DataFrame with columns: date, ice_cream_pct, hot_bev_pct.
    Values are % of grand-total revenue, matching daily_panel.revenue_pct scale.
    """
    tx = transactions.copy()
    tx["date"] = tx["created_at"].dt.tz_convert(_TZ).dt.date
    total_rev = tx["revenue"].sum()

    def _daily_pct(category: str, col: str) -> pd.DataFrame:
        d = (
            tx[tx["name_category"] == category]
            .groupby("date", as_index=False)["revenue"]
            .sum()
            .rename(columns={"revenue": col})
        )
        d[col] = d[col] / total_rev * 100
        return d

    ice = _daily_pct("Ice Cream", "ice_cream_pct")
    hot = _daily_pct("Hot Beverages", "hot_bev_pct")
    return (
        ice.merge(hot, on="date", how="outer")
        .sort_values("date")
        .reset_index(drop=True)
    )


# -- Correlations --------------------------------------------------------------

def weather_correlations(
    daily_panel: pd.DataFrame,
    daily_cat_rev: pd.DataFrame | None = None,
    n_boot: int = _N_BOOT,
    block_size: int = _BLOCK_SIZE,
) -> pd.DataFrame:
    """
    Pearson and Spearman correlations of daily revenue (total, Ice Cream,
    Hot Beverages) vs each CORR_VAR, with block-bootstrap 95% CIs.

    P-values are approximate under autocorrelation; CIs are autocorrelation-robust.

    Returns a tidy DataFrame with columns:
        outcome, weather_var,
        pearson_r,  pearson_p,  pearson_ci_lower,  pearson_ci_upper,  pearson_p_adj,
        spearman_r, spearman_p, spearman_ci_lower, spearman_ci_upper, spearman_p_adj.
    """
    weather_df = daily_panel.set_index("date")[CORR_VARS]

    outcomes: dict[str, pd.Series] = {
        "Total": daily_panel.set_index("date")["revenue_pct"]
    }
    if daily_cat_rev is not None:
        cat = daily_cat_rev.set_index("date")
        if "ice_cream_pct" in cat.columns:
            outcomes["Ice Cream"] = cat["ice_cream_pct"]
        if "hot_bev_pct" in cat.columns:
            outcomes["Hot Beverages"] = cat["hot_bev_pct"]

    rows: list[dict] = []

    for outcome_name, outcome_series in outcomes.items():
        merged = weather_df.join(outcome_series.rename("y"), how="inner").dropna(subset=["y"])

        for var in CORR_VARS:
            pair = merged[["y", var]].dropna().sort_index()
            if len(pair) < 10:
                rows.append({
                    "outcome": outcome_name, "weather_var": var,
                    **{k: float("nan") for k in [
                        "pearson_r", "pearson_p", "pearson_ci_lower", "pearson_ci_upper",
                        "spearman_r", "spearman_p", "spearman_ci_lower", "spearman_ci_upper",
                    ]},
                })
                continue

            x = pair[var].to_numpy(float)
            y = pair["y"].to_numpy(float)

            pr, pp = ss.pearsonr(x, y)
            p_ci_lo, p_ci_hi = _block_bootstrap_ci(
                x, y,
                lambda a, b: ss.pearsonr(a, b)[0],
                n_boot=n_boot, block_size=block_size,
            )

            sr, sp = ss.spearmanr(x, y)
            s_ci_lo, s_ci_hi = _block_bootstrap_ci(
                x, y,
                lambda a, b: ss.spearmanr(a, b).statistic,
                n_boot=n_boot, block_size=block_size,
            )

            rows.append({
                "outcome": outcome_name, "weather_var": var,
                "pearson_r":        round(float(pr), 4),
                "pearson_p":        round(float(pp), 6),
                "pearson_ci_lower": round(p_ci_lo, 4),
                "pearson_ci_upper": round(p_ci_hi, 4),
                "spearman_r":        round(float(sr), 4),
                "spearman_p":        round(float(sp), 6),
                "spearman_ci_lower": round(s_ci_lo, 4),
                "spearman_ci_upper": round(s_ci_hi, 4),
            })

    df = pd.DataFrame(rows)

    # BH FDR across the full family (all outcomes x all vars, both methods)
    all_p = pd.concat([df["pearson_p"], df["spearman_p"]], ignore_index=True)
    p_adj = apply_fdr(all_p)
    n = len(df)
    df["pearson_p_adj"]  = np.round(p_adj[:n], 6)
    df["spearman_p_adj"] = np.round(p_adj[n:], 6)

    return df


# -- Multicollinearity ---------------------------------------------------------

def weather_multicollinearity(
    daily_panel: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Pearson correlation matrix and VIF for CORR_VARS.

    Returns
    -------
    corr_matrix : pd.DataFrame  shape (7x7), Pearson r rounded to 4 dp
    vif_df      : pd.DataFrame  columns ['predictor', 'VIF']
    """
    wx = daily_panel[CORR_VARS].dropna()

    corr_matrix = wx.corr(method="pearson").round(4)

    # Add a constant column so auxiliary regressions have an intercept (standard VIF)
    X = wx.to_numpy(float)
    X_const = np.column_stack([np.ones(len(X)), X])
    vif_df = pd.DataFrame({
        "predictor": CORR_VARS,
        "VIF": [round(variance_inflation_factor(X_const, i + 1), 2)
                for i in range(len(CORR_VARS))],
    })

    return corr_matrix, vif_df


# -- Group comparisons ---------------------------------------------------------

def group_comparisons(
    daily_panel: pd.DataFrame,
    n_boot: int = _N_BOOT,
) -> pd.DataFrame:
    """
    Mann-Whitney U, rank-biserial r, and bootstrap CI on median difference
    for four revenue splits.

    Returns tidy DataFrame:
        split, group_a, group_b, n_a, n_b,
        U_stat, p_value, rank_biserial_r,
        median_a, median_b, median_diff, ci_lower, ci_upper, p_adj.
    """
    df = daily_panel.copy()
    df["month"] = pd.to_datetime(df["date"].astype(str)).dt.month

    temp_med = df["temperature_2m_max"].median()
    day_med  = df["daylight_duration"].median()

    def _compare(name: str, label_a: str, label_b: str, mask: pd.Series) -> dict:
        # mask may contain NaN for rows where the split variable is missing
        valid = df.loc[mask.notna() & df["revenue_pct"].notna()].copy()
        valid["_a"] = mask.loc[valid.index].astype(bool)
        a = valid.loc[valid["_a"],  "revenue_pct"].to_numpy(float)
        b = valid.loc[~valid["_a"], "revenue_pct"].to_numpy(float)

        base = {"split": name, "group_a": label_a, "group_b": label_b,
                "n_a": int(len(a)), "n_b": int(len(b))}
        if len(a) < 3 or len(b) < 3:
            return {**base, **{k: float("nan") for k in [
                "U_stat", "p_value", "rank_biserial_r",
                "median_a", "median_b", "median_diff", "ci_lower", "ci_upper",
            ]}}

        U, p     = ss.mannwhitneyu(a, b, alternative="two-sided")
        r_rb     = float((2 * U) / (len(a) * len(b)) - 1)
        med_diff = float(np.median(a) - np.median(b))
        ci_lo, ci_hi = _two_sample_bootstrap_ci(a, b, n_boot=n_boot)

        return {
            **base,
            "U_stat":           round(float(U), 1),
            "p_value":          round(float(p), 6),
            "rank_biserial_r":  round(r_rb, 4),
            "median_a":         round(float(np.median(a)), 4),
            "median_b":         round(float(np.median(b)), 4),
            "median_diff":      round(med_diff, 4),
            "ci_lower":         round(ci_lo, 4),
            "ci_upper":         round(ci_hi, 4),
        }

    rows = [
        _compare(
            "Rainy vs Dry", "Rainy (>1 mm)", "Dry (<=1 mm)",
            df["precipitation_sum"].gt(1).where(df["precipitation_sum"].notna()),
        ),
        _compare(
            "Warm vs Cold",
            f"Warm (>={temp_med:.1f}C)", f"Cold (<{temp_med:.1f}C)",
            df["temperature_2m_max"].ge(temp_med).where(df["temperature_2m_max"].notna()),
        ),
        _compare(
            "Long vs Short Daylight",
            f"Long (>={day_med/3600:.1f}h)", f"Short (<{day_med/3600:.1f}h)",
            df["daylight_duration"].ge(day_med).where(df["daylight_duration"].notna()),
        ),
        _compare(
            "Summer vs Winter", "Summer (Apr-Sep)", "Winter (Oct-Mar)",
            df["month"].isin([4, 5, 6, 7, 8, 9]).astype(bool),
        ),
    ]

    result = pd.DataFrame(rows)
    result["p_adj"] = np.round(apply_fdr(result["p_value"]), 6)
    return result


# -- Per-store weather sensitivity ---------------------------------------------

def store_weather_sensitivity(store_panel: pd.DataFrame) -> pd.DataFrame:
    """
    Spearman correlation of each store's daily revenue_pct with temperature_2m_max.

    Returns DataFrame sorted by |spearman_r| descending:
        store_label, spearman_r, p_value, n_days, p_adj.
    """
    rows: list[dict] = []
    for label in sorted(store_panel["store_label"].unique()):
        sub = (
            store_panel[store_panel["store_label"] == label]
            [["temperature_2m_max", "revenue_pct"]]
            .dropna()
        )
        if len(sub) < 5:
            rows.append({"store_label": label, "spearman_r": float("nan"),
                         "p_value": float("nan"), "n_days": len(sub)})
            continue
        r, p = ss.spearmanr(sub["temperature_2m_max"], sub["revenue_pct"])
        rows.append({
            "store_label": label,
            "spearman_r":  round(float(r), 4),
            "p_value":     round(float(p), 6),
            "n_days":      len(sub),
        })

    df = (
        pd.DataFrame(rows)
        .assign(_abs_r=lambda d: d["spearman_r"].abs())
        .sort_values("_abs_r", ascending=False)
        .drop(columns=["_abs_r"])
        .reset_index(drop=True)
    )
    df["p_adj"] = np.round(apply_fdr(df["p_value"]), 6)
    return df


# -- Run full analysis ---------------------------------------------------------

def run_analysis(
    daily_panel: pd.DataFrame,
    store_panel: pd.DataFrame,
    transactions: pd.DataFrame,
    n_boot: int = _N_BOOT,
    block_size: int = _BLOCK_SIZE,
) -> dict[str, pd.DataFrame]:
    """
    Run the full weather x revenue analysis.

    Returns a dict with keys:
        correlations, corr_matrix, vif, group_comparisons, store_sensitivity.

    Side-effect: writes three figures to figures/.
    """
    daily_cat = build_daily_category_revenue(transactions)

    corr   = weather_correlations(daily_panel, daily_cat,
                                   n_boot=n_boot, block_size=block_size)
    corr_matrix, vif = weather_multicollinearity(daily_panel)
    groups = group_comparisons(daily_panel, n_boot=n_boot)
    sens   = store_weather_sensitivity(store_panel)

    plot_correlation_heatmap(corr)
    plot_category_scatter(daily_panel, daily_cat)
    plot_group_comparisons(daily_panel)

    return {
        "correlations":      corr,
        "corr_matrix":       corr_matrix,
        "vif":               vif,
        "group_comparisons": groups,
        "store_sensitivity": sens,
    }


# -- Figures -------------------------------------------------------------------

_FIG_RC = {
    "figure.dpi":        150,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.linestyle":    "--",
    "grid.alpha":        0.4,
}


def plot_correlation_heatmap(
    corr_results: pd.DataFrame,
    out_path: Path | None = None,
) -> Path:
    """
    Heatmap of Spearman r (outcomes x weather vars).
    Cells show r to 2 dp; * marks BH-adjusted p < 0.05.
    Saved to figures/weather_revenue_correlations.png.
    """
    out_path = out_path or FIGURES_DIR / "weather_revenue_correlations.png"
    FIGURES_DIR.mkdir(exist_ok=True)

    pivot_r = (
        corr_results
        .pivot(index="outcome", columns="weather_var", values="spearman_r")
        .reindex(columns=CORR_VARS)
    )
    pivot_p = (
        corr_results
        .pivot(index="outcome", columns="weather_var", values="spearman_p_adj")
        .reindex(columns=CORR_VARS)
    )

    annot = pivot_r.copy().astype(object)
    for row in pivot_r.index:
        for col in pivot_r.columns:
            r_val = pivot_r.loc[row, col]
            p_val = pivot_p.loc[row, col]
            star = "*" if pd.notna(p_val) and p_val < 0.05 else ""
            annot.loc[row, col] = f"{r_val:.2f}{star}" if pd.notna(r_val) else "--"

    col_labels = [_VAR_LABELS.get(c, c) for c in CORR_VARS]

    with plt.rc_context(_FIG_RC):
        fig, ax = plt.subplots(figsize=(11, 3.5))
        sns.heatmap(
            pivot_r.astype(float),
            annot=annot, fmt="",
            cmap="RdBu_r", center=0, vmin=-1, vmax=1,
            linewidths=0.5, ax=ax,
            xticklabels=col_labels,
            cbar_kws={"label": "Spearman r", "shrink": 0.8},
        )
        ax.set_title(
            "Spearman r: daily revenue vs weather  (* = BH-adjusted p < 0.05)",
            fontsize=11,
        )
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right", fontsize=9)
        fig.tight_layout()
        fig.savefig(out_path, bbox_inches="tight")
        plt.close(fig)

    return out_path


def plot_category_scatter(
    daily_panel: pd.DataFrame,
    daily_cat_rev: pd.DataFrame,
    out_path: Path | None = None,
) -> Path:
    """
    Scatter of Ice Cream and Hot Beverages daily revenue_pct vs temperature,
    with OLS fit lines.
    Saved to figures/category_weather_scatter.png.
    """
    out_path = out_path or FIGURES_DIR / "category_weather_scatter.png"
    FIGURES_DIR.mkdir(exist_ok=True)

    base = (
        daily_panel[["date", "temperature_2m_max"]]
        .merge(daily_cat_rev, on="date", how="inner")
        .dropna(subset=["temperature_2m_max"])
    )

    categories = [
        ("ice_cream_pct", "Ice Cream",     "#1f77b4"),
        ("hot_bev_pct",   "Hot Beverages", "#d62728"),
    ]

    with plt.rc_context(_FIG_RC):
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
        for ax, (col, label, color) in zip(axes, categories):
            sub = base.dropna(subset=[col])
            x = sub["temperature_2m_max"].to_numpy(float)
            y = sub[col].to_numpy(float)
            ax.scatter(x, y, alpha=0.35, s=18, color=color, linewidths=0)
            slope, intercept, r, p, _ = ss.linregress(x, y)
            x_fit = np.linspace(x.min(), x.max(), 200)
            ax.plot(x_fit, intercept + slope * x_fit,
                    color=color, linewidth=2,
                    label=f"r = {r:.2f}  p = {p:.3f}")
            ax.set_xlabel("Temperature max (C)", fontsize=10)
            ax.set_ylabel("Daily revenue share (%)", fontsize=10)
            ax.set_title(label, fontsize=11)
            ax.legend(fontsize=9)
        fig.suptitle("Daily category revenue vs temperature (all stores combined)",
                     fontsize=12)
        fig.tight_layout()
        fig.savefig(out_path, bbox_inches="tight")
        plt.close(fig)

    return out_path


def plot_group_comparisons(
    daily_panel: pd.DataFrame,
    out_path: Path | None = None,
) -> Path:
    """
    Violin plots for the four revenue group-comparison splits.
    Saved to figures/group_comparison_boxplots.png.
    """
    out_path = out_path or FIGURES_DIR / "group_comparison_boxplots.png"
    FIGURES_DIR.mkdir(exist_ok=True)

    df = daily_panel.copy()
    df["month"] = pd.to_datetime(df["date"].astype(str)).dt.month
    temp_med = df["temperature_2m_max"].median()
    day_med  = df["daylight_duration"].median()

    panels = [
        (
            "Rainy vs Dry",
            df.assign(group=np.where(df["precipitation_sum"] > 1,
                                      "Rainy\n(>1 mm)", "Dry\n(<=1 mm)")),
            ["Dry\n(<=1 mm)", "Rainy\n(>1 mm)"],
            "precipitation_sum",
        ),
        (
            "Warm vs Cold",
            df.assign(group=np.where(
                df["temperature_2m_max"] >= temp_med,
                f"Warm\n(>={temp_med:.0f}C)", f"Cold\n(<{temp_med:.0f}C)",
            )),
            [f"Cold\n(<{temp_med:.0f}C)", f"Warm\n(>={temp_med:.0f}C)"],
            "temperature_2m_max",
        ),
        (
            "Long vs Short Daylight",
            df.assign(group=np.where(
                df["daylight_duration"] >= day_med,
                f"Long\n(>={day_med/3600:.0f}h)", f"Short\n(<{day_med/3600:.0f}h)",
            )),
            [f"Short\n(<{day_med/3600:.0f}h)", f"Long\n(>={day_med/3600:.0f}h)"],
            "daylight_duration",
        ),
        (
            "Summer vs Winter",
            df.assign(group=np.where(
                df["month"].isin([4, 5, 6, 7, 8, 9]),
                "Summer\n(Apr-Sep)", "Winter\n(Oct-Mar)",
            )),
            ["Winter\n(Oct-Mar)", "Summer\n(Apr-Sep)"],
            None,
        ),
    ]

    palette = ["#4878d0", "#ee854a"]

    with plt.rc_context(_FIG_RC):
        fig, axes = plt.subplots(1, 4, figsize=(14, 5))
        for ax, (title, data, order, drop_col) in zip(axes, panels):
            subset_cols = ["revenue_pct"]
            if drop_col:
                subset_cols.append(drop_col)
            valid = data.dropna(subset=subset_cols + ["group"])
            sns.violinplot(
                data=valid, x="group", y="revenue_pct",
                order=order, hue="group", legend=False,
                palette=palette,
                inner="box", ax=ax, linewidth=0.8, cut=0,
            )
            ax.set_title(title, fontsize=10)
            ax.set_xlabel("")
            ax.set_ylabel("Daily revenue share (%)" if ax is axes[0] else "")
        fig.suptitle("Daily revenue distribution by weather condition", fontsize=12)
        fig.tight_layout()
        fig.savefig(out_path, bbox_inches="tight")
        plt.close(fig)

    return out_path


# -- Check ---------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    from src.categorization import categorize
    from src.cleaning import clean_transactions
    from src.config import DATA_DIR
    from src.data_loading import build_transactions, load_raw_tables
    from src.weather import build_weather_panel

    tables = load_raw_tables(DATA_DIR)
    tx     = categorize(clean_transactions(build_transactions(tables)))
    locs   = tables["locations"]

    print("Building weather panels ...")
    store_panel, daily_panel = build_weather_panel(tx, locs)

    print("Running analysis ...")
    results = run_analysis(daily_panel, store_panel, tx)

    _SEP = "=" * 78

    # -- Correlation table ------------------------------------------------------
    print(f"\n{_SEP}")
    print("WEATHER x REVENUE  --  PEARSON & SPEARMAN CORRELATIONS")
    print("(p-values are approximate under autocorrelation; CIs are block-bootstrap)")
    print(_SEP)
    corr = results["correlations"]
    display_corr = corr[[
        "outcome", "weather_var",
        "pearson_r", "pearson_p", "pearson_p_adj",
        "pearson_ci_lower", "pearson_ci_upper",
        "spearman_r", "spearman_p", "spearman_p_adj",
        "spearman_ci_lower", "spearman_ci_upper",
    ]].copy()
    display_corr.columns = [
        "outcome", "weather_var",
        "pear_r", "pear_p", "pear_p_adj", "pear_ci_lo", "pear_ci_hi",
        "spear_r", "spear_p", "spear_p_adj", "spear_ci_lo", "spear_ci_hi",
    ]
    print(display_corr.to_string(index=False))

    # -- VIF table --------------------------------------------------------------
    print(f"\n{_SEP}")
    print("WEATHER PREDICTOR MULTICOLLINEARITY  --  VIF")
    print("(temperature and apparent temperature are highly collinear; VIF >90)")
    print(_SEP)
    print(results["corr_matrix"].to_string())
    print()
    print(results["vif"].to_string(index=False))

    # -- Group comparisons ------------------------------------------------------
    print(f"\n{_SEP}")
    print("GROUP COMPARISONS  --  MANN-WHITNEY U  +  RANK-BISERIAL r")
    print(_SEP)
    gc = results["group_comparisons"][[
        "split", "group_a", "group_b", "n_a", "n_b",
        "U_stat", "p_value", "p_adj", "rank_biserial_r",
        "median_diff", "ci_lower", "ci_upper",
    ]]
    print(gc.to_string(index=False))

    # -- Store sensitivity ------------------------------------------------------
    print(f"\n{_SEP}")
    print("PER-STORE TEMPERATURE SENSITIVITY  --  SPEARMAN r WITH TEMP MAX")
    print(_SEP)
    print(results["store_sensitivity"].to_string(index=False))

    print(f"\nFigures written to {FIGURES_DIR}")
