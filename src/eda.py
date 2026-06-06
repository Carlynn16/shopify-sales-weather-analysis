"""
Sales EDA helpers for Block A.  All plot functions save a PNG and return the figure.

anonymize=True (default for all public functions)
    • Store names  → 'Store A' … 'Store G' (ranked by revenue, derived at runtime)
    • Revenue/units → % of total or revenue index max=100
    • Axis labels say '% of total revenue' or 'Revenue index (max=100)' — never DKK
anonymize=False
    • Original absolute values; for local use only, never committed as output.
"""
import calendar
from zoneinfo import ZoneInfo

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns

from src.anonymize import pct_of_total, revenue_index, store_labels
from src.config import FIGURES_DIR

# ── Palette constants ────────────────────────────────────────────────────────
_BLUES    = "Blues_r"
_CAT_PAL  = "Set2"
_BAR_80   = "#2c7bb6"
_BAR_TAIL = "#abd9e9"
_LINE_RED = "#d7191c"

_TZ       = ZoneInfo("Europe/Copenhagen")
_WEEKDAYS = list(calendar.day_name)
_MONTHS   = list(calendar.month_abbr)[1:]


# ── Style ────────────────────────────────────────────────────────────────────

def setup_style() -> None:
    """Apply the project-wide plot theme.  Called once at import."""
    sns.set_theme(
        style="white",
        font_scale=1.15,
        rc={
            "axes.spines.top":   False,
            "axes.spines.right": False,
            "axes.grid":         True,
            "grid.linestyle":    "--",
            "grid.alpha":        0.45,
            "grid.color":        "#cccccc",
            "figure.dpi":        150,
        },
    )


setup_style()


# ── Save helper ──────────────────────────────────────────────────────────────

def save_fig(fig: plt.Figure, name: str) -> None:
    """Save *fig* to FIGURES_DIR/{name}.png at 150 DPI."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES_DIR / f"{name}.png", dpi=150, bbox_inches="tight")


# ── Internal helpers ─────────────────────────────────────────────────────────

def _add_local_time(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return df (copy) with temporal columns in Europe/Copenhagen local time.
    Safe to call repeatedly — no-op if columns already present.
    """
    if "hour" in df.columns:
        return df
    df = df.copy()
    local = df["created_at"].dt.tz_convert(_TZ)
    df["created_at_local"] = local
    df["hour"]         = local.dt.hour
    df["weekday"]      = local.dt.dayofweek
    df["weekday_name"] = local.dt.day_name()
    df["month_num"]    = local.dt.month
    df["year"]         = local.dt.year
    df["month_period"] = pd.PeriodIndex(local.values.astype("datetime64[ns]"), freq="M")
    return df


def _strip_prefix(name: str) -> str:
    return name.replace("the client ", "")


# ── Top-products ──────────────────────────────────────────────────────────────

def top_products(
    df: pd.DataFrame,
    by: str = "revenue",
    n: int = 15,
    anonymize: bool = True,
) -> pd.DataFrame:
    """
    Ranked product table.

    Columns: name, units, revenue, pct_revenue.  Index is 1-based rank.
    anonymize=True  → revenue = % of total revenue; units = % of total units.
    anonymize=False → revenue in DKK; units as count.
    """
    if by not in {"revenue", "units"}:
        raise ValueError(f"by must be 'revenue' or 'units', got {by!r}")

    agg = df.groupby("name", as_index=False).agg(
        units=("quantity", "sum"),
        revenue=("revenue", "sum"),
    )
    total_rev   = agg["revenue"].sum()
    total_units = df["quantity"].sum()
    agg["pct_revenue"] = agg["revenue"] / total_rev * 100

    sort_col = "revenue" if by == "revenue" else "units"
    result = (
        agg.sort_values(sort_col, ascending=False)
        .head(n)
        .reset_index(drop=True)
    )
    result.index = result.index + 1
    result = result[["name", "units", "revenue", "pct_revenue"]].copy()

    if anonymize:
        result["revenue"] = result["revenue"] / total_rev * 100
        result["units"]   = result["units"] / total_units * 100
    return result


def plot_top_products(
    df: pd.DataFrame,
    by: str = "revenue",
    n: int = 15,
    anonymize: bool = True,
) -> plt.Figure:
    """Horizontal bar chart of the top-*n* products ranked by *by*."""
    tbl     = top_products(df, by=by, n=n, anonymize=anonymize)
    val_col = "revenue" if by == "revenue" else "units"
    names_r = tbl["name"].values[::-1]
    vals_r  = tbl[val_col].values[::-1].astype(float)
    pcts_r  = tbl["pct_revenue"].values[::-1]
    xmax    = vals_r.max()

    fig, ax = plt.subplots(figsize=(14, max(6, n * 0.58)))
    colors  = sns.color_palette(_BLUES, n)
    ax.barh(range(n), vals_r, color=colors)
    ax.set_yticks(range(n))
    ax.set_yticklabels(names_r, fontsize=11)

    for i, (val, pct) in enumerate(zip(vals_r, pcts_r)):
        if anonymize:
            label = f"{val:.1f}%"
        elif by == "revenue":
            label = f"{val:,.0f} DKK  ({pct:.1f}%)"
        else:
            label = f"{int(val):,} units"
        ax.text(val + xmax * 0.012, i, label, va="center", ha="left", fontsize=9.5)

    if anonymize:
        xlabel = "% of total revenue" if by == "revenue" else "% of total units"
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.1f}%"))
    else:
        xlabel = "Revenue (DKK)" if by == "revenue" else "Units Sold"
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))

    ax.set_xlim(0, xmax * 1.40)
    ax.set_xlabel(xlabel, labelpad=8, fontsize=12)
    ax.set_ylabel("")
    ax.set_title(
        f"Top {n} Products by {'Revenue' if by == 'revenue' else 'Units Sold'}",
        fontweight="bold", pad=14,
    )
    fig.tight_layout()
    save_fig(fig, f"top_{n}_products_by_{by}")
    return fig


# ── Pareto analysis ───────────────────────────────────────────────────────────

def pareto_analysis(
    df: pd.DataFrame,
    anonymize: bool = True,
) -> tuple[pd.DataFrame, int]:
    """
    Cumulative revenue share across all products.

    Returns (prod_df, n80).
    anonymize=True  → revenue column is an index (max=100); cumulative_share unchanged (already %).
    anonymize=False → revenue in DKK.
    """
    prod = (
        df.groupby("name", as_index=False)["revenue"]
        .sum()
        .sort_values("revenue", ascending=False)
        .reset_index(drop=True)
    )
    total = prod["revenue"].sum()
    prod["cumulative_share"] = prod["revenue"].cumsum() / total * 100
    prod["rank"] = prod.index + 1

    n80 = int(np.searchsorted(prod["cumulative_share"].values, 80.0)) + 1

    if anonymize:
        prod["revenue"] = revenue_index(prod["revenue"])
    return prod, n80


def plot_pareto(
    df: pd.DataFrame,
    n: int = 30,
    anonymize: bool = True,
) -> tuple[plt.Figure, int]:
    """
    Pareto chart: revenue bars + cumulative % line.
    Bars inside the 80 % threshold are darker.  Returns (fig, n80).
    """
    prod, n80 = pareto_analysis(df, anonymize=anonymize)
    display   = prod.head(n)

    fig, ax1  = plt.subplots(figsize=(16, 7))
    bar_colors = [_BAR_80 if i < n80 else _BAR_TAIL for i in range(len(display))]

    ax1.bar(range(len(display)), display["revenue"], color=bar_colors, width=0.75, zorder=2)
    ax1.set_xticks(range(len(display)))
    ax1.set_xticklabels(display["name"], rotation=45, ha="right", fontsize=8.5)

    if anonymize:
        ax1.set_ylabel("Revenue index (max=100)", fontsize=12, labelpad=8)
    else:
        ax1.set_ylabel("Revenue (DKK)", fontsize=12, labelpad=8)
        ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))

    ax2 = ax1.twinx()
    ax2.plot(
        range(len(display)), display["cumulative_share"],
        color=_LINE_RED, linewidth=2.2, marker="o", markersize=4.5, zorder=3,
    )
    ax2.axhline(80, color=_LINE_RED, linestyle="--", linewidth=1.4, alpha=0.7)
    ax2.text(n - 0.5, 81.5, "80 %", color=_LINE_RED, fontsize=11, va="bottom", ha="right")
    ax2.set_ylabel("Cumulative Revenue Share (%)", fontsize=12, labelpad=8)
    ax2.set_ylim(0, 108)
    ax2.grid(False)

    if n80 <= n:
        ax1.axvline(n80 - 0.5, color=_LINE_RED, linestyle=":", linewidth=1.3, alpha=0.55)

    ax1.set_title(
        f"Pareto Analysis — Top {n} Products by Revenue\n"
        f"({n80} products account for 80 % of total revenue)",
        fontweight="bold", pad=14,
    )
    fig.tight_layout()
    save_fig(fig, "pareto_revenue")
    return fig, n80


# ── Category breakdown ────────────────────────────────────────────────────────

def revenue_by_category(
    df: pd.DataFrame,
    anonymize: bool = True,
) -> pd.DataFrame:
    """
    Category summary table.  Columns: name_category, units, revenue, pct_revenue.
    anonymize=True  → revenue = % of total; units = % of total units.
    anonymize=False → revenue in DKK; units as count.
    """
    total_rev   = df["revenue"].sum()
    total_units = df["quantity"].sum()
    cat = df.groupby("name_category", as_index=False).agg(
        units=("quantity", "sum"),
        revenue=("revenue", "sum"),
    )
    cat["pct_revenue"] = cat["revenue"] / total_rev * 100
    cat = cat.sort_values("revenue", ascending=False).reset_index(drop=True)
    cat.index = cat.index + 1

    if anonymize:
        cat["revenue"] = cat["revenue"] / total_rev * 100
        cat["units"]   = cat["units"] / total_units * 100
    return cat


def plot_category_breakdown(
    df: pd.DataFrame,
    anonymize: bool = True,
) -> plt.Figure:
    """Horizontal bar chart of revenue by product category."""
    tbl  = revenue_by_category(df, anonymize=anonymize)
    n    = len(tbl)
    # always use pct_revenue for bar width (it is always % of total)
    vals = tbl["pct_revenue"].values[::-1]
    xmax = vals.max()

    fig, ax = plt.subplots(figsize=(13, max(5, n * 0.72)))
    colors  = sns.color_palette(_CAT_PAL, n)
    ax.barh(range(n), vals, color=colors[::-1])
    ax.set_yticks(range(n))
    ax.set_yticklabels(tbl["name_category"].values[::-1], fontsize=12)

    rev_vals = tbl["revenue"].values[::-1]
    for i, (val, pct) in enumerate(zip(rev_vals, vals)):
        if anonymize:
            label = f"{pct:.1f}%"
        else:
            label = f"{val:,.0f} DKK  ({pct:.1f}%)"
        ax.text(xmax * 0.012 + pct, i, label, va="center", ha="left", fontsize=11)

    ax.set_xlim(0, xmax * 1.45)
    ax.set_xlabel("% of total revenue", labelpad=8, fontsize=12)
    ax.set_ylabel("")
    ax.set_title("Revenue by Product Category", fontweight="bold", pad=14)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.1f}%"))

    fig.tight_layout()
    save_fig(fig, "category_revenue_breakdown")
    return fig


# ── Monthly revenue ───────────────────────────────────────────────────────────

def revenue_by_month(
    df: pd.DataFrame,
    anonymize: bool = True,
) -> pd.DataFrame:
    """
    Revenue by calendar month.  Columns: month_period, month_label, revenue, order_count.
    anonymize=True  → revenue = index max=100; order_count = % of total orders.
    anonymize=False → revenue in DKK; order_count as count.
    """
    df = _add_local_time(df)
    agg = (
        df.groupby("month_period", as_index=False)
        .agg(revenue=("revenue", "sum"), order_count=("order_id", "nunique"))
        .sort_values("month_period")
    )
    agg["month_label"] = agg["month_period"].dt.strftime("%b %Y")

    if anonymize:
        agg["revenue"]     = revenue_index(agg["revenue"])
        agg["order_count"] = pct_of_total(agg["order_count"])
    return agg[["month_period", "month_label", "revenue", "order_count"]]


def plot_monthly_revenue(
    df: pd.DataFrame,
    anonymize: bool = True,
) -> plt.Figure:
    """Bar chart of monthly revenue across the full window."""
    tbl      = revenue_by_month(df, anonymize=anonymize)
    labels   = tbl["month_label"].tolist()
    vals     = tbl["revenue"].values
    peak_idx = int(vals.argmax())
    colors   = [_BAR_80 if i == peak_idx else _BAR_TAIL for i in range(len(tbl))]

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.bar(range(len(tbl)), vals, color=colors, width=0.72, zorder=2)
    ax.set_xticks(range(len(tbl)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=10)

    ylabel = "Revenue index (max=100)" if anonymize else "Revenue (DKK)"
    ax.set_ylabel(ylabel, fontsize=12, labelpad=8)
    if not anonymize:
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))

    ax.text(
        peak_idx, vals[peak_idx] * 1.015, labels[peak_idx],
        ha="center", va="bottom", fontsize=10, fontweight="bold", color=_BAR_80,
    )
    ax.set_title("Monthly Revenue — Jan 2024 to Mar 2025", fontweight="bold", pad=14)
    fig.tight_layout()
    save_fig(fig, "monthly_revenue")
    return fig


# ── Weekday revenue ───────────────────────────────────────────────────────────

def revenue_by_weekday(
    df: pd.DataFrame,
    anonymize: bool = True,
) -> pd.DataFrame:
    """
    Revenue by day of week (Mon → Sun).
    Columns: weekday, weekday_name, revenue, order_count, avg_order_value.
    anonymize=True  → revenue = index max=100; order_count = % of total; aov = index.
    anonymize=False → absolute values.
    """
    df = _add_local_time(df)
    agg = (
        df.groupby("weekday", as_index=False)
        .agg(
            weekday_name=("weekday_name", "first"),
            revenue=("revenue", "sum"),
            order_count=("order_id", "nunique"),
        )
        .sort_values("weekday")
    )
    agg["avg_order_value"] = agg["revenue"] / agg["order_count"]

    if anonymize:
        agg["revenue"]         = revenue_index(agg["revenue"])
        agg["order_count"]     = pct_of_total(agg["order_count"])
        agg["avg_order_value"] = revenue_index(agg["avg_order_value"])
    return agg[["weekday", "weekday_name", "revenue", "order_count", "avg_order_value"]]


def plot_weekday(
    df: pd.DataFrame,
    anonymize: bool = True,
) -> plt.Figure:
    """Bar chart of revenue by weekday, Mon → Sun."""
    tbl      = revenue_by_weekday(df, anonymize=anonymize)
    vals     = tbl["revenue"].values
    labels   = tbl["weekday_name"].tolist()
    peak_idx = int(vals.argmax())
    colors   = [_BAR_80 if i >= peak_idx - 1 else _BAR_TAIL for i in range(7)]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(range(7), vals, color=colors, width=0.65, zorder=2)
    ax.set_xticks(range(7))
    ax.set_xticklabels(labels, fontsize=11)

    ylabel = "Revenue index (max=100)" if anonymize else "Revenue (DKK)"
    ax.set_ylabel(ylabel, fontsize=12, labelpad=8)

    for i, (bar, val) in enumerate(zip(bars, vals)):
        label = f"{val:.0f}" if anonymize else f"{val/1e6:.2f}M"
        ax.text(i, val * 1.008, label, ha="center", va="bottom", fontsize=9.5)

    if not anonymize:
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
    ax.set_title("Revenue by Day of Week (local time)", fontweight="bold", pad=14)
    fig.tight_layout()
    save_fig(fig, "weekday_revenue")
    return fig


# ── Hourly revenue ────────────────────────────────────────────────────────────

def revenue_by_hour(
    df: pd.DataFrame,
    anonymize: bool = True,
) -> pd.DataFrame:
    """
    Revenue by hour of day (Copenhagen local time).
    anonymize=True  → revenue = index max=100; order_count = % of total.
    anonymize=False → absolute values.
    """
    df = _add_local_time(df)
    agg = (
        df.groupby("hour", as_index=False)
        .agg(revenue=("revenue", "sum"), order_count=("order_id", "nunique"))
        .sort_values("hour")
    )
    if anonymize:
        agg["revenue"]     = revenue_index(agg["revenue"])
        agg["order_count"] = pct_of_total(agg["order_count"])
    return agg[["hour", "revenue", "order_count"]]


def plot_hourly(
    df: pd.DataFrame,
    anonymize: bool = True,
) -> plt.Figure:
    """Bar chart of revenue by hour of day (local time)."""
    tbl    = revenue_by_hour(df, anonymize=anonymize)
    colors = [_BAR_80 if 10 <= h <= 18 else _BAR_TAIL for h in tbl["hour"]]

    fig, ax = plt.subplots(figsize=(13, 5))
    ax.bar(tbl["hour"], tbl["revenue"], color=colors, width=0.8, zorder=2)
    ax.set_xticks(range(0, 24, 2))
    ax.set_xticklabels([f"{h:02d}:00" for h in range(0, 24, 2)], fontsize=10)
    ax.set_xlabel("Hour of Day (Europe/Copenhagen)", fontsize=11, labelpad=8)

    ylabel = "Revenue index (max=100)" if anonymize else "Revenue (DKK)"
    ax.set_ylabel(ylabel, fontsize=12, labelpad=8)
    if not anonymize:
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
    ax.set_title("Revenue by Hour of Day (local time)", fontweight="bold", pad=14)
    fig.tight_layout()
    save_fig(fig, "hourly_revenue")
    return fig


# ── Category seasonality ──────────────────────────────────────────────────────

def category_seasonality(
    df: pd.DataFrame,
    anonymize: bool = True,
) -> pd.DataFrame:
    """
    Revenue pivot: rows = categories, cols = month periods.
    anonymize=True  → each row normalised to % of that category's annual revenue.
    anonymize=False → absolute DKK values.
    """
    df = _add_local_time(df)
    long = (
        df.groupby(["name_category", "month_period"], as_index=False)["revenue"]
        .sum()
    )
    pivot = (
        long.pivot(index="name_category", columns="month_period", values="revenue")
        .fillna(0)
    )
    pivot = pivot.loc[pivot.sum(axis=1).sort_values(ascending=False).index]
    pivot.columns = [str(c) for c in pivot.columns]

    if anonymize:
        pivot = pivot.div(pivot.sum(axis=1), axis=0) * 100
    return pivot


def plot_category_seasonality(
    df: pd.DataFrame,
    anonymize: bool = True,
) -> plt.Figure:
    """
    Heatmap: category × month.
    anonymize=True  → % of category's annual revenue (reveals seasonal profile).
    anonymize=False → DKK thousands.
    """
    pivot = category_seasonality(df, anonymize=anonymize)

    if anonymize:
        display_pivot = pivot
        cbar_label    = "% of category annual revenue"
        fmt           = ".0f"
        title_note    = "Each row = % of that category's annual revenue"
    else:
        display_pivot = pivot / 1_000
        cbar_label    = "Revenue (DKK thousands)"
        fmt           = ".0f"
        title_note    = "Revenue in DKK thousands"

    col_labels = [pd.Period(c).strftime("%b %y") for c in display_pivot.columns]

    fig, ax = plt.subplots(figsize=(16, 5))
    sns.heatmap(
        display_pivot,
        ax=ax,
        cmap="YlOrRd",
        annot=True,
        fmt=fmt,
        linewidths=0.4,
        linecolor="#eeeeee",
        cbar_kws={"label": cbar_label, "shrink": 0.7},
        xticklabels=col_labels,
    )
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=9.5)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=11)
    ax.set_title(
        f"Revenue by Category and Month — {title_note}\n"
        "Ice Cream peaks summer; Hot Beverages & Christmas peak winter",
        fontweight="bold", pad=14,
    )
    fig.tight_layout()
    save_fig(fig, "category_seasonality")
    return fig


# ── Store revenue ─────────────────────────────────────────────────────────────

def revenue_by_store(
    df: pd.DataFrame,
    anonymize: bool = True,
) -> pd.DataFrame:
    """
    Revenue, order count, and avg order value per store (excludes Unknown Store).
    Columns: store_name, store_label, revenue, order_count, avg_order_value.
    anonymize=True  → store_label = 'Store A'…'G'; revenue = % of total;
                       order_count = % of total; avg_order_value = index max=100.
    anonymize=False → store_label = short name; absolute values.
    """
    sub = df[df["store_name"] != "Unknown Store"]
    agg = (
        sub.groupby("store_name", as_index=False)
        .agg(revenue=("revenue", "sum"), order_count=("order_id", "nunique"))
        .sort_values("revenue", ascending=False)
        .reset_index(drop=True)
    )
    agg.index = agg.index + 1
    agg["avg_order_value"] = agg["revenue"] / agg["order_count"]

    if anonymize:
        labels = store_labels(df)
        agg["store_label"]     = agg["store_name"].map(labels)
        agg["revenue"]         = pct_of_total(agg["revenue"])
        agg["order_count"]     = pct_of_total(agg["order_count"])
        agg["avg_order_value"] = revenue_index(agg["avg_order_value"])
        # Drop real store names from anonymized output to prevent accidental exposure
        return agg[["store_label", "revenue", "order_count", "avg_order_value"]]
    else:
        agg["store_label"] = agg["store_name"].apply(_strip_prefix)
        return agg[["store_name", "store_label", "revenue", "order_count", "avg_order_value"]]


def plot_store_revenue(
    df: pd.DataFrame,
    anonymize: bool = True,
) -> plt.Figure:
    """Three-panel chart: revenue, order count, and avg order value per store."""
    tbl    = revenue_by_store(df, anonymize=anonymize)
    labels = tbl["store_label"].tolist()[::-1]  # reversed: lowest first for barh
    n      = len(tbl)

    if anonymize:
        metrics = [
            ("revenue",         "% of total revenue",    _BLUES,    ".1f%"),
            ("order_count",     "% of total orders",     "Blues_r", ".1f%"),
            ("avg_order_value", "Avg order value (index max=100)", "Blues_r", ".0f"),
        ]
    else:
        metrics = [
            ("revenue",         "Revenue (DKK)",          _BLUES,    ",.0f"),
            ("order_count",     "Orders",                 "Blues_r", ",.0f"),
            ("avg_order_value", "Avg Order Value (DKK)",  "Blues_r", ",.0f"),
        ]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, (col, xlabel, pal, fmt) in zip(axes, metrics):
        vals   = tbl[col].values[::-1].astype(float)
        colors = sns.color_palette(pal, n)
        bars   = ax.barh(range(n), vals, color=colors)
        ax.set_yticks(range(n))
        ax.set_yticklabels(labels, fontsize=10)
        ax.set_xlabel(xlabel, fontsize=10, labelpad=6)
        if anonymize and "%" in fmt:
            ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.1f}%"))
        else:
            ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
        for i, v in enumerate(vals):
            label = f"{v:{fmt}}" if not anonymize else (f"{v:.1f}%" if "%" in fmt else f"{v:.0f}")
            ax.text(v * 1.01, i, label, va="center", ha="left", fontsize=8.5)
        ax.set_xlim(0, vals.max() * 1.35)
        ax.set_title(xlabel, fontweight="bold", fontsize=10, wrap=True)

    fig.suptitle("Store Performance Overview", fontweight="bold", fontsize=14, y=1.02)
    fig.tight_layout()
    save_fig(fig, "store_revenue")
    return fig


# ── Store × category mix ──────────────────────────────────────────────────────

def store_category_mix(
    df: pd.DataFrame,
    anonymize: bool = True,
) -> pd.DataFrame:
    """
    100 %-normalised revenue share of each category per store.
    anonymize=True  → row index = 'Store A' … 'Store G'.
    anonymize=False → row index = short store names.
    Values are always share (0–100) — no DKK.
    """
    labels = store_labels(df) if anonymize else None
    sub    = df[df["store_name"] != "Unknown Store"].copy()

    if anonymize:
        sub["store_label"] = sub["store_name"].map(labels)
    else:
        sub["store_label"] = sub["store_name"].apply(_strip_prefix)

    long = (
        sub.groupby(["store_label", "name_category"], as_index=False)["revenue"]
        .sum()
    )
    pivot = (
        long.pivot(index="store_label", columns="name_category", values="revenue")
        .fillna(0)
    )
    pivot = pivot.div(pivot.sum(axis=1), axis=0) * 100
    pivot = pivot.sort_values("Ice Cream", ascending=True)
    return pivot


def plot_store_category_mix(
    df: pd.DataFrame,
    anonymize: bool = True,
) -> plt.Figure:
    """100 %-stacked horizontal bar: each store row shows its category revenue mix."""
    pivot  = store_category_mix(df, anonymize=anonymize)
    cats   = pivot.columns.tolist()
    stores = pivot.index.tolist()
    n_cats = len(cats)
    colors = sns.color_palette(_CAT_PAL, n_cats)

    fig, ax = plt.subplots(figsize=(14, 5))
    left = np.zeros(len(stores))
    for cat, color in zip(cats, colors):
        vals = pivot[cat].values
        ax.barh(range(len(stores)), vals, left=left, label=cat, color=color, height=0.65)
        for i, (v, l) in enumerate(zip(vals, left)):
            if v > 8:
                ax.text(l + v / 2, i, f"{v:.0f}%", ha="center", va="center",
                        fontsize=8, color="white", fontweight="bold")
        left += vals

    ax.set_yticks(range(len(stores)))
    ax.set_yticklabels(stores, fontsize=11)
    ax.set_xlim(0, 101)
    ax.set_xlabel("Revenue Share (%)", fontsize=11, labelpad=8)
    ax.set_title("Category Mix by Store (% of store revenue)", fontweight="bold", pad=14)
    ax.legend(
        loc="upper center", bbox_to_anchor=(0.5, -0.12),
        ncol=n_cats, fontsize=9, frameon=False,
    )
    fig.tight_layout()
    save_fig(fig, "store_category_mix")
    return fig
