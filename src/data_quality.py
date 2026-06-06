"""
Data quality analysis: the 'Unknown Product' pattern.

A row is flagged when title == 'Unknown Product', which happens when the
line_item's product_id was null or absent from the Shopify products export,
so the products-table join produced no match.

Key finding: the row rate (~1 %) and unit rate (~12 %) diverge sharply
because unknown-product lines tend to carry very high quantities (bulk /
box purchases entered at the POS without a catalogue-linked SKU).
"""
from zoneinfo import ZoneInfo

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd
import seaborn as sns

from src.anonymize import store_labels
from src.eda import save_fig, setup_style

_TZ           = ZoneInfo("Europe/Copenhagen")
_UNKNOWN_FLAG = "Unknown Product"   # title value that signals a failed products join

setup_style()


# ── Internal helpers ─────────────────────────────────────────────────────────

def _add_month(df: pd.DataFrame) -> pd.DataFrame:
    """Return df (copy) with a month_period column in Copenhagen local time."""
    if "month_period" in df.columns:
        return df
    df = df.copy()
    local = df["created_at"].dt.tz_convert(_TZ)
    df["month_period"] = pd.PeriodIndex(
        local.values.astype("datetime64[ns]"), freq="M"
    )
    return df


def _is_unknown(df: pd.DataFrame) -> pd.Series:
    return df["title"] == _UNKNOWN_FLAG


# ── Summary ───────────────────────────────────────────────────────────────────

def unknown_product_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Overall Unknown Product rate — row-weighted and quantity-weighted.

    Returns a 2-row DataFrame:
        metric | unknown_count | total | rate_pct

    The two rates typically differ significantly because unknown-product
    line items often carry high quantities (bulk sales without a catalogue SKU).
    """
    mask = _is_unknown(df)
    return pd.DataFrame({
        "metric":        ["rows (line items)", "units (quantity-weighted)"],
        "unknown_count": [int(mask.sum()),
                          int(df.loc[mask, "quantity"].sum())],
        "total":         [len(df),
                          int(df["quantity"].sum())],
        "rate_pct":      [mask.mean() * 100,
                          df.loc[mask, "quantity"].sum() / df["quantity"].sum() * 100],
    })


# ── By store ──────────────────────────────────────────────────────────────────

def unknown_by_store(df: pd.DataFrame) -> pd.DataFrame:
    """
    Unknown Product rate per store (anonymized Store A–G).

    Columns: store_label, unknown_rows, total_rows, row_rate,
             unknown_units, total_units, unit_rate.
    Sorted by unit_rate descending so the most-affected store is row 1.
    """
    labels = store_labels(df)
    sub    = df[df["store_name"] != "Unknown Store"].copy()
    sub["_unk"] = _is_unknown(sub)

    rows = []
    for store, grp in sub.groupby("store_name", sort=False):
        unk_r = int(grp["_unk"].sum())
        tot_r = len(grp)
        unk_u = int(grp.loc[grp["_unk"], "quantity"].sum())
        tot_u = int(grp["quantity"].sum())
        rows.append({
            "store_label":  labels.get(store, store),
            "unknown_rows": unk_r,
            "total_rows":   tot_r,
            "row_rate":     unk_r / tot_r if tot_r else 0.0,
            "unknown_units": unk_u,
            "total_units":  tot_u,
            "unit_rate":    unk_u / tot_u if tot_u else 0.0,
        })

    result = (
        pd.DataFrame(rows)
        .sort_values("unit_rate", ascending=False)
        .reset_index(drop=True)
    )
    result.index = result.index + 1
    return result[["store_label", "unknown_rows", "total_rows", "row_rate",
                   "unknown_units", "total_units", "unit_rate"]]


def plot_unknown_by_store(df: pd.DataFrame) -> plt.Figure:
    """Horizontal bar chart: Unknown Product unit-rate per store."""
    tbl    = unknown_by_store(df)
    n      = len(tbl)
    labels = tbl["store_label"].values[::-1]
    rates  = tbl["unit_rate"].values[::-1] * 100

    fig, ax = plt.subplots(figsize=(11, max(4, n * 0.65)))
    colors  = sns.color_palette("Reds_r", n)
    bars    = ax.barh(range(n), rates, color=colors)

    ax.set_yticks(range(n))
    ax.set_yticklabels(labels, fontsize=11)
    for i, (bar, v) in enumerate(zip(bars, rates)):
        ax.text(v + max(rates) * 0.015, i, f"{v:.1f}%",
                va="center", ha="left", fontsize=10)

    ax.set_xlim(0, max(rates) * 1.35 + 0.5)
    ax.set_xlabel("Unknown Product rate (% of units)", fontsize=11, labelpad=8)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0f}%"))
    ax.set_title(
        "'Unknown Product' Rate by Store  (unit-weighted)\n"
        "Row rate is shown in the table; units diverge due to bulk-quantity lines",
        fontweight="bold", pad=14,
    )

    fig.tight_layout()
    save_fig(fig, "dq_unknown_by_store")
    return fig


# ── Over time (all stores) ────────────────────────────────────────────────────

def unknown_over_time(df: pd.DataFrame) -> pd.DataFrame:
    """
    Monthly Unknown Product rate across all stores.

    Columns: month_period, month_label, total_rows, unknown_rows, row_rate,
             total_units, unknown_units, unit_rate.
    """
    df   = _add_month(df)
    rows = []
    for period, grp in df.groupby("month_period", sort=True):
        unk_mask = _is_unknown(grp)
        rows.append({
            "month_period":  period,
            "month_label":   period.strftime("%b %Y"),
            "total_rows":    len(grp),
            "unknown_rows":  int(unk_mask.sum()),
            "row_rate":      unk_mask.mean(),
            "total_units":   int(grp["quantity"].sum()),
            "unknown_units": int(grp.loc[unk_mask, "quantity"].sum()),
            "unit_rate":     (grp.loc[unk_mask, "quantity"].sum() /
                              grp["quantity"].sum() if grp["quantity"].sum() else 0.0),
        })
    return pd.DataFrame(rows)


def plot_unknown_over_time(df: pd.DataFrame) -> plt.Figure:
    """Line chart: monthly Unknown Product row-rate and unit-rate (all stores)."""
    tbl    = unknown_over_time(df)
    labels = tbl["month_label"].tolist()
    xs     = range(len(tbl))

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(xs, tbl["unit_rate"] * 100, color="#c0392b", linewidth=2.2,
            marker="o", markersize=5.5, label="Unit rate", zorder=3)
    ax.plot(xs, tbl["row_rate"] * 100, color="#e8a09a", linewidth=1.6,
            marker="s", markersize=4, linestyle="--", label="Row rate", zorder=2)
    ax.fill_between(xs, tbl["unit_rate"] * 100, alpha=0.10, color="#c0392b")

    ax.set_xticks(xs)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=10)
    ax.set_ylabel("Unknown Product rate (%)", fontsize=11, labelpad=8)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0f}%"))
    ax.legend(fontsize=10, frameon=False)
    ax.set_title(
        "'Unknown Product' Rate Over Time — All Stores\n"
        "Unit rate (solid) vs row rate (dashed): bulk lines inflate the unit count",
        fontweight="bold", pad=14,
    )

    fig.tight_layout()
    save_fig(fig, "dq_unknown_over_time")
    return fig


# ── Most-affected store over time ─────────────────────────────────────────────

def unknown_store_over_time(df: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    """
    Monthly Unknown Product rate for the most-affected store (highest unit_rate).

    Returns (monthly_df, store_label).
    """
    by_store    = unknown_by_store(df)
    worst_label = by_store.iloc[0]["store_label"]

    inv_labels  = {v: k for k, v in store_labels(df).items()}
    worst_store = inv_labels[worst_label]

    df   = _add_month(df)
    sub  = df[df["store_name"] == worst_store]
    rows = []
    for period, grp in sub.groupby("month_period", sort=True):
        unk_mask = _is_unknown(grp)
        rows.append({
            "month_period":  period,
            "month_label":   period.strftime("%b %Y"),
            "total_rows":    len(grp),
            "unknown_rows":  int(unk_mask.sum()),
            "row_rate":      unk_mask.mean(),
            "total_units":   int(grp["quantity"].sum()),
            "unknown_units": int(grp.loc[unk_mask, "quantity"].sum()),
            "unit_rate":     (grp.loc[unk_mask, "quantity"].sum() /
                              grp["quantity"].sum() if grp["quantity"].sum() else 0.0),
        })

    return pd.DataFrame(rows), worst_label


def plot_unknown_store_over_time(df: pd.DataFrame) -> tuple[plt.Figure, str]:
    """
    Line chart: Unknown Product rate over time for the most-affected store.
    Returns (fig, store_label).
    """
    tbl, worst_label = unknown_store_over_time(df)
    labels = tbl["month_label"].tolist()
    xs     = range(len(tbl))

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(xs, tbl["unit_rate"] * 100, color="#c0392b", linewidth=2.2,
            marker="o", markersize=5.5, label="Unit rate", zorder=3)
    ax.plot(xs, tbl["row_rate"] * 100, color="#e8a09a", linewidth=1.6,
            marker="s", markersize=4, linestyle="--", label="Row rate", zorder=2)
    ax.fill_between(xs, tbl["unit_rate"] * 100, alpha=0.10, color="#c0392b")

    ax.set_xticks(xs)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=10)
    ax.set_ylabel("Unknown Product rate (%)", fontsize=11, labelpad=8)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0f}%"))
    ax.legend(fontsize=10, frameon=False)
    ax.set_title(
        f"'Unknown Product' Rate Over Time — {worst_label}\n"
        "Highest-affected store; pattern suggests POS SKU sync issue, later resolved",
        fontweight="bold", pad=14,
    )

    fig.tight_layout()
    save_fig(fig, "dq_unknown_store_over_time")
    return fig, worst_label
