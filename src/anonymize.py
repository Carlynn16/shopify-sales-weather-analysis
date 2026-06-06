"""
Runtime anonymization helpers.

Store names and absolute financial figures are never hardcoded here — labels
and scales are derived from the data at call time.
"""
import pandas as pd


def store_labels(df: pd.DataFrame) -> dict[str, str]:
    """
    Deterministically map each real store_name to 'Store A', 'Store B', …
    ranked by total revenue (A = highest-revenue store).

    Derived from the data at runtime — no real names are hardcoded.
    'Unknown Store' rows are excluded from the ranking.
    """
    rev_rank = (
        df[df["store_name"] != "Unknown Store"]
        .groupby("store_name")["revenue"]
        .sum()
        .sort_values(ascending=False)
    )
    return {name: f"Store {chr(65 + i)}" for i, name in enumerate(rev_rank.index)}


def pct_of_total(series: pd.Series) -> pd.Series:
    """Express each value as % of the series total (result sums to 100)."""
    return series / series.sum() * 100


def revenue_index(series: pd.Series, base: float = 100.0) -> pd.Series:
    """
    Rescale so the maximum = *base* (default 100).

    Preserves the shape of the distribution while hiding absolute magnitude.
    """
    return series / series.max() * base
