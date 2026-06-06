import pandas as pd
import pytest

from src.categorization import categorize
from src.cleaning import clean_transactions
from src.config import DATA_DIR
from src.data_loading import build_transactions, load_raw_tables

EXPECTED_CATEGORIES = {
    "Ice Cream", "Hot Beverages", "Buns & Bakery", "Chocolate",
    "Gifts & Cards", "Snacks & Nuts", "Christmas", "Others",
}
UNCATEGORIZED_THRESHOLD = 0.01  # flag if > 1% of rows are Uncategorized

LEGACY_CSV = DATA_DIR.parent / "_legacy" / "shopify_data.csv"


@pytest.fixture(scope="module")
def categorized():
    tables = load_raw_tables(DATA_DIR)
    raw = build_transactions(tables)
    cleaned = clean_transactions(raw)
    return categorize(cleaned)


@pytest.fixture(scope="module")
def legacy_dist():
    legacy = pd.read_csv(LEGACY_CSV, usecols=["name", "name_category"])
    return (
        legacy["name_category"]
        .value_counts(normalize=True)
        .rename("legacy_share")
    )


def test_no_null_category(categorized):
    assert categorized["name_category"].notna().all()


def test_expected_categories_present(categorized):
    found = set(categorized["name_category"].unique()) - {"Uncategorized"}
    assert EXPECTED_CATEGORIES.issubset(found), (
        f"Missing categories: {EXPECTED_CATEGORIES - found}"
    )


def test_uncategorized_below_threshold(categorized):
    share = (categorized["name_category"] == "Uncategorized").mean()
    assert share <= UNCATEGORIZED_THRESHOLD, (
        f"Uncategorized share {share:.1%} exceeds threshold {UNCATEGORIZED_THRESHOLD:.1%}"
    )


def test_regression_category_distribution(categorized, legacy_dist):
    """Our category shares must match the legacy within 1 pp per bucket."""
    our_dist = (
        categorized["name_category"]
        .value_counts(normalize=True)
        .rename("our_share")
    )
    comparison = legacy_dist.to_frame().join(our_dist, how="outer").fillna(0)
    diff = (comparison["legacy_share"] - comparison["our_share"]).abs()
    bad = diff[diff > 0.01]
    assert bad.empty, (
        f"Category share drift > 1 pp:\n{comparison.join(diff.rename('abs_diff'))}"
    )


def test_flodeboller_are_chocolate(categorized):
    mask = categorized["name"].str.contains("flødebolle", case=False, na=False)
    # Flødebollekursus are services → Others; actual products → Chocolate
    products = categorized[mask & ~categorized["name"].str.contains("kursus", case=False)]
    assert (products["name_category"] == "Chocolate").all(), (
        f"Expected Chocolate:\n{products[['name','name_category']].drop_duplicates()}"
    )


def test_bars_in_buns_bakery_moved_to_chocolate(categorized):
    # Snackbar - Musli is explicitly Snacks & Nuts (overridden in legacy); everything
    # else with 'bar' that was Buns & Bakery should be Chocolate
    bar_items = categorized[
        categorized["name"].str.contains(r"\bbar\b", case=False, na=False, regex=True)
        & ~categorized["name"].str.contains("Musli", case=False, na=False)
        & ~categorized["name"].str.contains("snackbox", case=False, na=False)
        & ~categorized["name"].str.contains("gavepose", case=False, na=False)
    ]
    non_choc = bar_items[bar_items["name_category"] != "Chocolate"][["name", "name_category"]].drop_duplicates()
    assert non_choc.empty, f"Bar items not in Chocolate:\n{non_choc}"


def test_revenue_ranking_ice_cream_first(categorized):
    revenue_by_cat = (
        categorized.groupby("name_category")["revenue"]
        .sum()
        .sort_values(ascending=False)
    )
    top = revenue_by_cat.index[0]
    assert top == "Ice Cream", f"Expected Ice Cream #1, got {top}"
