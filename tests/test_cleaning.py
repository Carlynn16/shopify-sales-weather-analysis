import pytest
import pandas as pd

from src.cleaning import clean_transactions
from src.config import DATA_DIR
from src.data_loading import build_transactions, load_raw_tables

LEGACY_ROW_COUNT = 454_821
TOLERANCE = 50  # allow tiny drift from minor implementation differences


@pytest.fixture(scope="module")
def cleaned():
    tables = load_raw_tables(DATA_DIR)
    raw = build_transactions(tables)
    return clean_transactions(raw)


def test_row_count_matches_legacy(cleaned):
    """Regression guard: our reimplementation must reproduce the original row count."""
    delta = abs(len(cleaned) - LEGACY_ROW_COUNT)
    assert delta <= TOLERANCE, (
        f"Cleaned row count {len(cleaned):,} differs from legacy {LEGACY_ROW_COUNT:,} "
        f"by {delta} rows (tolerance: {TOLERANCE})"
    )


def test_no_malte_test_store(cleaned):
    assert "Malte TEST" not in cleaned["store_name"].values


def test_status_columns_dropped(cleaned):
    for col in ["financial_status", "fulfillment_status", "cancelled_at", "cancel_reason"]:
        assert col not in cleaned.columns, f"Column {col!r} should have been dropped"


def test_dropped_high_missing_columns_absent(cleaned):
    for col in ["sku", "customer_id", "email", "product_type"]:
        assert col not in cleaned.columns, f"Column {col!r} should have been dropped"


def test_no_indpakning_rows(cleaned):
    assert not (cleaned["tags"] == "Indpakning").any()
    assert not (cleaned["tags"] == "INDPAKNING").any()


def test_tags_uppercased(cleaned):
    assert (cleaned["tags"] == cleaned["tags"].str.upper()).all()


def test_revenue_non_null(cleaned):
    assert cleaned["revenue"].notna().all()


def test_revenue_equals_price_times_quantity(cleaned):
    expected = (cleaned["price"] * cleaned["quantity"]).reset_index(drop=True)
    pd.testing.assert_series_equal(
        cleaned["revenue"].reset_index(drop=True),
        expected,
        check_names=False,
    )


def test_cappuccino_label_normalised(cleaned):
    long_label = "Cappuccino   -   latte   -   cortado   -   flat white"
    assert not (cleaned["name"] == long_label).any()
    assert not (cleaned["title"] == long_label).any()


def test_vendor_no_unknown_alias(cleaned):
    assert not (cleaned["vendor"] == "vendor-unknown").any()


def test_no_null_store_name(cleaned):
    assert cleaned["store_name"].notna().all()
