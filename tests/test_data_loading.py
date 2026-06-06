import pandas as pd
import pytest

from src.config import DATA_DIR
from src.data_loading import build_transactions, load_raw_tables


@pytest.fixture(scope="module")
def tables():
    return load_raw_tables(DATA_DIR)


@pytest.fixture(scope="module")
def tx(tables):
    return build_transactions(tables)


def test_row_count_gte_line_items(tables, tx):
    """Merge must not silently drop line_item rows."""
    assert len(tx) >= len(tables["line_items"])


def test_expected_columns_present(tx):
    required = {
        "order_id", "line_item_id", "product_id", "name",
        "quantity", "price", "total_discount",
        "created_at", "processed_at",
        "financial_status", "fulfillment_status",
        "location_id", "store_name", "city",
        "title", "vendor", "tags",
        "refund_amount",
    }
    missing = required - set(tx.columns)
    assert not missing, f"Missing columns: {missing}"


def test_no_null_order_id(tx):
    assert tx["order_id"].notna().all(), "order_id must never be null after merge"


def test_created_at_is_utc_datetime(tx):
    assert pd.api.types.is_datetime64_any_dtype(tx["created_at"])
    assert tx["created_at"].dt.tz is not None, "created_at must be tz-aware (UTC)"


def test_refund_amount_no_nulls(tx):
    assert tx["refund_amount"].notna().all(), "refund_amount should be 0, not NaN, when absent"


def test_refund_amount_non_negative(tx):
    assert (tx["refund_amount"] >= 0).all()
