from pathlib import Path

import pandas as pd

from src.config import DATA_DIR

TABLE_NAMES = [
    "orders",
    "line_items",
    "products",
    "locations",
    "customers",
    "discounts",
    "refunds",
]


def load_raw_tables(data_dir=DATA_DIR) -> dict[str, pd.DataFrame]:
    """Load all 7 raw Shopify CSV tables into a dict keyed by table name."""
    data_dir = Path(data_dir)
    return {name: pd.read_csv(data_dir / f"{name}.csv") for name in TABLE_NAMES}


def build_transactions(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Merge the raw tables into a single transaction-level DataFrame.

    Join order (left joins throughout):
        line_items
        <- orders         on order_id      (adds timestamps, financials, status flags)
        <- products       on product_id    (adds title, product_type, vendor, tags)
        <- locations      on location_id   (adds store_name, city)
        <- refunds        on order_id      (adds refund_amount per order, filled 0)

    No filtering or cleaning is applied here.
    """
    line_items = tables["line_items"].copy()
    orders = tables["orders"].copy()
    products = tables["products"].copy()
    locations = tables["locations"].copy()
    refunds = tables["refunds"].copy()

    # line_items.created_at is the API-record timestamp, less useful than the
    # order-level created_at; drop it to avoid a suffix collision on merge.
    line_items = line_items.drop(columns=["created_at"])

    # --- line_items <- orders ---
    df = line_items.merge(orders, on="order_id", how="left")

    # --- <- products (dedup so each product_id maps to one metadata row) ---
    products_meta = (
        products
        .drop_duplicates(subset="product_id")[["product_id", "title", "product_type", "vendor", "tags"]]
    )
    df = df.merge(products_meta, on="product_id", how="left")

    # --- <- locations (rename id/name to match FK on df) ---
    locations_clean = (
        locations
        .rename(columns={"id": "location_id", "name": "store_name"})[["location_id", "store_name", "city"]]
    )
    df = df.merge(locations_clean, on="location_id", how="left")

    # --- <- refunds (aggregate to one row per order, fill missing with 0) ---
    refund_totals = (
        refunds
        .groupby("order_id", as_index=False)["amount"]
        .sum()
        .rename(columns={"amount": "refund_amount"})
    )
    df = df.merge(refund_totals, on="order_id", how="left")
    df["refund_amount"] = df["refund_amount"].fillna(0)

    # --- parse datetimes (UTC-aware) ---
    for col in ["created_at", "processed_at"]:
        df[col] = pd.to_datetime(df[col], utc=True, errors="coerce")

    return df
