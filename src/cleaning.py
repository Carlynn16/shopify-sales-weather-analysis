import pandas as pd

_CAPPUCCINO_LONG = "Cappuccino   -   latte   -   cortado   -   flat white"
_CAPPUCCINO_CLEAN = "Cappuccino/Latte/Cortado/Flat White"

_HIGH_MISSING_COLS = ["sku", "customer_id", "email", "product_type"]
_STATUS_COLS = ["financial_status", "fulfillment_status", "cancelled_at", "cancel_reason"]


def clean_transactions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply all cleaning rules from the brief (section 3) to the raw transaction DataFrame.

    Steps (order matters):
      1. Drop >95%-missing columns.
      2. Normalise the long Cappuccino label in name and title.
      3. Fill missing categoricals; remove Indpakning rows; uppercase tags.
      4. Keep only completed sales (paid + fulfilled + not cancelled + real store).
      5. Drop now-redundant status columns.
      6. Add revenue = price * quantity.
    """
    df = df.copy()

    # 1. Drop high-missingness columns (safe even if a column was already absent)
    df = df.drop(columns=[c for c in _HIGH_MISSING_COLS if c in df.columns])

    # 2. Normalise the verbose Cappuccino POS label in both identity columns
    for col in ("name", "title"):
        if col in df.columns:
            mask = df[col] == _CAPPUCCINO_LONG
            df.loc[mask, col] = _CAPPUCCINO_CLEAN

    # 3. Fill missing categoricals; handle vendor alias; clean tags
    df["store_name"] = df["store_name"].fillna("Unknown Store")
    df["title"] = df["title"].fillna("Unknown Product")
    # fillna covers NaN; replace covers any pre-existing 'vendor-unknown' strings
    df["vendor"] = (
        df["vendor"]
        .fillna("Unknown Vendor")
        .replace({"vendor-unknown": "Unknown Vendor"})
    )
    df["tags"] = df["tags"].fillna("Unknown Tag")
    # Remove Indpakning rows *before* uppercasing (original order)
    df = df[df["tags"] != "Indpakning"]
    df["tags"] = df["tags"].str.upper()

    # 4. Keep only completed, paid, non-cancelled sales at real stores
    mask = (
        (df["financial_status"] == "paid")
        & (df["fulfillment_status"] == "fulfilled")
        & df["cancelled_at"].isna()
        & (df["store_name"] != "Malte TEST")
    )
    df = df[mask]

    # 5. Drop status columns that were only needed for the filter above
    df = df.drop(columns=[c for c in _STATUS_COLS if c in df.columns])

    # 6. Add revenue
    df["revenue"] = df["price"] * df["quantity"]

    return df.reset_index(drop=True)
