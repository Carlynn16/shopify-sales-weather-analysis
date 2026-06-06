from pathlib import Path

import pandas as pd

from src.config import ROOT_DIR

CATEGORIES_CSV = ROOT_DIR / "src" / "product_categories.csv"


def categorize(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add a name_category column by exact-name lookup in product_categories.csv.

    The CSV is the single source of truth for all 8 product families:
      Ice Cream, Hot Beverages, Buns & Bakery, Chocolate, Gifts & Cards,
      Snacks & Nuts, Christmas, Others.

    Client feedback is already baked into the CSV: items whose names contain
    'flødebolle' or 'bar' (that were originally Buns & Bakery) are mapped to
    Chocolate. Names absent from the CSV are labeled 'Uncategorized'.
    """
    df = df.copy()

    mapping = (
        pd.read_csv(CATEGORIES_CSV, encoding="utf-8")
        .set_index("name")["category"]
        .to_dict()
    )

    df["name_category"] = df["name"].map(mapping).fillna("Uncategorized")
    return df
