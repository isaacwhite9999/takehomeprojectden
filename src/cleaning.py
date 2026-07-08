"""Data cleaning and merging for the three input CSVs (sales, inventory, catalog)."""

import pandas as pd
from fastapi import HTTPException


def _normalize_product_id(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows without a product_id and normalize the ID to a trimmed string."""
    df = df.dropna(subset=["product_id"]).copy()
    df["product_id"] = df["product_id"].astype(str).str.strip()
    return df


def _fill_numeric_with_zero(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Coerce columns to numeric, replacing unparseable or missing values with 0."""
    for column in columns:
        df[column] = pd.to_numeric(df[column], errors="coerce").fillna(0)
    return df


def clean_sales(sales: pd.DataFrame) -> pd.DataFrame:
    """Deduplicate, normalize IDs, fill missing numerics, and parse dates."""
    sales = sales.drop_duplicates()
    sales = _normalize_product_id(sales)
    sales = _fill_numeric_with_zero(sales, ["units_sold", "revenue"])

    sales["date"] = pd.to_datetime(sales["date"], errors="coerce")
    sales = sales.dropna(subset=["date"])
    if sales.empty:
        raise HTTPException(
            status_code=400,
            detail="Sales file has no rows with a valid 'date' after cleaning.",
        )

    sales["store_id"] = sales["store_id"].astype(str).str.strip()
    return sales


def clean_inventory(inventory: pd.DataFrame) -> pd.DataFrame:
    """Deduplicate and normalize inventory to one row per product."""
    inventory = inventory.drop_duplicates()
    inventory = _normalize_product_id(inventory)
    inventory = _fill_numeric_with_zero(
        inventory, ["current_stock", "reorder_level", "lead_time_days"]
    )
    # Keep the most recent row if a product appears more than once.
    return inventory.drop_duplicates(subset=["product_id"], keep="last")


def clean_catalog(catalog: pd.DataFrame) -> pd.DataFrame:
    """Deduplicate, normalize IDs and prices, and standardize name/category text."""
    catalog = catalog.drop_duplicates()
    catalog = _normalize_product_id(catalog)
    catalog = _fill_numeric_with_zero(catalog, ["price"])

    catalog["product_name"] = catalog["product_name"].fillna("Unknown").astype(str).str.strip()
    catalog["category"] = (
        catalog["category"].fillna("uncategorized").astype(str).str.strip().str.lower()
    )
    return catalog.drop_duplicates(subset=["product_id"], keep="last")


def merge_data(
    sales: pd.DataFrame, inventory: pd.DataFrame, catalog: pd.DataFrame
) -> pd.DataFrame:
    """Left-join cleaned sales with inventory and catalog on product_id."""
    merged = sales.merge(inventory, on="product_id", how="left")
    merged = merged.merge(catalog, on="product_id", how="left")

    # Products missing from inventory/catalog still flow through with safe defaults.
    merged = _fill_numeric_with_zero(
        merged, ["current_stock", "reorder_level", "lead_time_days", "price"]
    )
    merged["product_name"] = merged["product_name"].fillna("Unknown")
    merged["category"] = merged["category"].fillna("uncategorized")

    if merged.empty:
        raise HTTPException(
            status_code=400,
            detail="Merged dataset is empty. Check that product_id values match across files.",
        )
    return merged


def clean_and_merge(
    sales: pd.DataFrame, inventory: pd.DataFrame, catalog: pd.DataFrame
) -> pd.DataFrame:
    """Full cleaning pipeline: clean each file, then merge on product_id."""
    return merge_data(clean_sales(sales), clean_inventory(inventory), clean_catalog(catalog))
