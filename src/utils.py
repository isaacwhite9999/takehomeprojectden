"""Shared helpers for loading and validating uploaded CSVs."""

import io

import pandas as pd
from fastapi import HTTPException

SALES_COLUMNS = ["date", "product_id", "units_sold", "revenue", "store_id"]
INVENTORY_COLUMNS = ["product_id", "current_stock", "reorder_level", "lead_time_days"]
CATALOG_COLUMNS = ["product_id", "product_name", "category", "price"]


def read_csv_upload(file_bytes: bytes, filename: str, required_columns: list[str]) -> pd.DataFrame:
    """Parse uploaded bytes into a DataFrame and validate required columns."""
    try:
        df = pd.read_csv(io.BytesIO(file_bytes))
    except Exception:
        raise HTTPException(
            status_code=400,
            detail=f"Could not parse '{filename}' as CSV. Please upload a valid CSV file.",
        )

    if df.empty:
        raise HTTPException(status_code=400, detail=f"'{filename}' contains no rows.")

    df.columns = [str(c).strip().lower() for c in df.columns]
    missing = [c for c in required_columns if c not in df.columns]
    if missing:
        raise HTTPException(
            status_code=400,
            detail=f"'{filename}' is missing required columns: {', '.join(missing)}.",
        )
    return df
