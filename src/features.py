"""Feature engineering: date features, per-product stats, and weekly training data."""

import pandas as pd

# Features used by the ML model on the weekly-aggregated data.
MODEL_FEATURES = [
    "week_units_sold",
    "avg_daily_sales",
    "total_units_sold",
    "total_revenue",
    "current_stock",
    "reorder_level",
    "lead_time_days",
    "price",
    "month",
]


def add_date_features(merged: pd.DataFrame) -> pd.DataFrame:
    merged = merged.copy()
    merged["day_of_week"] = merged["date"].dt.dayofweek
    merged["week_number"] = merged["date"].dt.isocalendar().week.astype(int)
    merged["month"] = merged["date"].dt.month
    return merged


def build_product_stats(merged: pd.DataFrame) -> pd.DataFrame:
    """Per-product aggregates used for both modeling and inventory analysis."""
    n_days = max((merged["date"].max() - merged["date"].min()).days + 1, 1)

    stats = (
        merged.groupby("product_id")
        .agg(
            total_units_sold=("units_sold", "sum"),
            total_revenue=("revenue", "sum"),
            current_stock=("current_stock", "first"),
            reorder_level=("reorder_level", "first"),
            lead_time_days=("lead_time_days", "first"),
            price=("price", "first"),
            product_name=("product_name", "first"),
            category=("category", "first"),
        )
        .reset_index()
    )
    stats["avg_daily_sales"] = stats["total_units_sold"] / n_days
    return stats


def build_weekly_training_data(merged: pd.DataFrame, product_stats: pd.DataFrame):
    """Aggregate daily sales into weekly product totals, then create
    (this week's features -> next week's units sold) training pairs.

    Returns (training_df, latest_week_df). The latest week has no known
    target and is used to predict the upcoming week.
    """
    merged = add_date_features(merged)

    weekly = (
        merged.groupby(["product_id", pd.Grouper(key="date", freq="W")])
        .agg(week_units_sold=("units_sold", "sum"), month=("month", "max"))
        .reset_index()
        .sort_values(["product_id", "date"])
    )

    # Target: the following week's units sold for the same product.
    weekly["next_week_units_sold"] = weekly.groupby("product_id")["week_units_sold"].shift(-1)

    feature_cols = [
        "avg_daily_sales",
        "total_units_sold",
        "total_revenue",
        "current_stock",
        "reorder_level",
        "lead_time_days",
        "price",
        "product_name",
        "category",
    ]
    weekly = weekly.merge(product_stats[["product_id"] + feature_cols], on="product_id", how="left")

    training = weekly.dropna(subset=["next_week_units_sold"]).copy()
    latest_week = weekly[weekly["date"] == weekly["date"].max()].copy()
    return training, latest_week
