"""Feature engineering: date features, per-product statistics, and weekly training data."""

import pandas as pd

# Feature set used by the forecasting model on weekly-aggregated data.
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

# Per-product attributes joined onto each weekly row.
PRODUCT_ATTRIBUTES = [
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


def add_date_features(merged: pd.DataFrame) -> pd.DataFrame:
    """Add calendar features derived from the sale date."""
    merged = merged.copy()
    merged["day_of_week"] = merged["date"].dt.dayofweek
    merged["week_number"] = merged["date"].dt.isocalendar().week.astype(int)
    merged["month"] = merged["date"].dt.month
    return merged


def build_product_stats(merged: pd.DataFrame) -> pd.DataFrame:
    """Compute per-product aggregates used for both modeling and inventory analysis."""
    observed_days = max((merged["date"].max() - merged["date"].min()).days + 1, 1)

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
    stats["avg_daily_sales"] = stats["total_units_sold"] / observed_days
    return stats


def build_weekly_training_data(
    merged: pd.DataFrame, product_stats: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Aggregate daily sales into weekly totals and frame the forecasting problem.

    Each row pairs a product's features for one week with the target
    `next_week_units_sold` (that same product's sales the following week).

    Returns:
        training: rows with a known next-week target, used to fit the model.
        latest_week: the most recent week, whose target is unknown -- these
            rows are what the model scores to forecast the upcoming week.
    """
    merged = add_date_features(merged)

    weekly = (
        merged.groupby(["product_id", pd.Grouper(key="date", freq="W")])
        .agg(week_units_sold=("units_sold", "sum"), month=("month", "max"))
        .reset_index()
        .sort_values(["product_id", "date"])
    )

    weekly["next_week_units_sold"] = weekly.groupby("product_id")["week_units_sold"].shift(-1)

    weekly = weekly.merge(
        product_stats[["product_id"] + PRODUCT_ATTRIBUTES], on="product_id", how="left"
    )

    training = weekly.dropna(subset=["next_week_units_sold"]).copy()
    latest_week = weekly[weekly["date"] == weekly["date"].max()].copy()
    return training, latest_week
