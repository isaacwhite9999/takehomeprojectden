"""Inventory business logic: stockout risk levels and slow-moving products."""

import numpy as np
import pandas as pd

# A product is "medium" risk if it stocks out within this many days past its lead time.
MEDIUM_RISK_BUFFER_DAYS = 7

# Window used to judge whether a product's recent sales are weak.
RECENT_SALES_WINDOW_DAYS = 14


def find_stockout_risks(product_stats: pd.DataFrame) -> pd.DataFrame:
    """Flag products likely to run out of stock before replenishment arrives.

    Risk levels, based on estimated days of stock remaining vs. supplier lead time:
        high   -> stocks out within the lead time (reorder now is already late)
        medium -> stocks out within lead time + 7 days (reorder now)
        low    -> comfortable runway (excluded from results)
    """
    df = product_stats.copy()

    df["estimated_days_until_stockout"] = np.where(
        df["avg_daily_sales"] > 0,
        df["current_stock"] / df["avg_daily_sales"],
        np.inf,
    )

    def classify_risk(row: pd.Series) -> str:
        days_remaining = row["estimated_days_until_stockout"]
        if days_remaining <= row["lead_time_days"]:
            return "high"
        if days_remaining <= row["lead_time_days"] + MEDIUM_RISK_BUFFER_DAYS:
            return "medium"
        return "low"

    df["risk_level"] = df.apply(classify_risk, axis=1)

    risks = df[df["risk_level"].isin(["high", "medium"])].copy()
    risks = risks.sort_values("estimated_days_until_stockout")
    risks["estimated_days_until_stockout"] = (
        risks["estimated_days_until_stockout"].replace(np.inf, -1).round(1)
    )
    return risks


def find_slow_movers(product_stats: pd.DataFrame, merged: pd.DataFrame) -> pd.DataFrame:
    """Flag overstocked products with weak overall and recent sales.

    A slow mover satisfies all three conditions:
        1. current_stock exceeds its reorder level (capital tied up in inventory)
        2. average daily sales at or below the median across products
        3. sales over the last 14 days at or below the median across products
    """
    df = product_stats.copy()

    recent_cutoff = merged["date"].max() - pd.Timedelta(days=RECENT_SALES_WINDOW_DAYS)
    recent_units = (
        merged[merged["date"] > recent_cutoff]
        .groupby("product_id")["units_sold"]
        .sum()
        .rename("recent_14d_units")
    )
    df = df.merge(recent_units, on="product_id", how="left")
    df["recent_14d_units"] = df["recent_14d_units"].fillna(0)

    low_avg_sales = df["avg_daily_sales"].median()
    low_recent_sales = df["recent_14d_units"].median()

    slow = df[
        (df["current_stock"] > df["reorder_level"])
        & (df["avg_daily_sales"] <= low_avg_sales)
        & (df["recent_14d_units"] <= low_recent_sales)
    ].copy()

    slow["avg_daily_sales"] = slow["avg_daily_sales"].round(2)
    return slow.sort_values("avg_daily_sales")
