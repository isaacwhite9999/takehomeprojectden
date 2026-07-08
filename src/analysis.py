"""Inventory analysis: stockout risks and slow-moving products."""

import numpy as np
import pandas as pd


def find_stockout_risks(product_stats: pd.DataFrame) -> pd.DataFrame:
    """Estimate days until stockout and bucket products into risk levels."""
    df = product_stats.copy()

    df["estimated_days_until_stockout"] = np.where(
        df["avg_daily_sales"] > 0,
        df["current_stock"] / df["avg_daily_sales"],
        np.inf,
    )

    def risk_level(row) -> str:
        days = row["estimated_days_until_stockout"]
        if days <= row["lead_time_days"]:
            return "high"
        if days <= row["lead_time_days"] + 7:
            return "medium"
        return "low"

    df["risk_level"] = df.apply(risk_level, axis=1)

    risks = df[df["risk_level"].isin(["high", "medium"])].copy()
    risks = risks.sort_values("estimated_days_until_stockout")
    risks["estimated_days_until_stockout"] = (
        risks["estimated_days_until_stockout"].replace(np.inf, -1).round(1)
    )
    return risks


def find_slow_movers(product_stats: pd.DataFrame, merged: pd.DataFrame) -> pd.DataFrame:
    """Products with excess stock, low average sales, and weak recent sales."""
    df = product_stats.copy()

    # Weak recent sales: units sold in the most recent 14 days, per product.
    recent_cutoff = merged["date"].max() - pd.Timedelta(days=14)
    recent = (
        merged[merged["date"] > recent_cutoff]
        .groupby("product_id")["units_sold"]
        .sum()
        .rename("recent_14d_units")
    )
    df = df.merge(recent, on="product_id", how="left")
    df["recent_14d_units"] = df["recent_14d_units"].fillna(0)

    low_sales_threshold = df["avg_daily_sales"].median()
    recent_threshold = df["recent_14d_units"].median()

    slow = df[
        (df["current_stock"] > df["reorder_level"])
        & (df["avg_daily_sales"] <= low_sales_threshold)
        & (df["recent_14d_units"] <= recent_threshold)
    ].copy()

    slow["avg_daily_sales"] = slow["avg_daily_sales"].round(2)
    return slow.sort_values("avg_daily_sales")
