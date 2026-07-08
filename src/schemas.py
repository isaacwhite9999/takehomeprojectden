"""Pydantic response models for the API."""

from pydantic import BaseModel


class ModelMetrics(BaseModel):
    mae: float
    r2: float
    training_rows: int
    model_name: str


class Prediction(BaseModel):
    product_id: str
    product_name: str
    category: str
    predicted_next_week_units: float
    predicted_next_week_revenue: float


class StockoutRisk(BaseModel):
    product_id: str
    product_name: str
    category: str
    current_stock: float
    avg_daily_sales: float
    lead_time_days: float
    estimated_days_until_stockout: float
    risk_level: str


class SlowMover(BaseModel):
    product_id: str
    product_name: str
    category: str
    current_stock: float
    reorder_level: float
    avg_daily_sales: float
    recent_14d_units: float


class AnalysisResponse(BaseModel):
    model_metrics: ModelMetrics
    top_predictions: list[Prediction]
    stockout_risks: list[StockoutRisk]
    slow_movers: list[SlowMover]
    executive_summary: str
