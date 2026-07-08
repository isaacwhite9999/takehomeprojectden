"""Pydantic response models for the API."""

from pydantic import BaseModel


class ModelMetrics(BaseModel):
    """Evaluation metrics from the chronologically held-out test split."""

    mae: float
    rmse: float
    r2: float
    training_rows: int
    test_rows: int
    model_name: str


class Prediction(BaseModel):
    """Forecasted sales for one product for the upcoming week."""

    product_id: str
    product_name: str
    category: str
    predicted_next_week_units: float
    predicted_next_week_revenue: float


class StockoutRisk(BaseModel):
    """A product whose stock may run out before replenishment arrives."""

    product_id: str
    product_name: str
    category: str
    current_stock: float
    avg_daily_sales: float
    lead_time_days: float
    estimated_days_until_stockout: float
    risk_level: str


class SlowMover(BaseModel):
    """Overstocked product with weak overall and recent sales."""

    product_id: str
    product_name: str
    category: str
    current_stock: float
    reorder_level: float
    avg_daily_sales: float
    recent_14d_units: float


class AnalysisResponse(BaseModel):
    """Full pipeline output returned by /analyze and /sample-analysis."""

    model_metrics: ModelMetrics
    top_predictions: list[Prediction]
    stockout_risks: list[StockoutRisk]
    slow_movers: list[SlowMover]
    executive_summary: str
