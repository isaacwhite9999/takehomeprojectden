"""AI Retail Sales Copilot API.

Upload sales, inventory, and catalog CSVs; receive next week's sales forecast,
stockout risks, slow-moving inventory, and an LLM-generated executive summary.

Run locally:
    uvicorn main:app --reload
"""

from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile

from src.analysis import find_slow_movers, find_stockout_risks
from src.cleaning import clean_and_merge
from src.features import build_product_stats, build_weekly_training_data
from src.llm_summary import generate_summary
from src.model import train_and_predict
from src.schemas import AnalysisResponse
from src.utils import CATALOG_COLUMNS, INVENTORY_COLUMNS, SALES_COLUMNS, read_csv_upload

load_dotenv()

DATA_DIR = Path(__file__).parent / "data"
TOP_N_RESULTS = 10

PREDICTION_COLUMNS = [
    "product_id",
    "product_name",
    "category",
    "predicted_next_week_units",
    "predicted_next_week_revenue",
]
STOCKOUT_COLUMNS = [
    "product_id",
    "product_name",
    "category",
    "current_stock",
    "avg_daily_sales",
    "lead_time_days",
    "estimated_days_until_stockout",
    "risk_level",
]
SLOW_MOVER_COLUMNS = [
    "product_id",
    "product_name",
    "category",
    "current_stock",
    "reorder_level",
    "avg_daily_sales",
    "recent_14d_units",
]

app = FastAPI(
    title="AI Retail Sales Copilot API",
    description=(
        "Forecast next week's sales, flag stockout risks and slow movers, "
        "and generate an executive summary from three retail CSVs."
    ),
    version="1.0.0",
)


def run_pipeline(
    sales: pd.DataFrame, inventory: pd.DataFrame, catalog: pd.DataFrame
) -> AnalysisResponse:
    """Execute the full pipeline: clean -> features -> model -> analysis -> LLM summary."""
    merged = clean_and_merge(sales, inventory, catalog)
    product_stats = build_product_stats(merged)

    training, latest_week = build_weekly_training_data(merged, product_stats)
    metrics, predictions = train_and_predict(training, latest_week)

    stockout_risks = find_stockout_risks(product_stats)
    slow_movers = find_slow_movers(product_stats, merged)

    top_predictions = (
        predictions[PREDICTION_COLUMNS].head(TOP_N_RESULTS).to_dict(orient="records")
    )
    risks = (
        stockout_risks[STOCKOUT_COLUMNS].round(2).head(TOP_N_RESULTS).to_dict(orient="records")
    )
    slow = (
        slow_movers[SLOW_MOVER_COLUMNS].round(2).head(TOP_N_RESULTS).to_dict(orient="records")
    )

    total_units = float(predictions["predicted_next_week_units"].sum())
    total_revenue = float(predictions["predicted_next_week_revenue"].sum())

    executive_summary = generate_summary(top_predictions, risks, slow, total_units, total_revenue)

    return AnalysisResponse(
        model_metrics=metrics,
        top_predictions=top_predictions,
        stockout_risks=risks,
        slow_movers=slow,
        executive_summary=executive_summary,
    )


@app.get("/")
def root() -> dict:
    """Project overview and available endpoints."""
    return {
        "project": "AI Retail Sales Copilot API",
        "endpoints": {
            "GET /": "This overview",
            "GET /health": "Health check",
            "POST /analyze": "Upload sales_file, inventory_file, and catalog_file CSVs",
            "GET /sample-analysis": "Run the pipeline on the bundled sample data",
            "GET /docs": "Interactive API documentation",
        },
    }


@app.get("/health")
def health() -> dict:
    """Health check."""
    return {"status": "ok"}


@app.post("/analyze", response_model=AnalysisResponse)
async def analyze(
    sales_file: UploadFile = File(..., description="Sales CSV"),
    inventory_file: UploadFile = File(..., description="Inventory CSV"),
    catalog_file: UploadFile = File(..., description="Product catalog CSV"),
) -> AnalysisResponse:
    """Run the full analysis pipeline on three uploaded CSV files."""
    sales = read_csv_upload(await sales_file.read(), sales_file.filename, SALES_COLUMNS)
    inventory = read_csv_upload(
        await inventory_file.read(), inventory_file.filename, INVENTORY_COLUMNS
    )
    catalog = read_csv_upload(await catalog_file.read(), catalog_file.filename, CATALOG_COLUMNS)
    return run_pipeline(sales, inventory, catalog)


@app.get("/sample-analysis", response_model=AnalysisResponse)
def sample_analysis() -> AnalysisResponse:
    """Run the full analysis pipeline on the bundled sample dataset."""
    try:
        sales = pd.read_csv(DATA_DIR / "sample_sales.csv")
        inventory = pd.read_csv(DATA_DIR / "sample_inventory.csv")
        catalog = pd.read_csv(DATA_DIR / "sample_catalog.csv")
    except FileNotFoundError:
        raise HTTPException(status_code=500, detail="Sample data files not found in data/.")
    return run_pipeline(sales, inventory, catalog)
