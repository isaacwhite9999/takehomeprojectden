# AI Retail Sales Copilot API

A FastAPI backend that turns raw retail CSVs into decisions. Upload sales history, inventory levels, and a product catalog; the service cleans and merges the data, trains a sales forecasting model, flags inventory risks, and generates an executive summary with an LLM.

## Project Overview

Retail teams sit on three disconnected spreadsheets: what sold, what's in stock, and what things cost. This project connects them and answers the questions an operations manager actually asks:

- **What will sell next week?** — per-product forecasts from a scikit-learn model
- **What will run out?** — stockout risk levels based on sales velocity vs. supplier lead time
- **What isn't moving?** — overstocked products with weak demand, candidates for markdown
- **What should I do about it?** — a concise, LLM-written executive report

The scope is intentionally focused: no database, no auth, no Docker. One clean request/response pipeline that demonstrates the complete AI/ML workflow — data cleaning, feature engineering, model training and evaluation, business logic, and LLM integration — behind a well-structured API.

## Architecture

```text
Sales CSV
Inventory CSV
Catalog CSV
        │
        ▼
Data Cleaning (Pandas)
        │
        ▼
Feature Engineering
        │
        ▼
RandomForestRegressor
        │
        ├── Sales Forecast
        ├── Stockout Risk
        └── Slow Movers
        │
        ▼
LLM Executive Summary
        │
        ▼
JSON API Response
```

Each stage lives in its own module under `src/`, and `main.py` orchestrates them in `run_pipeline()`. The pipeline is stateless: every request trains on exactly the data it was given.

## ML Pipeline

**1. Data cleaning** (`src/cleaning.py`)

- Remove duplicate rows
- Drop rows missing `product_id`; normalize IDs to trimmed strings
- Coerce numeric columns, filling missing/unparseable values with 0
- Parse dates and drop rows where parsing fails
- Standardize product names and lowercase categories
- Left-join sales ← inventory ← catalog on `product_id`
- Invalid uploads return readable HTTP 400 errors (missing columns, empty files, unparseable CSVs)

**2. Feature engineering** (`src/features.py`)

- Calendar features: `day_of_week`, `week_number`, `month`
- Per-product statistics: `avg_daily_sales`, `total_units_sold`, `total_revenue`
- Inventory context: `current_stock`, `reorder_level`, `lead_time_days`, `price`
- Daily sales are aggregated into weekly per-product totals; each row's target is the **following** week's units sold, so the model learns "given this week, predict next week"

**3. Model training and evaluation** (`src/model.py`)

- `RandomForestRegressor` (200 trees, fixed random state)
- 80/20 `train_test_split`, evaluated on the held-out split
- Metrics returned with every response: **MAE**, **RMSE**, **R²**
- The most recent week's features are scored to produce next week's forecast

**4. Business logic** (`src/analysis.py`)

- **Stockout risk**: `estimated_days_until_stockout = current_stock / avg_daily_sales`, bucketed as **high** (stocks out within the supplier lead time), **medium** (within lead time + 7 days), or **low** (excluded from results)
- **Slow movers**: products where stock exceeds the reorder level *and* average daily sales *and* last-14-day sales are at or below the median across products

## LLM Pipeline

`src/llm_summary.py` injects the pipeline's structured outputs — top predicted best-sellers, stockout risks, slow movers, and projected totals — into a single prompt, and asks the model to write as a retail operations analyst. The summary covers key insights, reorder recommendations, discount recommendations, business risks, and next actions.

- With `OPENAI_API_KEY` set: calls the OpenAI Chat Completions API (`gpt-4o-mini` by default, configurable via `OPENAI_MODEL`)
- Without a key (or if the call fails): returns a clearly labeled deterministic mock built from the same numbers, so the API works fully offline

This design keeps the LLM grounded: it summarizes computed results rather than guessing from raw data.

## Folder Structure

```text
retail-sales-copilot-api/
├── main.py              # FastAPI app, endpoints, pipeline orchestration
├── requirements.txt
├── .env.example
├── data/                # Sample dataset: 30 products, 2 stores, 90 days
│   ├── sample_sales.csv
│   ├── sample_inventory.csv
│   └── sample_catalog.csv
└── src/
    ├── cleaning.py      # Clean and merge the three input files
    ├── features.py      # Date features, product stats, weekly training frame
    ├── model.py         # Train, evaluate (MAE/RMSE/R²), and forecast
    ├── analysis.py      # Stockout risk and slow-mover business logic
    ├── llm_summary.py   # LLM executive summary with offline mock fallback
    ├── schemas.py       # Pydantic response models
    └── utils.py         # CSV upload parsing and validation
```

## API Endpoints

| Method | Path | Description |
|---|---|---|
| GET | `/` | Project overview and endpoint list |
| GET | `/health` | Health check |
| POST | `/analyze` | Upload `sales_file`, `inventory_file`, `catalog_file` CSVs and run the pipeline |
| GET | `/sample-analysis` | Run the pipeline on the bundled sample data |
| GET | `/docs` | Interactive Swagger documentation |

## Technologies Used

- **Python 3.11+**
- **FastAPI** + **Uvicorn** — API framework and server
- **Pandas** / **NumPy** — data cleaning and feature engineering
- **scikit-learn** — forecasting model and evaluation metrics
- **OpenAI API** — executive summary generation
- **python-dotenv** — environment configuration

## Dataset Schema

**sales CSV**

| column | type | description |
|---|---|---|
| date | date (YYYY-MM-DD) | Day of sale |
| product_id | string | Product identifier |
| units_sold | number | Units sold that day |
| revenue | number | Revenue for the row |
| store_id | string | Store identifier |

**inventory CSV**

| column | type | description |
|---|---|---|
| product_id | string | Product identifier |
| current_stock | number | Units on hand |
| reorder_level | number | Stock level that triggers a reorder |
| lead_time_days | number | Days for replenishment to arrive |

**catalog CSV**

| column | type | description |
|---|---|---|
| product_id | string | Product identifier |
| product_name | string | Display name |
| category | string | Product category |
| price | number | Unit price |

The bundled sample data covers 30 products across 3 categories, 2 stores, and 90 days of daily sales — including a few duplicates and missing values for the cleaning step to handle.

## Getting Started

```bash
git clone https://github.com/isaacwhite9999/takehomeprojectden.git
cd takehomeprojectden

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Optional: enable real LLM summaries (mock fallback is used otherwise)
cp .env.example .env   # then add your OPENAI_API_KEY

uvicorn main:app --reload
```

The API runs at `http://127.0.0.1:8000` with interactive docs at `/docs`.

## Example Request/Response

```bash
# Quickest demo: run the pipeline on the bundled sample data
curl http://127.0.0.1:8000/sample-analysis

# Or upload your own CSVs
curl -X POST http://127.0.0.1:8000/analyze \
  -F "sales_file=@data/sample_sales.csv" \
  -F "inventory_file=@data/sample_inventory.csv" \
  -F "catalog_file=@data/sample_catalog.csv"
```

Abbreviated response:

```json
{
  "model_metrics": {
    "mae": 8.37,
    "rmse": 11.51,
    "r2": 0.954,
    "training_rows": 389,
    "model_name": "RandomForestRegressor"
  },
  "top_predictions": [
    {
      "product_id": "P018",
      "product_name": "Wall Clock",
      "category": "home goods",
      "predicted_next_week_units": 16.9,
      "predicted_next_week_revenue": 212.1
    }
  ],
  "stockout_risks": [
    {
      "product_id": "P005",
      "product_name": "USB-C Cable",
      "category": "electronics",
      "current_stock": 86.0,
      "avg_daily_sales": 22.27,
      "lead_time_days": 12.0,
      "estimated_days_until_stockout": 3.9,
      "risk_level": "high"
    }
  ],
  "slow_movers": [
    {
      "product_id": "P016",
      "product_name": "Cutting Board",
      "category": "home goods",
      "current_stock": 60.0,
      "reorder_level": 18.0,
      "avg_daily_sales": 2.3,
      "recent_14d_units": 27.0
    }
  ],
  "executive_summary": "Next week we project 356 units sold for roughly $15,396.58 in revenue, led by Wall Clock. Immediate reorders are recommended for stockout-risk items..."
}
```

## Design Decisions

- **Weekly aggregation over daily forecasting** — daily retail sales are noisy (many zero-sale days); weekly totals give the model a learnable signal while still answering "what sells next week?"
- **RandomForest over deep learning** — with ~13 weeks of history per product, a tree ensemble on engineered features is more robust than anything sequence-based, and trains in under a second per request
- **Stateless pipeline** — each request trains a fresh model on the uploaded data, keeping the API simple and correct for arbitrary uploads
- **Grounded LLM prompt with mock fallback** — the model summarizes computed results (not raw data), and the API is fully demoable without a key or network access

## Future Improvements

- Per-store forecasting instead of aggregating across stores
- Time-series-aware validation (rolling-origin backtesting instead of a random split)
- Seasonality and holiday features
- Confidence intervals on predictions
- Model caching between requests for repeat analyses
- Reorder quantity suggestions (economic order quantity)
