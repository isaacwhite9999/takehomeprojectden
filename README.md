# AI Retail Sales Copilot API

A backend-only FastAPI service that turns three retail CSVs (sales, inventory, product catalog) into actionable insights:

- Predicts next week's per-product sales with a scikit-learn model
- Flags products at risk of stocking out before replenishment arrives
- Identifies slow-moving inventory tying up capital
- Generates an LLM-written executive summary for store leadership

## What this project demonstrates

- **Data cleaning with Pandas** — deduplication, missing-value handling, type normalization, and multi-file merging with validation and readable API errors for bad uploads
- **Machine learning with scikit-learn** — a weekly sales forecasting model with a properly framed prediction target (this week's features → next week's units) and honest evaluation metrics (MAE, R²)
- **LLM integration** — a focused prompt built from the pipeline's outputs, with a graceful offline fallback when no API key is present
- **API design with FastAPI** — typed Pydantic response models, multipart file uploads, auto-generated Swagger docs, and clear error handling
- **Pragmatic scoping** — no database, no auth, no Docker; just a clean, readable pipeline that does one thing well

## Project structure

```text
retail-sales-copilot-api/
├── main.py              # FastAPI app, endpoints, pipeline orchestration
├── requirements.txt
├── .env.example
├── data/                # Generated sample CSVs (30 products, 2 stores, 90 days)
│   ├── sample_sales.csv
│   ├── sample_inventory.csv
│   └── sample_catalog.csv
└── src/
    ├── cleaning.py      # Clean and merge the three input files
    ├── features.py      # Date features, per-product stats, weekly training data
    ├── model.py         # Train RandomForest, predict next week, report metrics
    ├── analysis.py      # Stockout risk levels and slow-mover detection
    ├── llm_summary.py   # OpenAI executive summary with mock fallback
    ├── schemas.py       # Pydantic response models
    └── utils.py         # CSV upload parsing and validation
```

## Tech stack

- **FastAPI** + **Uvicorn** — API server
- **Pandas** / **NumPy** — data cleaning and feature engineering
- **scikit-learn** — RandomForestRegressor for weekly sales forecasting
- **OpenAI API** — executive summary (optional; mock fallback if no key)
- **python-dotenv** — environment config

## API endpoints

| Method | Path | Description |
|---|---|---|
| GET | `/` | Project name and available endpoints |
| GET | `/health` | Health check |
| POST | `/analyze` | Upload `sales_file`, `inventory_file`, `catalog_file` CSVs and run the full pipeline |
| GET | `/sample-analysis` | Run the full pipeline on the bundled sample data |
| GET | `/docs` | Interactive Swagger docs |

### Response shape (`/analyze` and `/sample-analysis`)

```json
{
  "model_metrics": { "mae": 4.2, "r2": 0.87, "training_rows": 350, "model_name": "RandomForestRegressor" },
  "top_predictions": [ { "product_id": "P001", "product_name": "...", "predicted_next_week_units": 42.0, "...": "..." } ],
  "stockout_risks": [ { "product_id": "P002", "risk_level": "high", "estimated_days_until_stockout": 3.5, "...": "..." } ],
  "slow_movers": [ { "product_id": "P003", "current_stock": 120, "avg_daily_sales": 0.4, "...": "..." } ],
  "executive_summary": "..."
}
```

## CSV schemas

**sales CSV**

| column | type |
|---|---|
| date | date (YYYY-MM-DD) |
| product_id | string |
| units_sold | number |
| revenue | number |
| store_id | string |

**inventory CSV**

| column | type |
|---|---|
| product_id | string |
| current_stock | number |
| reorder_level | number |
| lead_time_days | number |

**catalog CSV**

| column | type |
|---|---|
| product_id | string |
| product_name | string |
| category | string |
| price | number |

## How to run locally

```bash
git clone <repo-url>
cd retail-sales-copilot-api

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Optional: enable real LLM summaries
cp .env.example .env   # then add your OPENAI_API_KEY

uvicorn main:app --reload
```

The API is now at `http://127.0.0.1:8000` (interactive docs at `/docs`).

## Example curl requests

```bash
# Health check
curl http://127.0.0.1:8000/health

# Run the pipeline on the bundled sample data
curl http://127.0.0.1:8000/sample-analysis

# Upload your own CSVs
curl -X POST http://127.0.0.1:8000/analyze \
  -F "sales_file=@data/sample_sales.csv" \
  -F "inventory_file=@data/sample_inventory.csv" \
  -F "catalog_file=@data/sample_catalog.csv"
```

## ML approach

1. **Cleaning** — drop duplicates and rows without a `product_id`, fill missing numerics with 0, parse dates, normalize IDs/text, then merge the three files on `product_id`. Bad uploads return readable 400 errors.
2. **Features** — date features (`day_of_week`, `week_number`, `month`) plus per-product aggregates: `avg_daily_sales`, `total_units_sold`, `total_revenue`, `current_stock`, `reorder_level`, `lead_time_days`, and `price`.
3. **Weekly aggregation** — daily sales are rolled up into weekly per-product totals. Each row's target is the *following* week's units sold (`next_week_units_sold`).
4. **Model** — a `RandomForestRegressor` (200 trees) trained on an 80/20 split, reporting **MAE** and **R²**. The most recent week's features are used to predict the upcoming week.

**Stockout risk** — `estimated_days_until_stockout = current_stock / avg_daily_sales`, then bucketed: **high** if within `lead_time_days`, **medium** if within `lead_time_days + 7`, **low** otherwise.

**Slow movers** — products where `current_stock > reorder_level`, average daily sales are at or below the median, and sales in the last 14 days are weak.

## LLM approach

`src/llm_summary.py` builds a prompt containing the top 5 predicted best-sellers, top 5 stockout risks, top 5 slow movers, and total projected units/revenue, and asks the model to respond as a retail operations analyst (reorder recommendations, discount recommendations, risks, next actions).

- If `OPENAI_API_KEY` is set, it calls the OpenAI Chat Completions API (`gpt-4o-mini` by default, configurable via `OPENAI_MODEL`).
- If not, it returns a deterministic mock summary so the API works fully offline.

## Sample data

`data/` contains generated sample CSVs: 30 products across 3 categories (electronics, home goods, apparel), 2 stores, 90 days of daily sales with weekend boosts and per-product trends, plus realistic inventory levels, reorder points, lead times, and prices. A few duplicates and missing values are included so the cleaning step has something to do.

## Design decisions

- **Weekly aggregation over daily forecasting** — daily retail sales are noisy (many zero-sale days); weekly totals give the model a learnable signal while still answering the business question "what sells next week?"
- **RandomForest over deep learning** — with ~13 weeks of history per product, a tree ensemble on engineered features is more robust than anything sequence-based, and trains in under a second per request
- **Stateless pipeline, no persistence** — each request trains a fresh model on the uploaded data. That keeps the API simple and correct for arbitrary uploads; caching would be the first optimization if this served real traffic
- **Mock LLM fallback** — the API is fully functional and demoable without any API key or network access; the mock is clearly labeled so it can't be mistaken for real model output

## Future improvements

- Per-store forecasting instead of aggregating across stores
- Time-series-aware validation (rolling-origin backtesting instead of a random split)
- Seasonality and holiday features
- Confidence intervals on predictions
- Caching/persisting trained models between requests
- Reorder quantity suggestions (economic order quantity)
