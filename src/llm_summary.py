"""LLM executive summary via the OpenAI API, with an offline mock fallback.

The pipeline's structured outputs (forecasts, risks, slow movers) are injected
directly into the prompt, so the model reasons over real numbers rather than
raw data. If no OPENAI_API_KEY is configured, a deterministic mock summary is
returned so the API remains fully functional offline.
"""

import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

DEFAULT_MODEL = "gpt-4o-mini"

SYSTEM_PROMPT = (
    "You are a retail operations analyst writing for store leadership. "
    "Produce a concise, business-focused executive summary covering: key insights, "
    "reorder recommendations, discount recommendations for slow-moving stock, "
    "business risks, and next actions. Use short paragraphs or bullet points. "
    "Keep it under 250 words."
)


def _format_product_lines(rows: list[dict], value_key: str, value_label: str) -> str:
    """Render products as prompt-friendly bullet lines."""
    if not rows:
        return "- none"
    return "\n".join(
        f"- {row['product_name']} ({row['category']}): {row[value_key]} {value_label}"
        for row in rows
    )


def build_prompt(
    top_predictions: list[dict],
    stockout_risks: list[dict],
    slow_movers: list[dict],
    total_projected_units: float,
    total_projected_revenue: float,
) -> str:
    """Assemble the analysis results into a single prompt for the LLM."""
    return f"""Here is next week's retail forecast and inventory analysis.

Top predicted best-sellers next week:
{_format_product_lines(top_predictions[:5], "predicted_next_week_units", "units")}

Stockout-risk products:
{_format_product_lines(stockout_risks[:5], "estimated_days_until_stockout", "days of stock left")}

Slow-moving products:
{_format_product_lines(slow_movers[:5], "avg_daily_sales", "avg units/day")}

Total projected units sold next week: {total_projected_units:,.0f}
Total projected revenue next week: ${total_projected_revenue:,.2f}

Write the executive summary."""


def _mock_summary(
    top_predictions: list[dict],
    stockout_risks: list[dict],
    slow_movers: list[dict],
    total_projected_units: float,
    total_projected_revenue: float,
) -> str:
    """Deterministic fallback used when the OpenAI API is unavailable."""
    top_seller = top_predictions[0]["product_name"] if top_predictions else "n/a"
    risk_names = ", ".join(r["product_name"] for r in stockout_risks[:3]) or "none identified"
    slow_names = ", ".join(s["product_name"] for s in slow_movers[:3]) or "none identified"

    return (
        "[Mock summary - set OPENAI_API_KEY for an AI-generated version]\n\n"
        f"Next week we project {total_projected_units:,.0f} units sold for roughly "
        f"${total_projected_revenue:,.2f} in revenue, led by {top_seller}. "
        f"Immediate reorders are recommended for stockout-risk items: {risk_names}. "
        f"Slow movers ({slow_names}) are candidates for markdowns or promotions to free up "
        "working capital. Key risk: high-velocity items may stock out before replenishment "
        "arrives given current lead times. Next actions: place priority purchase orders for "
        "high-risk SKUs, launch a targeted discount on slow-moving stock, and review reorder "
        "levels against the updated demand forecast."
    )


def generate_summary(
    top_predictions: list[dict],
    stockout_risks: list[dict],
    slow_movers: list[dict],
    total_projected_units: float,
    total_projected_revenue: float,
) -> str:
    """Generate an executive summary, falling back to a mock if no API key is set."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return _mock_summary(
            top_predictions,
            stockout_risks,
            slow_movers,
            total_projected_units,
            total_projected_revenue,
        )

    prompt = build_prompt(
        top_predictions,
        stockout_risks,
        slow_movers,
        total_projected_units,
        total_projected_revenue,
    )

    try:
        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model=os.getenv("OPENAI_MODEL", DEFAULT_MODEL),
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=0.4,
        )
        return response.choices[0].message.content.strip()
    except Exception as exc:  # Keep the API usable even if the LLM call fails.
        return f"[LLM call failed: {exc}]\n\n" + _mock_summary(
            top_predictions,
            stockout_risks,
            slow_movers,
            total_projected_units,
            total_projected_revenue,
        )
