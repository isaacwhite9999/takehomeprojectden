"""Executive summary generation via the OpenAI API, with an offline mock fallback."""

import os

from dotenv import load_dotenv

load_dotenv()

SYSTEM_PROMPT = (
    "You are a retail operations analyst. Write a concise, business-focused "
    "executive summary for store leadership. Include reorder recommendations, "
    "discount recommendations for slow-moving stock, key risks, and next actions. "
    "Use short paragraphs or bullet points. Keep it under 250 words."
)


def _format_products(rows: list[dict], value_key: str, value_label: str) -> str:
    if not rows:
        return "- none"
    return "\n".join(
        f"- {r['product_name']} ({r['category']}): {r[value_key]} {value_label}" for r in rows
    )


def build_prompt(
    top_predictions: list[dict],
    stockout_risks: list[dict],
    slow_movers: list[dict],
    total_projected_units: float,
    total_projected_revenue: float,
) -> str:
    return f"""Here is next week's retail forecast and inventory analysis.

Top 5 predicted best-sellers next week:
{_format_products(top_predictions[:5], "predicted_next_week_units", "units")}

Top 5 stockout-risk products (days of stock left):
{_format_products(stockout_risks[:5], "estimated_days_until_stockout", "days until stockout")}

Top 5 slow-moving products (avg daily sales):
{_format_products(slow_movers[:5], "avg_daily_sales", "avg units/day")}

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
    top_name = top_predictions[0]["product_name"] if top_predictions else "n/a"
    risk_names = ", ".join(r["product_name"] for r in stockout_risks[:3]) or "none identified"
    slow_names = ", ".join(s["product_name"] for s in slow_movers[:3]) or "none identified"

    return (
        "[Mock summary - set OPENAI_API_KEY for an AI-generated version]\n\n"
        f"Next week we project {total_projected_units:,.0f} units sold for roughly "
        f"${total_projected_revenue:,.2f} in revenue, led by {top_name}. "
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
    """Generate an executive summary. Falls back to a mock if no API key is set."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return _mock_summary(
            top_predictions, stockout_risks, slow_movers,
            total_projected_units, total_projected_revenue,
        )

    prompt = build_prompt(
        top_predictions, stockout_risks, slow_movers,
        total_projected_units, total_projected_revenue,
    )

    try:
        from openai import OpenAI

        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=0.4,
        )
        return response.choices[0].message.content.strip()
    except Exception as exc:
        return (
            f"[LLM call failed: {exc}]\n\n"
            + _mock_summary(
                top_predictions, stockout_risks, slow_movers,
                total_projected_units, total_projected_revenue,
            )
        )
