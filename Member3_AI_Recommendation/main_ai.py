
"""Entry points for Member 3's AI recommendation module."""

from ai_client import create_ai_client
from lead_utils import categorize_lead
from ai_explanation import generate_lead_explanation
from recommendation import (
    generate_actual_recommendation,
    generate_offline_recommendation
)
from follow_up_strategy import generate_followup_strategy
from ai_sales_assistant import ai_sales_assistant


def run_lead_ai(lead_details, probability, ai_client=None):
    """
    Generate lead category, explanation, recommendation,
    and follow-up strategy with offline fallbacks.
    """

    category = categorize_lead(probability)

    # Try creating the API client.
    if ai_client is None:
        try:
            ai_client = create_ai_client()
        except Exception:
            ai_client = None

    # 1. Generate explanation.
    try:
        if ai_client is None:
            raise RuntimeError("AI API unavailable")

        explanation = generate_lead_explanation(
            ai_client, lead_details, probability, category
        )
    except Exception:
        explanation = (
            f"The ML model estimates a conversion probability of "
            f"{float(probability) * 100:.2f}%. "
            f"The lead is classified as {category} priority. "
            "This is a model estimate, not a guarantee of conversion."
        )

    # 2. Generate sales recommendation.
    try:
        recommendation = generate_actual_recommendation(
            ai_client, lead_details, probability, category
        )
    except Exception:
        recommendation = generate_offline_recommendation(
            lead_details, probability, category
        )

    # 3. Generate follow-up strategy.
    try:
        if ai_client is None:
            raise RuntimeError("AI API unavailable")

        follow_up = generate_followup_strategy(
            ai_client, lead_details, probability, category
        )
    except Exception:
        if float(probability) >= 0.70:
            follow_up = (
                "High priority: Contact within 24 hours. "
                "Use a personalized approach and verify communication preferences."
            )
        elif float(probability) >= 0.40:
            follow_up = (
                "Medium priority: Follow up within 2–3 days. "
                "Share relevant information and understand the lead's needs."
            )
        else:
            follow_up = (
                "Low priority: Follow up within 5–7 days. "
                "Use a helpful, low-pressure approach and review engagement."
            )

    return {
        "probability": float(probability),
        "category": category,
        "explanation": explanation,
        "recommendation": recommendation,
        "follow_up": follow_up
    }


def ask_about_lead(question, lead_details, probability, category, ai_client=None):
    """Answer a salesperson's question with an offline fallback."""

    try:
        if ai_client is None:
            ai_client = create_ai_client()

        return ai_sales_assistant(
            ai_client, question, lead_details, probability, category
        )

    except Exception:
        return (
            "Offline Sales Assistant Response\n\n"
            f"Lead priority: {category}\n"
            f"Estimated conversion probability: {float(probability) * 100:.2f}%\n\n"
            "Suggested approach: Review the available lead attributes, "
            "understand the lead's requirements, and select an appropriate "
            "follow-up action. Verify communication preferences before contacting.\n\n"
            f"Your question: {question}\n\n"
            "This response was generated using the offline fallback "
            "because the AI API was unavailable."
        )


if __name__ == "__main__":
    print("Member 3 AI module ready.")
