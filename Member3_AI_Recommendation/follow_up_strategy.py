from ai_utils import generate_text

def generate_followup_strategy(ai_client, lead_details, probability, category):
    prompt = f"""
You are an AI sales follow-up assistant.

Lead details: {lead_details}
Conversion probability: {float(probability) * 100:.2f}%
Lead category: {category}

Generate:
1. Follow-up objective
2. Recommended communication channel
3. Suggested timing
4. Personalized follow-up message
5. Next recommended action

Use only available lead information. Do not invent customer details, dates,
URLs, or company resources. Label timing as a suggested plan, not a fact from
the dataset. Do not assume marketing consent from a Do Not Email field alone.
Use placeholders for missing names or contact information. Do not guarantee
conversion.
"""
    return generate_text(
        ai_client, prompt,
        "You are an expert sales follow-up strategist.",
        max_tokens=650
    )
