from ai_utils import generate_text

def generate_lead_explanation(ai_client, lead_details, probability, category):
    prompt = f"""
Explain this machine-learning lead conversion prediction.

Lead details: {lead_details}
Conversion probability: {float(probability) * 100:.2f}%
Lead category: {category}

Include what the probability means, observed attributes, positive indicators
(if any), possible concerns supported by the data, and a short business
interpretation.

Do not claim an attribute caused the prediction unless model explainability
evidence is supplied. Distinguish facts from interpretations, point out
inconsistent or missing attributes, do not invent customer details, and do
not guarantee conversion.
"""
    return generate_text(
        ai_client, prompt,
        "You are an accurate AI sales analyst. Explain model outputs cautiously.",
        max_tokens=650
    )
