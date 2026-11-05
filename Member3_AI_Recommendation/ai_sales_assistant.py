from ai_utils import generate_text

def ai_sales_assistant(ai_client, question, lead_details, probability, category):
    prompt = f"""
You are an expert AI sales assistant.

Lead details: {lead_details}
ML conversion probability: {float(probability) * 100:.2f}%
Lead category: {category}
Salesperson's question: {question}

Answer clearly using the supplied lead information. Give practical sales
suggestions. Do not invent customer information or guarantee conversion.
Do not claim a feature caused the prediction without supporting model
explainability evidence. If the supplied data does not answer the question,
say what information is missing.
"""
    return generate_text(
        ai_client, prompt,
        "You are an intelligent, accurate AI sales assistant.",
        max_tokens=650
    )
