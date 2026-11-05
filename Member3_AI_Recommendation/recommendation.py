
from ai_utils import generate_text


def generate_actual_recommendation(ai_client, lead_details, probability, category):
    prompt = f"""
You are an expert AI sales recommendation assistant.

Actual lead details: {lead_details}
Conversion probability: {float(probability) * 100:.2f}%
Lead category: {category}

Generate a report containing:
1. Lead profile summary
2. Prediction interpretation
3. Recommended sales action
4. Personalized follow-up strategy
5. Suggested communication channel
6. Suggested follow-up timing
7. Personalized sales message
8. Next steps

Use only supplied attributes. Distinguish facts from suggestions.
Do not guarantee conversion or assume marketing consent.
"""

    try:
        return generate_text(
            ai_client, prompt,
            "You are an expert AI sales analyst.",
            max_tokens=800
        )

    except Exception:
        return generate_offline_recommendation(
            lead_details, probability, category
        )


def generate_offline_recommendation(lead_details, probability, category):
    probability_percent = float(probability) * 100

    if probability_percent >= 70:
        action = "Prioritize this lead and arrange a personalized sales discussion."
        timing = "Within 24 hours"
        priority = "High"
    elif probability_percent >= 40:
        action = "Share relevant product information and address possible concerns."
        timing = "Within 2–3 days"
        priority = "Medium"
    else:
        action = "Use a low-pressure educational follow-up to understand the lead's needs."
        timing = "Within 5–7 days"
        priority = "Low"

    details = str(lead_details)

    return f"""
## AI Sales Recommendation
<text>

### 1. Lead Profile Summary
Lead attributes: {details}

### 2. Prediction Interpretation
Predicted conversion probability: {probability_percent:.2f}%
Lead category: {category}

This is an ML prediction, not a guarantee of conversion.

### 3. Recommended Sales Action
{action}

### 4. Personalized Follow-up Strategy
Review the available lead attributes and tailor the conversation
to the lead's recorded interests and requirements.

### 5. Suggested Communication Channel
Use the communication channel recorded in the lead details,
if available and permitted by the lead's preferences.

### 6. Suggested Follow-up Timing
{timing}

### 7. Personalized Sales Message
Hello,

Thank you for your interest. We would be happy to understand
your requirements and help you explore suitable options.

Please let us know a convenient time to connect.

Regards,
Sales Team

### 8. Next Steps
1. Review the lead details.
2. Verify communication preferences and consent.
3. Follow the suggested sales action.
4. Record the outcome of the interaction.

Recommendation priority: {priority}

Note: This recommendation was generated using the offline
fallback because the AI API was unavailable.
</text>
"""
