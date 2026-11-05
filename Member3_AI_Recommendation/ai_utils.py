import time

def generate_text(ai_client, prompt, system_message, max_tokens=650, retries=3):
    """Call OpenRouter, retrying empty responses and temporary failures."""
    last_error = None
    for attempt in range(retries):
        try:
            response = ai_client.chat.completions.create(
                model="nvidia/nemotron-3.5-super-vl:free",
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=max_tokens
            )
            if response.choices:
                text = response.choices[0].message.content
                if text and text.strip():
                    return text.strip()
            last_error = "The model returned an empty response."
        except Exception as exc:
            last_error = str(exc)
        if attempt < retries - 1:
            time.sleep(2)
    raise RuntimeError(f"OpenRouter failed after {retries} attempts: {last_error}")
