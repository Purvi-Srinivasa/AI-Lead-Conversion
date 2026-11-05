import os
from openai import OpenAI

def create_ai_client():
    """Create an OpenRouter client using OPENROUTER_API_KEY."""
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("Set the OPENROUTER_API_KEY environment variable.")
    return OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)
