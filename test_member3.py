
from dotenv import load_dotenv
import os
import sys

load_dotenv()

sys.path.insert(0, "Member3_AI_Recommendation")

from main_ai import run_lead_ai

lead_details = {
    "Lead Origin": "API",
    "Lead Source": "olark chat",
    "Do Not Email": "No",
    "TotalVisits": 0,
    "Total Time Spent on Website": 0,
    "Page Views Per Visit": 0,
    "Last Activity": "Page Visited on Website",
    "Specialization": "Unknown",
    "What is your current occupation": "Unemployed",
    "A free copy of Mastering The Interview": "No"
}

probability = 0.1518

try:
    result = run_lead_ai(lead_details, probability)

    print("\n--- AI LEAD RESULTS ---")
    print("Probability:", result["probability"])
    print("Category:", result["category"])

    print("\n--- EXPLANATION ---")
    print(result["explanation"])

    print("\n--- RECOMMENDATION ---")
    print(result["recommendation"])

    print("\n--- FOLLOW-UP STRATEGY ---")
    print(result["follow_up"])

except Exception as e:
    print("AI test failed:", str(e))
