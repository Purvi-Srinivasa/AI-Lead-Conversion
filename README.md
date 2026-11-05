# LeadPulse — AI-Based Lead Conversion Prediction and Intelligent Sales Recommendation

Integrated project assembled from the submitted Member 2, Member 3 and Member 4 files.

## What is included
- Streamlit sales analytics dashboard (`app.py`)
- Cleaned lead dataset (`cleaned_leads.csv`)
- Member 2 fitted preprocessor (`preprocessor.pkl`)
- Member 2 XGBoost binary classifier (`best_lead_model.pkl`)
- Member 2 transformed test matrix (`X_test_processed.pkl`) for reproducible classifier smoke tests
- Processed feature-name list and verified model evaluation metrics
- Member 3 OpenRouter-backed AI explanation, recommendation, follow-up and sales assistant modules
- Automated tests and integration status

## Model connection
The dashboard expects the raw lead attributes to be transformed by `preprocessor.pkl`, then passed to `best_lead_model.pkl`. The classifier returns class probabilities for `Converted=0` and `Converted=1`. The dashboard uses probability at index 1 as conversion probability.

The supplied classifier expects 50 transformed features. Member 2's saved processed test matrix has shape 1,848 × 50, and the classifier successfully generated probabilities from it.

## Verified classifier evaluation
Using Member 2's saved test matrix and target (1,848 test records), the supplied XGBoost classifier achieved:

| Metric | Result |
|---|---:|
| Accuracy | 82.14% |
| Precision (Converted=1) | 76.53% |
| Recall (Converted=1) | 77.39% |
| F1 score (Converted=1) | 76.96% |
| ROC-AUC | 88.94% |

These figures evaluate the classifier on Member 2's saved transformed test set. They do not independently verify the raw-input preprocessing step.

## Setup — Windows PowerShell
Use Python 3.11 or 3.12 and the exact dependency versions in `requirements.txt`.

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Run the dashboard
```powershell
streamlit run app.py
```

## AI setup
Member 3's code uses OpenRouter through the OpenAI-compatible Python client. Set the API key in PowerShell; never commit the real key or `.env` file.

```powershell
$env:OPENROUTER_API_KEY="YOUR_KEY_HERE"
streamlit run app.py
```

If the key is not configured or the provider is unavailable, ML prediction can still be used; AI recommendation and assistant requests will fail gracefully with an error message.

## Tests
```powershell
python -m compileall app.py Member3_AI_Recommendation tests
python -m pytest -q
```

The preprocessor was serialized with scikit-learn 1.6.1. It failed to load under scikit-learn 1.8.0 in the preparation environment because of a private scikit-learn class change. Run the tests after installing the pinned version above. Do not change scikit-learn versions without retraining/reserializing the preprocessor.

## Security
- Do not commit API keys, `.env`, or private customer information.
- Model outputs are estimates, not guarantees.
- AI-generated explanations and recommendations are advisory and should be reviewed by a salesperson.
