from pathlib import Path
import joblib
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def test_dataset_has_target_and_expected_features():
    df = pd.read_csv(ROOT / "cleaned_leads.csv")
    assert "Converted" in df.columns
    assert set(df["Converted"].dropna().unique()).issubset({0, 1})
    assert len(df.columns) > 1


def test_classifier_loads_and_has_binary_classes():
    model = joblib.load(ROOT / "best_lead_model.pkl")
    assert hasattr(model, "predict_proba")
    assert list(model.classes_) == [0, 1]
    assert int(model.n_features_in_) == 50


def test_classifier_predicts_from_member2_saved_test_matrix():
    # The saved test matrix was produced by Member 2's fitted preprocessor.
    # This verifies the classifier against the exact representation it expects.
    model = joblib.load(ROOT / "best_lead_model.pkl")
    X_test = joblib.load(ROOT / "X_test_processed.pkl")
    probabilities = model.predict_proba(X_test[:8])[:, 1]
    assert len(probabilities) == 8
    assert np.all((probabilities >= 0) & (probabilities <= 1))


def test_preprocessor_loads_and_transforms_raw_dataset():
    # Run this test in the pinned scikit-learn==1.6.1 environment.
    preprocessor = joblib.load(ROOT / "preprocessor.pkl")
    df = pd.read_csv(ROOT / "cleaned_leads.csv")
    X = df.drop(columns=["Converted"])
    transformed = preprocessor.transform(X.iloc[:8])
    assert transformed.shape == (8, 50)


def test_member3_ai_modules_exist():
    ai_dir = ROOT / "Member3_AI_Recommendation"
    for filename in ["main_ai.py", "ai_client.py", "recommendation.py",
                     "ai_explanation.py", "follow_up_strategy.py",
                     "ai_sales_assistant.py"]:
        assert (ai_dir / filename).exists()
