def categorize_lead(probability):
    """High >= 0.70; Medium >= 0.40 and < 0.70; otherwise Low."""
    probability = float(probability)
    if not 0 <= probability <= 1:
        raise ValueError("probability must be between 0 and 1.")
    if probability >= 0.70:
        return "High"
    if probability >= 0.40:
        return "Medium"
    return "Low"
