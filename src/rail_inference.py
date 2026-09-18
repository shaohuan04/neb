import joblib
from .rail_features import extract_rail_features

def predict_rail_file(df, model_path):
    model = joblib.load(model_path)
    x = extract_rail_features(df)
    pred = str(model.predict(x)[0])
    probabilities = {}
    confidence = 1.0
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(x)[0]
        probabilities = {str(c): float(p) for c, p in zip(model.classes_, proba)}
        confidence = float(max(proba))
    return pred, confidence, x, probabilities
