import joblib
from .rail_features import extract_rail_features

def predict_rail_file(df, model_path):
    model=joblib.load(model_path)
    x=extract_rail_features(df)
    pred=str(model.predict(x)[0])
    conf=1.0
    if hasattr(model,'predict_proba'): conf=float(model.predict_proba(x)[0].max())
    return pred,conf,x
