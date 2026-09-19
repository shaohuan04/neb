"""Train the Rail Corrugation classifier.

Usage:
    python -m training.train_rail --train-dir path/to/Train --labels path/to/Train_Labels.csv

Train_Labels.csv must have `filename` and `label` columns (per
Rail_Corrugation_Info_Kit.md section 2.2), with label in
{"Normal", "Side I", "Side II"}.

Split strategy: the Info Kit documents no grouping variable (e.g. run ID,
operating condition) alongside each file, and every file is already one
self-contained 1-second recording — so a stratified k-fold over the file
labels is used as the train/validation split. If a future data drop adds
a grouping column, prefer sklearn's GroupKFold over this instead, to keep
recordings from the same run out of both the train and validation folds.

Model selection: several classifiers are benchmarked under identical
stratified 5-fold CV (scored on macro F1, the subsystem's actual grading
metric) and the best-scoring one is refit on the full training set and
saved. Metadata (per-model CV scores, feature names, feature importances
where available) is written alongside the model so the app can show its
provenance instead of a hardcoded number.
"""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.rail_features import extract_rail_features

CANDIDATE_MODELS = {
    "random_forest": RandomForestClassifier(
        n_estimators=400, class_weight="balanced", random_state=42, n_jobs=-1
    ),
    "gradient_boosting": GradientBoostingClassifier(random_state=42),
    "logistic_regression": make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42),
    ),
}


def build_dataset(train_dir, labels_path):
    labels = pd.read_csv(labels_path)
    rows, y = [], []
    for _, r in labels.iterrows():
        path = Path(train_dir) / r["filename"]
        rows.append(extract_rail_features(pd.read_csv(path)).iloc[0])
        y.append(r["label"])
    X = pd.DataFrame(rows).fillna(0)
    return X, pd.Series(y)


def benchmark_models(X, y, cv):
    scores = {}
    for name, model in CANDIDATE_MODELS.items():
        s = cross_val_score(model, X, y, scoring="f1_macro", cv=cv, n_jobs=-1)
        scores[name] = {"mean": float(s.mean()), "std": float(s.std())}
        print(f"{name}: macro F1 = {s.mean():.4f} +/- {s.std():.4f}")
    return scores


def feature_importances(model, feature_names, top_n=20):
    """Works for both tree ensembles (feature_importances_) and linear
    models (coef_, one row per class for multi-class) so the app's
    explainability panel isn't silently empty when a linear model wins
    the benchmark.
    """
    estimator = model.steps[-1][1] if hasattr(model, "steps") else model
    if hasattr(estimator, "feature_importances_"):
        importances = estimator.feature_importances_
    elif hasattr(estimator, "coef_"):
        importances = np.mean(np.abs(estimator.coef_), axis=0)
    else:
        return []
    order = np.argsort(importances)[::-1][:top_n]
    return [{"feature": feature_names[i], "importance": float(importances[i])} for i in order]


SEVERITY_FEATURES = ["side1_rms", "side2_rms", "side_rms_ratio", "side_std_ratio", "side_peak_ratio"]


def healthy_baselines(X, y):
    """Percentile spread of key indicators across the *Normal* files only.

    The app grades a new recording against these, so it can say "higher than
    97% of healthy recordings" instead of showing a bare number the operator
    has no reference point for.
    """
    healthy = X[y == "Normal"]
    if healthy.empty:
        return {}
    out = {}
    for feat in SEVERITY_FEATURES:
        if feat not in healthy.columns:
            continue
        values = healthy[feat].astype(float)
        out[feat] = {
            "p50": float(values.quantile(0.50)),
            "p90": float(values.quantile(0.90)),
            "p97": float(values.quantile(0.97)),
            "p99": float(values.quantile(0.99)),
            "max": float(values.max()),
        }
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train-dir", required=True)
    p.add_argument("--labels", required=True)
    p.add_argument("--output", default="models/rail_model.joblib")
    p.add_argument("--meta-output", default="models/rail_model_meta.json")
    a = p.parse_args()

    X, y = build_dataset(a.train_dir, a.labels)
    print(f"Loaded {len(X)} training files. Class distribution:\n{y.value_counts()}")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = benchmark_models(X, y, cv)
    best_name = max(scores, key=lambda k: scores[k]["mean"])
    best_model = CANDIDATE_MODELS[best_name]
    print(f"Selected best model: {best_name} (macro F1 = {scores[best_name]['mean']:.4f})")

    best_model.fit(X, y)
    Path(a.output).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, a.output)

    meta = {
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "n_training_files": len(X),
        "class_distribution": y.value_counts().to_dict(),
        "selected_model": best_name,
        "cv_scores_macro_f1": scores,
        "feature_names": list(X.columns),
        "top_feature_importances": feature_importances(best_model, list(X.columns)),
        "healthy_baselines": healthy_baselines(X, y),
    }
    Path(a.meta_output).write_text(json.dumps(meta, indent=2))
    print(f"Saved model to {a.output} and metadata to {a.meta_output}")


if __name__ == "__main__":
    main()
