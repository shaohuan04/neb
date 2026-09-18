import json

import pandas as pd

from training.train_rail import benchmark_models, build_dataset, feature_importances
from sklearn.model_selection import StratifiedKFold
from src.rail_inference import predict_rail_file


def _write_dataset(tmp_path, rail_dataframe_factory):
    train_dir = tmp_path / "Train"
    train_dir.mkdir()
    labels = []
    # A handful of files per class, small row counts to keep the test fast.
    plan = [(None, "Normal", 6), (1, "Side I", 3), (2, "Side II", 3)]
    idx = 1
    for fault_side, label, count in plan:
        for i in range(count):
            df = rail_dataframe_factory(n_rows=1500, fault_side=fault_side, seed=idx)
            fname = f"Train{idx}.csv"
            df.to_csv(train_dir / fname, index=False)
            labels.append({"filename": fname, "label": label})
            idx += 1
    labels_path = tmp_path / "Train_Labels.csv"
    pd.DataFrame(labels).to_csv(labels_path, index=False)
    return train_dir, labels_path


def test_end_to_end_training_and_inference(tmp_path, rail_dataframe_factory):
    train_dir, labels_path = _write_dataset(tmp_path, rail_dataframe_factory)

    X, y = build_dataset(train_dir, labels_path)
    assert X.shape[0] == 12
    assert set(y.unique()) == {"Normal", "Side I", "Side II"}

    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=0)
    scores = benchmark_models(X, y, cv)
    assert set(scores.keys()) == {"random_forest", "gradient_boosting", "logistic_regression"}
    for s in scores.values():
        assert 0.0 <= s["mean"] <= 1.0

    from training.train_rail import CANDIDATE_MODELS
    best_name = max(scores, key=lambda k: scores[k]["mean"])
    model = CANDIDATE_MODELS[best_name]
    model.fit(X, y)

    importances = feature_importances(model, list(X.columns))
    assert isinstance(importances, list)

    import joblib
    model_path = tmp_path / "rail_model.joblib"
    joblib.dump(model, model_path)
    meta_path = tmp_path / "rail_model_meta.json"
    meta_path.write_text(json.dumps({"selected_model": best_name, "cv_scores_macro_f1": scores}))

    test_df = pd.read_csv(train_dir / "Train1.csv")
    pred, conf, feats, probs = predict_rail_file(test_df, model_path)
    assert pred in {"Normal", "Side I", "Side II"}
    assert 0.0 <= conf <= 1.0
