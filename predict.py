"""Rail Corrugation batch inference CLI.

Generates the exact rail_predictions.csv submission format (file_id,
prediction) by running every CSV in --input through the trained model.

Usage:
    python predict.py --input path/to/Test --output rail_predictions.csv
"""

import argparse
from pathlib import Path

import pandas as pd

from src.rail_inference import predict_rail_file


def iter_input_files(input_path):
    p = Path(input_path)
    if p.is_dir():
        return sorted(p.glob("*.csv"))
    return [p]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="A CSV file or a directory of CSV files.")
    parser.add_argument("--output", default="rail_predictions.csv")
    parser.add_argument("--model", default="models/rail_model.joblib")
    args = parser.parse_args()

    rows = []
    for path in iter_input_files(args.input):
        try:
            df = pd.read_csv(path)
            pred, _, _, _ = predict_rail_file(df, args.model)
        except Exception as exc:
            print(f"WARNING: {path.name} failed ({exc}); skipping.")
            continue
        rows.append({"file_id": path.name, "prediction": pred})

    pd.DataFrame(rows, columns=["file_id", "prediction"]).to_csv(args.output, index=False)
    print(f"Wrote {len(rows)} predictions to {args.output}")


if __name__ == "__main__":
    main()
