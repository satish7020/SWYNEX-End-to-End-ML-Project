"""Run inference with a previously trained pipeline."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.datasets import load_breast_cancer


ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "breast_cancer_pipeline.joblib"


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict a breast-cancer dataset class.")
    parser.add_argument(
        "--input",
        type=Path,
        help="Path to a JSON object containing all 30 numeric feature values.",
    )
    args = parser.parse_args()

    if not MODEL_PATH.exists():
        raise SystemExit("Model not found. Run `python src/train.py` first.")

    artifact = joblib.load(MODEL_PATH)
    pipeline = artifact["pipeline"]
    feature_names = artifact["feature_names"]
    target_names = artifact["target_names"]

    if args.input:
        payload = json.loads(args.input.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise SystemExit("Input JSON must be an object mapping feature names to numbers.")
        missing = sorted(set(feature_names) - set(payload))
        extra = sorted(set(payload) - set(feature_names))
        if missing:
            raise SystemExit(f"Missing feature(s): {', '.join(missing)}")
        if extra:
            raise SystemExit(f"Unexpected feature(s): {', '.join(extra)}")
        sample = pd.DataFrame([[payload[name] for name in feature_names]], columns=feature_names)
    else:
        # Demonstration only: use the first available row as a sample.
        dataset = load_breast_cancer(as_frame=True)
        sample = dataset.data.iloc[[0]][feature_names]

    predicted_class = int(pipeline.predict(sample)[0])
    result = {
        "predicted_class": predicted_class,
        "predicted_label": target_names[predicted_class],
        "note": "Educational benchmark only; not for medical diagnosis.",
    }
    if hasattr(pipeline, "predict_proba"):
        probabilities = pipeline.predict_proba(sample)[0]
        result["class_probabilities"] = {
            target_names[i]: float(probabilities[i]) for i in range(len(target_names))
        }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
