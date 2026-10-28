"""Train and evaluate an end-to-end classification pipeline."""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"


def main() -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    dataset = load_breast_cancer(as_frame=True)
    X = dataset.data
    y = dataset.target
    target_names = list(dataset.target_names)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    max_iter=5000,
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
        ]
    )
    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test)

    metrics = {
        "dataset": "Breast Cancer Wisconsin (scikit-learn built-in)",
        "model": "StandardScaler + LogisticRegression",
        "random_state": 42,
        "test_size": 0.20,
        "n_samples": int(len(X)),
        "n_features": int(X.shape[1]),
        "train_samples": int(len(X_train)),
        "test_samples": int(len(X_test)),
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision_weighted": float(
            precision_score(y_test, predictions, average="weighted", zero_division=0)
        ),
        "recall_weighted": float(
            recall_score(y_test, predictions, average="weighted", zero_division=0)
        ),
        "f1_weighted": float(
            f1_score(y_test, predictions, average="weighted", zero_division=0)
        ),
        "class_names": target_names,
    }

    report = classification_report(
        y_test, predictions, target_names=target_names, zero_division=0
    )
    matrix = confusion_matrix(y_test, predictions)

    joblib.dump(
        {
            "pipeline": pipeline,
            "feature_names": list(X.columns),
            "target_names": target_names,
        },
        MODEL_DIR / "breast_cancer_pipeline.joblib",
    )
    (REPORT_DIR / "metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    (REPORT_DIR / "classification_report.txt").write_text(report, encoding="utf-8")
    pd.DataFrame(
        matrix,
        index=[f"actual_{name}" for name in target_names],
        columns=[f"predicted_{name}" for name in target_names],
    ).to_csv(REPORT_DIR / "confusion_matrix.csv")

    print("Training complete.\n")
    print(json.dumps(metrics, indent=2))
    print("\nClassification report:\n")
    print(report)
    print(f"\nSaved model: {MODEL_DIR / 'breast_cancer_pipeline.joblib'}")
    print(f"Saved reports to: {REPORT_DIR}")


if __name__ == "__main__":
    main()
