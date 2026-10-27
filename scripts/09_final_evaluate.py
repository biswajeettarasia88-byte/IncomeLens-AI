import json
from pathlib import Path
import sys

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from incomelens.data import load_test_data
from incomelens.evaluation import (
    NEGATIVE_LABEL_STR,
    POSITIVE_CLASS_INT,
    POSITIVE_LABEL_STR,
    evaluate_predictions,
    get_positive_class_index,
)
from incomelens.preprocessing import split_features_target

MODEL_PATH = ROOT / "models" / "random_forest_final.joblib"
REPORT_DIR = ROOT / "reports"


def main():
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"Production model not found at '{MODEL_PATH}'. "
            "Please ensure the model is trained before running final evaluation."
        )

    print(f"Loading trained production model: {MODEL_PATH}")
    model = joblib.load(MODEL_PATH)

    print("Loading held-out test split (data/processed/test.csv)...")
    test_df = load_test_data()

    X_test, y_test = split_features_target(test_df)
    print(f"Test instances: {len(X_test)}")

    # Validate probability column selection using model.classes_
    pos_idx = get_positive_class_index(
        model.classes_, positive_class=POSITIVE_CLASS_INT
    )

    print("Generating predictions and class probabilities on test set...")
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, pos_idx]

    metrics, cm_df, report_str = evaluate_predictions(
        y_true=y_test,
        y_pred=predictions,
        y_prob=probabilities,
        positive_class=POSITIVE_CLASS_INT,
    )

    print("\n--- FINAL TEST RESULTS ---")
    for name, value in metrics.items():
        print(f"{name}: {value:.4f}")

    print("\nClassification report:")
    print(report_str)

    print("Confusion matrix (rows=actual, columns=predicted):")
    print(cm_df)

    # Save evaluation reports (does NOT touch model binary)
    cm_path = REPORT_DIR / "confusion_matrix.csv"
    metrics_path = REPORT_DIR / "final_test_metrics.csv"
    manifest_path = REPORT_DIR / "evaluation_manifest.json"

    cm_df.to_csv(cm_path)
    pd.DataFrame([metrics]).to_csv(metrics_path, index=False)

    # Record model artifact and data split used
    manifest = {
        "analysis": "final_evaluation",
        "model_artifact": str(MODEL_PATH),
        "model_type": type(model.named_steps.get("classifier", model)).__name__ if hasattr(model, "named_steps") else type(model).__name__,
        "data_split": "data/processed/test.csv",
        "sample_count": len(X_test),
        "positive_class": POSITIVE_LABEL_STR,
        "metrics": metrics,
    }
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    # Post-execution verification of all generated artifacts
    for artifact in [cm_path, metrics_path, manifest_path]:
        if not artifact.is_file() or artifact.stat().st_size == 0:
            raise RuntimeError(f"Generated evaluation artifact is missing or empty: {artifact}")

    print(f"\nSaved confusion matrix: {cm_path}")
    print(f"Saved metrics: {metrics_path}")
    print(f"Saved manifest: {manifest_path}")
    print("Verified: All evaluation artifacts generated and non-empty.")
    print("Final test evaluation completed successfully without model retraining.")


if __name__ == "__main__":
    main()
