
from pathlib import Path
import sys

import joblib
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from incomelens.preprocessing import (
    build_preprocessor,
    split_features_target,
)


def main():
    data_dir = ROOT / "data" / "processed"
    model_dir = ROOT / "models"
    report_dir = ROOT / "reports"

    model_dir.mkdir(exist_ok=True)
    report_dir.mkdir(exist_ok=True)

    train_df = pd.read_csv(data_dir / "train.csv")
    val_df = pd.read_csv(data_dir / "validation.csv")
    test_df = pd.read_csv(data_dir / "test.csv")

    # Fit on training + validation, never on test.
    development_df = pd.concat(
        [train_df, val_df],
        ignore_index=True,
    )

    X_dev, y_dev = split_features_target(development_df)
    X_test, y_test = split_features_target(test_df)

    model = Pipeline([
        ("preprocessor", build_preprocessor()),
        (
            "classifier",
            RandomForestClassifier(
                n_estimators=150,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1,
            ),
        ),
    ])

    print("Training final selected model...")
    model.fit(X_dev, y_dev)

    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, predictions),
        "balanced_accuracy": balanced_accuracy_score(
            y_test, predictions
        ),
        "precision_high_income": precision_score(
            y_test, predictions, zero_division=0
        ),
        "recall_high_income": recall_score(
            y_test, predictions, zero_division=0
        ),
        "f1_high_income": f1_score(
            y_test, predictions, zero_division=0
        ),
        "roc_auc": roc_auc_score(y_test, probabilities),
    }

    print("\n--- FINAL TEST RESULTS ---")
    for name, value in metrics.items():
        print(f"{name}: {value:.4f}")

    print("\nClassification report:")
    print(classification_report(
        y_test,
        predictions,
        labels=[0, 1],
        target_names=["<=50K", ">50K"],
        zero_division=0,
    ))

    matrix = confusion_matrix(
        y_test, predictions, labels=[0, 1]
    )
    print("Confusion matrix (rows=actual, columns=predicted):")
    print(matrix)

    pd.DataFrame(
        matrix,
        index=["actual_<=50K", "actual_>50K"],
        columns=["predicted_<=50K", "predicted_>50K"],
    ).to_csv(report_dir / "confusion_matrix.csv")

    pd.DataFrame([metrics]).to_csv(
        report_dir / "final_test_metrics.csv",
        index=False,
    )

    joblib.dump(
        model,
        model_dir / "random_forest_final.joblib",
    )

    print("\nSaved model: models/random_forest_final.joblib")
    print("Saved metrics: reports/final_test_metrics.csv")
    print("Final test evaluation completed.")


if __name__ == "__main__":
    main()
