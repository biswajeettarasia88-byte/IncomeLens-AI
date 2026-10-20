
from pathlib import Path
import sys

import joblib
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
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
    model_dir.mkdir(parents=True, exist_ok=True)

    train_df = pd.read_csv(data_dir / "train.csv")
    val_df = pd.read_csv(data_dir / "validation.csv")

    X_train, y_train = split_features_target(train_df)
    X_val, y_val = split_features_target(val_df)

    models = {
        "logistic_regression": LogisticRegression(
            max_iter=1000,
            random_state=42,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=150,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
        ),
    }

    results = []

    for name, classifier in models.items():
        pipeline = Pipeline([
            ("preprocessor", build_preprocessor()),
            ("classifier", classifier),
        ])

        print(f"\nTraining {name}...")
        pipeline.fit(X_train, y_train)

        predictions = pipeline.predict(X_val)
        probabilities = pipeline.predict_proba(X_val)[:, 1]

        metrics = {
            "model": name,
            "accuracy": accuracy_score(y_val, predictions),
            "precision_high_income": precision_score(
                y_val, predictions, zero_division=0
            ),
            "recall_high_income": recall_score(
                y_val, predictions, zero_division=0
            ),
            "f1_high_income": f1_score(
                y_val, predictions, zero_division=0
            ),
            "roc_auc": roc_auc_score(y_val, probabilities),
        }

        results.append(metrics)

        print(pd.Series(metrics).to_string())

        model_path = model_dir / f"{name}_grouped.joblib"
        joblib.dump(pipeline, model_path)
        print("Saved:", model_path)

    results_df = pd.DataFrame(results)
    report_dir = ROOT / "reports"
    report_dir.mkdir(exist_ok=True)

    results_df.to_csv(
        report_dir / "grouped_validation_comparison.csv",
        index=False,
    )

    print("\n--- GROUPED VALIDATION COMPARISON ---")
    print(results_df.round(4).to_string(index=False))
    print("\nSaved: reports/grouped_validation_comparison.csv")
    print("Test set was not loaded or evaluated.")


if __name__ == "__main__":
    main()
