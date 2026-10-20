
from pathlib import Path
import sys

import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from incomelens.preprocessing import (
    build_preprocessor,
    split_features_target,
)


def main():
    df = pd.read_csv(
        ROOT / "data" / "processed" / "adult_cleaned.csv"
    )
    X, y = split_features_target(df)

    # Reserve a test set; use validation for model selection
    X_dev, X_test, y_dev, y_test = train_test_split(
        X, y,
        test_size=0.20,
        stratify=y,
        random_state=42,
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_dev, y_dev,
        test_size=0.25,
        stratify=y_dev,
        random_state=42,
    )

    print("Train:", len(X_train))
    print("Validation:", len(X_val))
    print("Test:", len(X_test))

    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            random_state=42,
        ),
        "Random Forest": RandomForestClassifier(
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
            ("classifier", clone(classifier)),
        ])

        print(f"\nTraining {name}...")
        pipeline.fit(X_train, y_train)

        # Use validation data for initial model comparison
        predictions = pipeline.predict(X_val)
        probabilities = pipeline.predict_proba(X_val)[:, 1]

        result = {
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

        results.append(result)

        print(pd.Series(result).to_string())

    results_df = pd.DataFrame(results)
    report_dir = ROOT / "reports"
    report_dir.mkdir(exist_ok=True)

    results_df.to_csv(
        report_dir / "validation_model_comparison.csv",
        index=False,
    )

    print("\nValidation comparison:")
    print(results_df.round(4).to_string(index=False))

    print("\nSaved to reports/validation_model_comparison.csv")
    print(
        "Choose a model using validation results. "
        "Do not use the test set for tuning."
    )


if __name__ == "__main__":
    main()
