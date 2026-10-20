
from pathlib import Path

import joblib
import pandas as pd

from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from incomelens.preprocessing import (
    build_preprocessor,
    split_features_target,
)

DATA_FILE = ROOT / "data" / "processed" / "adult_cleaned.csv"
MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)


def main():
    print("Loading cleaned dataset...")
    df = pd.read_csv(DATA_FILE)

    X, y = split_features_target(df)

    # Create a stratified holdout split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    print("Training rows:", len(X_train))
    print("Testing rows:", len(X_test))

    # Baseline 1: always predict the most common class
    dummy = DummyClassifier(strategy="most_frequent")
    dummy.fit(X_train, y_train)
    dummy_predictions = dummy.predict(X_test)

    print("\n--- Dummy baseline ---")
    print("Accuracy:", round(
        accuracy_score(y_test, dummy_predictions), 4
    ))

    # Baseline 2: logistic regression
    model = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1000,
                    class_weight=None,
                    random_state=42,
                ),
            ),
        ]
    )

    print("\nTraining logistic regression...")
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    print("\n--- Logistic regression results ---")
    print("Accuracy:", round(
        accuracy_score(y_test, predictions), 4
    ))
    print("ROC-AUC:", round(
        roc_auc_score(y_test, probabilities), 4
    ))
    print("\nClassification report:")
    print(classification_report(
        y_test,
        predictions,
        target_names=["<=50K", ">50K"],
        zero_division=0,
    ))

    # Save the complete fitted pipeline
    model_path = MODEL_DIR / "logistic_regression.joblib"
    joblib.dump(model, model_path)

    print("\nModel saved to:", model_path)
    print("Baseline training completed.")


if __name__ == "__main__":
    main()
