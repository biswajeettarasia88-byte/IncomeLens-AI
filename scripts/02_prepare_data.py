
from pathlib import Path

import pandas as pd

# Project paths
ROOT = Path(__file__).resolve().parents[1]
RAW_FILE = ROOT / "data" / "raw" / "adult.csv"
PROCESSED_DIR = ROOT / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = PROCESSED_DIR / "adult_cleaned.csv"


def main():
    print("Loading audited dataset...")

    if not RAW_FILE.exists():
        raise FileNotFoundError(
            f"Dataset not found: {RAW_FILE}\n"
            "Run scripts/01_audit_data.py first."
        )

    df = pd.read_csv(RAW_FILE)

    print("\nOriginal shape:", df.shape)

    # Standardize the target labels
    df["income"] = (
        df["income"]
        .astype("string")
        .str.strip()
        .str.rstrip(".")
    )

    # Validate the target before proceeding
    valid_labels = {"<=50K", ">50K"}
    actual_labels = set(df["income"].dropna().unique())

    if actual_labels != valid_labels:
        raise ValueError(
            f"Unexpected income labels: {actual_labels}"
        )

    print("\nCleaned income distribution:")
    print(df["income"].value_counts())

    print("\nIncome proportions:")
    print(df["income"].value_counts(normalize=True).round(4))

    # Missing-value report
    print("\nMissing values by column:")
    print(df.isna().sum().sort_values(ascending=False))

    # Numeric feature summary
    print("\nNumerical feature summary:")
    print(df.describe(include="number").T.round(2))

    # Categorical feature summary
    print("\nCategorical feature summary:")
    print(df.describe(include=["object", "string"]).T)

    # Inspect duplicates, but do not delete automatically
    duplicate_count = int(df.duplicated().sum())
    print("\nExact duplicate rows:", duplicate_count)
    print("Duplicate rows are retained for now pending investigation.")

    # Save a new processed copy; raw data remains unchanged
    df.to_csv(OUTPUT_FILE, index=False)

    print("\nProcessed dataset saved to:")
    print(OUTPUT_FILE)

    print("\nData preparation and EDA completed.")


if __name__ == "__main__":
    main()
