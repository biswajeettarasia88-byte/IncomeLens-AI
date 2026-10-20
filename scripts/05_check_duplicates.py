
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = ROOT / "data" / "processed" / "adult_cleaned.csv"


def main():
    df = pd.read_csv(DATA_FILE)

    feature_cols = [c for c in df.columns if c != "income"]

    print("--- DUPLICATE INVESTIGATION ---")
    print("Total rows:", len(df))
    print("Exact duplicate rows:", int(df.duplicated().sum()))

    # Identify repeated feature vectors, regardless of target
    repeated_features = df.duplicated(
        subset=feature_cols,
        keep=False,
    )

    print(
        "Rows with repeated feature values:",
        int(repeated_features.sum()),
    )

    # Check whether identical features have conflicting labels
    label_counts = (
        df.groupby(feature_cols, dropna=False)["income"]
        .nunique()
    )

    conflicting = label_counts[label_counts > 1]

    print(
        "Distinct feature combinations with conflicting labels:",
        len(conflicting),
    )

    if len(conflicting):
        print("\nExamples of conflicting records:")
        print(conflicting.head(10))

    print(
        "\nThis is an investigation only. "
        "No records have been deleted."
    )


if __name__ == "__main__":
    main()
