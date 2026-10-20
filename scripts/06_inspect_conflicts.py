
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = ROOT / "data" / "processed" / "adult_cleaned.csv"


def main():
    df = pd.read_csv(DATA_FILE)
    features = [c for c in df.columns if c != "income"]

    counts = (
        df.groupby(features, dropna=False)["income"]
        .value_counts()
        .unstack(fill_value=0)
    )

    if "<=50K" not in counts.columns:
        counts["<=50K"] = 0
    if ">50K" not in counts.columns:
        counts[">50K"] = 0

    conflicts = counts[
        (counts["<=50K"] > 0) & (counts[">50K"] > 0)
    ]

    print("Conflicting feature combinations:", len(conflicts))
    print("\nLabel counts for each conflicting combination:")
    print(conflicts[["<=50K", ">50K"]].to_string())

    print("\nNo records have been changed.")


if __name__ == "__main__":
    main()
