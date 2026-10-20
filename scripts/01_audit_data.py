
from pathlib import Path

import pandas as pd
from ucimlrepo import fetch_ucirepo

# Find the project root automatically
ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

def main():
    print("Downloading the UCI Adult Income dataset...")

    adult = fetch_ucirepo(id=2)
    X = adult.data.features.copy()
    y = adult.data.targets.copy()

    # Combine input features and target
    df = X.copy()
    df["income"] = y.iloc[:, 0].to_numpy()

    # Clean column names and text whitespace
    df.columns = df.columns.str.strip()

    for col in df.select_dtypes(include=["object", "string"]).columns:
        df[col] = df[col].str.strip()

    # Represent '?' as a missing value
    df = df.replace(r"^\s*\?\s*$", pd.NA, regex=True)

    print("\n--- DATA AUDIT ---")
    print(f"Rows: {df.shape[0]}")
    print(f"Columns including target: {df.shape[1]}")

    print("\nColumn names:")
    print(df.columns.tolist())

    print("\nMissing values:")
    print(df.isna().sum().sort_values(ascending=False))

    print("\nIncome distribution:")
    print(df["income"].value_counts(dropna=False))

    print("\nIncome proportions:")
    print(df["income"].value_counts(normalize=True).round(4))

    print(f"\nDuplicate rows: {df.duplicated().sum()}")

    output = RAW_DIR / "adult.csv"
    df.to_csv(output, index=False)

    print(f"\nDataset saved to: {output}")
    print("Audit completed.")

if __name__ == "__main__":
    main()
