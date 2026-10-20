
from pathlib import Path

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = ROOT / "data" / "processed" / "adult_cleaned.csv"
OUTPUT_DIR = ROOT / "data" / "processed"


def main():
    df = pd.read_csv(DATA_FILE)

    # Group rows with identical predictors together.
    feature_cols = [c for c in df.columns if c != "income"]
    groups = pd.util.hash_pandas_object(
        df[feature_cols].fillna("<MISSING>").astype(str),
        index=False,
    )

    # Split unique predictor groups into development and test.
    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=0.20,
        random_state=42,
    )

    dev_idx, test_idx = next(
        splitter.split(df, df["income"], groups=groups)
    )

    dev = df.iloc[dev_idx].copy()
    test = df.iloc[test_idx].copy()

    # Split development data into train and validation.
    dev_groups = groups.iloc[dev_idx]

    splitter2 = GroupShuffleSplit(
        n_splits=1,
        test_size=0.25,
        random_state=43,
    )

    train_rel, val_rel = next(
        splitter2.split(
            dev,
            dev["income"],
            groups=dev_groups,
        )
    )

    train = dev.iloc[train_rel].copy()
    val = dev.iloc[val_rel].copy()

    # Verify no predictor group crosses partitions.
    train_groups = set(groups.iloc[train.index])
    val_groups = set(groups.iloc[val.index])
    test_groups = set(groups.iloc[test.index])

    assert train_groups.isdisjoint(val_groups)
    assert train_groups.isdisjoint(test_groups)
    assert val_groups.isdisjoint(test_groups)

    train.to_csv(OUTPUT_DIR / "train.csv", index=False)
    val.to_csv(OUTPUT_DIR / "validation.csv", index=False)
    test.to_csv(OUTPUT_DIR / "test.csv", index=False)

    for name, part in [
        ("Train", train),
        ("Validation", val),
        ("Test", test),
    ]:
        print(f"\n{name}: {len(part)} rows")
        print(part["income"].value_counts(normalize=True).round(4))

    print("\nVerified: predictor groups do not overlap.")
    print("Saved train.csv, validation.csv, and test.csv.")


if __name__ == "__main__":
    main()
