"""Data loading and validation module for IncomeLens AI."""

import csv
import logging
from pathlib import Path
from typing import List, Optional, Union

import pandas as pd

from .preprocessing import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    TARGET_COLUMN,
)

logger = logging.getLogger(__name__)

# Canonical project paths
ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

RAW_DATA_PATH = RAW_DIR / "adult.csv"
CLEANED_DATA_PATH = PROCESSED_DIR / "adult_cleaned.csv"
TRAIN_DATA_PATH = PROCESSED_DIR / "train.csv"
VALIDATION_DATA_PATH = PROCESSED_DIR / "validation.csv"
TEST_DATA_PATH = PROCESSED_DIR / "test.csv"

# Canonical feature schema
EXPECTED_COLUMNS: List[str] = [
    "age",
    "workclass",
    "fnlwgt",
    "education",
    "education-num",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "capital-gain",
    "capital-loss",
    "hours-per-week",
    "native-country",
    TARGET_COLUMN,
]
EXPECTED_FEATURES: List[str] = [c for c in EXPECTED_COLUMNS if c != TARGET_COLUMN]


def resolve_data_path(path: Optional[Union[str, Path]] = None) -> Path:
    """Resolve a data file path against project root if relative."""
    if path is None:
        return TRAIN_DATA_PATH
    if not isinstance(path, (str, Path)):
        raise TypeError(f"Expected path as str or Path, got {type(path).__name__}")
    p = Path(path)
    if not p.is_absolute():
        if p.exists():
            return p.resolve()
        return (ROOT_DIR / p).resolve()
    return p


def validate_dataframe(
    df: pd.DataFrame,
    require_target: bool = True,
    dataset_name: str = "dataset",
) -> pd.DataFrame:
    """Validate that a DataFrame matches the required schema without mutating or dropping records.

    Parameters
    ----------
    df : pd.DataFrame
        The loaded DataFrame to validate.
    require_target : bool, default=True
        Whether the target column ('income') must be present.
    dataset_name : str, default="dataset"
        Context name for descriptive error messages.

    Returns
    -------
    pd.DataFrame
        The unmodified DataFrame.

    Raises
    ------
    TypeError
        If df is not a pandas DataFrame, or if columns have unexpected data types.
    ValueError
        If df is empty, has duplicate columns, is missing required features, or target contains missing values.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"Expected pandas DataFrame, got {type(df).__name__}")

    if df.empty:
        raise ValueError(f"The {dataset_name} is empty (0 rows).")

    # Check for duplicate column names in DataFrame
    if not df.columns.is_unique:
        duplicate_cols = sorted(set(df.columns[df.columns.duplicated(keep=False)]))
        raise ValueError(
            f"Duplicate column names detected in {dataset_name}: {duplicate_cols}"
        )

    # Check required columns
    expected = EXPECTED_COLUMNS if require_target else EXPECTED_FEATURES
    missing = [col for col in expected if col not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required columns in {dataset_name}: {missing}. "
            f"Expected schema: {expected}"
        )

    # Validate target column does not contain missing values when required
    if require_target and TARGET_COLUMN in df.columns:
        if df[TARGET_COLUMN].isna().any():
            null_count = int(df[TARGET_COLUMN].isna().sum())
            raise ValueError(
                f"Target column '{TARGET_COLUMN}' contains {null_count} missing value(s) in {dataset_name}."
            )

    # Validate unexpected data types in numeric columns
    for col in NUMERIC_FEATURES:
        if col in df.columns:
            if not pd.api.types.is_numeric_dtype(df[col]):
                non_nulls = df[col].dropna()
                if not non_nulls.empty:
                    try:
                        pd.to_numeric(non_nulls, errors="raise")
                    except (ValueError, TypeError) as e:
                        raise TypeError(
                            f"Column '{col}' has unexpected non-numeric data type or unparseable values in {dataset_name}: {e}"
                        )

    # Validate unexpected nested collections in categorical columns
    for col in CATEGORICAL_FEATURES:
        if col in df.columns:
            non_nulls = df[col].dropna()
            if not non_nulls.empty:
                first_val = non_nulls.iloc[0]
                if isinstance(first_val, (dict, list, set, tuple)):
                    raise TypeError(
                        f"Column '{col}' contains unexpected nested collection data type ({type(first_val).__name__}) in {dataset_name}."
                    )

    return df


def check_schema_consistency(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    check_dtypes: bool = True,
) -> bool:
    """Verify that two DataFrames share identical column names, ordering, and compatible data types.

    Parameters
    ----------
    df1, df2 : pd.DataFrame
        DataFrames to compare.
    check_dtypes : bool, default=True
        Whether to verify data type category compatibility.

    Returns
    -------
    bool
        True if schemas match.

    Raises
    ------
    ValueError
        If columns, column ordering, or dtypes mismatch.
    """
    if not isinstance(df1, pd.DataFrame) or not isinstance(df2, pd.DataFrame):
        raise TypeError("Both inputs must be pandas DataFrames.")

    cols1 = list(df1.columns)
    cols2 = list(df2.columns)

    if cols1 != cols2:
        raise ValueError(
            f"Column schema mismatch:\n"
            f"DF1 columns ({len(cols1)}): {cols1}\n"
            f"DF2 columns ({len(cols2)}): {cols2}"
        )

    if check_dtypes:
        for col in cols1:
            is_num1 = pd.api.types.is_numeric_dtype(df1[col])
            is_num2 = pd.api.types.is_numeric_dtype(df2[col])
            if is_num1 != is_num2:
                raise ValueError(
                    f"Data type category mismatch for column '{col}': "
                    f"DF1 is {'numeric' if is_num1 else 'categorical/object'}, "
                    f"DF2 is {'numeric' if is_num2 else 'categorical/object'}."
                )

    return True


def get_missing_values_summary(df: pd.DataFrame) -> pd.Series:
    """Return counts of missing values per column for a DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.

    Returns
    -------
    pd.Series
        Missing value counts per column.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"Expected pandas DataFrame, got {type(df).__name__}")
    return df.isna().sum()


def _check_csv_header_duplicates(filepath: Path) -> None:
    """Inspect the raw first line of a CSV file for duplicate column headers."""
    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
        reader = csv.reader(f)
        try:
            header = next(reader)
        except StopIteration:
            raise ValueError(f"CSV file at '{filepath}' is empty (0 bytes).")

    if not header or not any(col.strip() for col in header):
        raise ValueError(f"CSV file at '{filepath}' is empty or contains an empty header.")

    header_trimmed = [col.strip() for col in header]
    seen = set()
    duplicates = []
    for col in header_trimmed:
        if col in seen:
            duplicates.append(col)
        seen.add(col)

    if duplicates:
        raise ValueError(
            f"CSV file at '{filepath}' contains duplicate column headers: {sorted(set(duplicates))}"
        )


def load_data(
    path: Optional[Union[str, Path]] = None,
    require_target: bool = True,
) -> pd.DataFrame:
    """Load a census CSV dataset with schema and integrity validation.

    Parameters
    ----------
    path : str or Path, optional
        Path to CSV file. Defaults to canonical train split (data/processed/train.csv).
    require_target : bool, default=True
        Whether the target column ('income') is required.

    Returns
    -------
    pd.DataFrame
        Loaded DataFrame with all rows, columns, and target preserved.

    Raises
    ------
    FileNotFoundError
        If the specified CSV file does not exist.
    ValueError
        If the file is empty, malformed, contains duplicate columns, or misses required features.
    """
    resolved_path = resolve_data_path(path)
    logger.info("Loading dataset from '%s' (require_target=%s)", resolved_path, require_target)

    if not resolved_path.is_file():
        logger.error("Data file not found: %s", resolved_path)
        raise FileNotFoundError(
            f"Data file not found at '{resolved_path}'. Expected a valid CSV file. "
            f"Run 'python scripts/07_make_grouped_split.py' or 'python scripts/02_prepare_data.py' "
            f"to generate the required datasets."
        )

    # Check raw CSV header for duplicates before pandas mangles them with .1, .2
    _check_csv_header_duplicates(resolved_path)

    try:
        df = pd.read_csv(resolved_path)
    except pd.errors.EmptyDataError:
        logger.error("CSV file is empty: %s", resolved_path)
        raise ValueError(f"CSV file at '{resolved_path}' is empty (contains no data).")
    except pd.errors.ParserError as e:
        logger.error("Malformed CSV at '%s': %s", resolved_path, e)
        raise ValueError(f"Malformed CSV at '{resolved_path}': {e}") from e

    validate_dataframe(df, require_target=require_target, dataset_name=str(resolved_path))
    logger.info(
        "Successfully loaded dataset '%s': %d rows, %d columns",
        resolved_path.name,
        len(df),
        len(df.columns),
    )

    return df


def load_train_data(
    path: Optional[Union[str, Path]] = None,
) -> pd.DataFrame:
    """Load the canonical training split (data/processed/train.csv)."""
    return load_data(path or TRAIN_DATA_PATH, require_target=True)


def load_training_data(
    path: Optional[Union[str, Path]] = None,
) -> pd.DataFrame:
    """Alias for load_train_data."""
    return load_train_data(path)


def load_validation_data(
    path: Optional[Union[str, Path]] = None,
) -> pd.DataFrame:
    """Load the canonical validation split (data/processed/validation.csv)."""
    return load_data(path or VALIDATION_DATA_PATH, require_target=True)


def load_test_data(
    path: Optional[Union[str, Path]] = None,
) -> pd.DataFrame:
    """Load the canonical held-out test split (data/processed/test.csv)."""
    return load_data(path or TEST_DATA_PATH, require_target=True)


def load_raw_data(
    path: Optional[Union[str, Path]] = None,
) -> pd.DataFrame:
    """Load the raw UCI Adult dataset (data/raw/adult.csv)."""
    return load_data(path or RAW_DATA_PATH, require_target=True)


def load_cleaned_data(
    path: Optional[Union[str, Path]] = None,
) -> pd.DataFrame:
    """Load the pre-split cleaned dataset (data/processed/adult_cleaned.csv)."""
    return load_data(path or CLEANED_DATA_PATH, require_target=True)
