"""Comprehensive test suite for data loading, validation, and integrity.

This test suite covers:
1. Legacy test compatibility (backward compatibility preservation).
2. Correct training-data and general path resolution.
3. Required columns and target presence.
4. Duplicate column detection (CSV headers and DataFrames).
5. Missing files and malformed CSV inputs.
6. Unexpected data types (non-DataFrame inputs, invalid numeric/categorical types).
7. Missing-value handling (allowed in features, rejected in target, preserved on load).
8. Train/validation/test schema consistency and leakage prevention.
9. Protection against unintended mutation of input files.
"""

import hashlib
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from incomelens.data import (
    CATEGORICAL_FEATURES,
    CLEANED_DATA_PATH,
    EXPECTED_COLUMNS,
    EXPECTED_FEATURES,
    NUMERIC_FEATURES,
    RAW_DATA_PATH,
    TARGET_COLUMN,
    TEST_DATA_PATH,
    TRAIN_DATA_PATH,
    VALIDATION_DATA_PATH,
    check_schema_consistency,
    get_missing_values_summary,
    load_cleaned_data,
    load_data,
    load_raw_data,
    load_test_data,
    load_train_data,
    load_training_data,
    load_validation_data,
    resolve_data_path,
    validate_dataframe,
)

TRAIN_PATH = ROOT / "data" / "processed" / "train.csv"


# --- Fixtures for Fast Synthetic Testing ---

@pytest.fixture
def synthetic_records() -> List[Dict[str, object]]:
    """Return small, realistic synthetic census records matching production schema."""
    return [
        {
            "age": 39,
            "workclass": "State-gov",
            "fnlwgt": 77516,
            "education": "Bachelors",
            "education-num": 13,
            "marital-status": "Never-married",
            "occupation": "Adm-clerical",
            "relationship": "Not-in-family",
            "race": "White",
            "sex": "Male",
            "capital-gain": 2174,
            "capital-loss": 0,
            "hours-per-week": 40,
            "native-country": "United-States",
            "income": "<=50K",
        },
        {
            "age": 50,
            "workclass": "Self-emp-not-inc",
            "fnlwgt": 83311,
            "education": "Bachelors",
            "education-num": 13,
            "marital-status": "Married-civ-spouse",
            "occupation": "Exec-managerial",
            "relationship": "Husband",
            "race": "White",
            "sex": "Male",
            "capital-gain": 0,
            "capital-loss": 0,
            "hours-per-week": 13,
            "native-country": "United-States",
            "income": "<=50K",
        },
        {
            "age": 38,
            "workclass": "Private",
            "fnlwgt": 215646,
            "education": "HS-grad",
            "education-num": 9,
            "marital-status": "Divorced",
            "occupation": "Handlers-cleaners",
            "relationship": "Not-in-family",
            "race": "White",
            "sex": "Male",
            "capital-gain": 0,
            "capital-loss": 0,
            "hours-per-week": 40,
            "native-country": "United-States",
            "income": "<=50K",
        },
        {
            "age": 52,
            "workclass": "Self-emp-inc",
            "fnlwgt": 287927,
            "education": "HS-grad",
            "education-num": 9,
            "marital-status": "Married-civ-spouse",
            "occupation": "Exec-managerial",
            "relationship": "Wife",
            "race": "White",
            "sex": "Female",
            "capital-gain": 15024,
            "capital-loss": 0,
            "hours-per-week": 40,
            "native-country": "United-States",
            "income": ">50K",
        },
    ]


@pytest.fixture
def synthetic_valid_df(synthetic_records) -> pd.DataFrame:
    """Return a DataFrame with valid production schema and small row count."""
    return pd.DataFrame(synthetic_records)


@pytest.fixture
def synthetic_features_df(synthetic_valid_df) -> pd.DataFrame:
    """Return a feature-only DataFrame without the target column."""
    return synthetic_valid_df.drop(columns=[TARGET_COLUMN]).copy()


@pytest.fixture
def synthetic_csv_file(tmp_path, synthetic_valid_df) -> Path:
    """Write synthetic records to a temporary CSV file."""
    csv_file = tmp_path / "synthetic_census.csv"
    synthetic_valid_df.to_csv(csv_file, index=False)
    return csv_file


# --- Legacy Test Cases (Preserved for Backward Compatibility) ---

@pytest.fixture(scope="module")
def train_data():
    assert TRAIN_PATH.exists(), f"Training data not found: {TRAIN_PATH}"
    return pd.read_csv(TRAIN_PATH)


def test_training_data_exists():
    assert TRAIN_PATH.exists()


def test_training_data_loads(train_data):
    assert not train_data.empty
    assert len(train_data.columns) > 1


def test_training_data_has_no_duplicate_columns(train_data):
    assert train_data.columns.is_unique


def test_income_target_is_present(train_data):
    assert "income" in train_data.columns
    assert train_data["income"].notna().all()


# --- 1. Path Resolution Tests ---

def test_resolve_data_path_default():
    """Verify resolve_data_path(None) defaults to canonical TRAIN_DATA_PATH."""
    resolved = resolve_data_path()
    assert resolved == TRAIN_DATA_PATH
    assert resolved.is_absolute()
    assert resolved.name == "train.csv"


def test_resolve_data_path_relative_str_and_path():
    """Verify relative paths resolve properly against project root or current working dir."""
    rel_train = Path("data") / "processed" / "train.csv"
    resolved_str = resolve_data_path(str(rel_train))
    resolved_path = resolve_data_path(rel_train)

    assert resolved_str == TRAIN_DATA_PATH
    assert resolved_path == TRAIN_DATA_PATH
    assert resolve_data_path("data/processed/validation.csv") == VALIDATION_DATA_PATH
    assert resolve_data_path("data/processed/test.csv") == TEST_DATA_PATH


def test_resolve_data_path_custom_temporary_file(tmp_path):
    """Verify custom external or temporary paths resolve accurately to their absolute path."""
    temp_csv = tmp_path / "custom_split.csv"
    temp_csv.write_text("age,income\n30,<=50K\n", encoding="utf-8")

    resolved = resolve_data_path(temp_csv)
    assert resolved == temp_csv.resolve()
    assert resolved.exists()


def test_resolve_data_path_invalid_types_raise_typeerror():
    """Verify non-path objects raise TypeError rather than failing silently."""
    for invalid in [12345, ["path/to/file"], {"path": "file"}, 3.14]:
        with pytest.raises(TypeError) as exc_info:
            resolve_data_path(invalid)
        assert "Expected path as str or Path" in str(exc_info.value)


def test_canonical_paths_exist_on_disk():
    """Verify all canonical data path constants exist as valid files."""
    assert TRAIN_DATA_PATH.is_file(), f"Missing canonical train split: {TRAIN_DATA_PATH}"
    assert VALIDATION_DATA_PATH.is_file(), f"Missing canonical validation split: {VALIDATION_DATA_PATH}"
    assert TEST_DATA_PATH.is_file(), f"Missing canonical test split: {TEST_DATA_PATH}"


# --- 2. Required Columns and Target Presence Tests ---

def test_expected_columns_specification():
    """Verify canonical column constants have expected feature counts and target."""
    assert len(EXPECTED_COLUMNS) == 15
    assert len(EXPECTED_FEATURES) == 14
    assert TARGET_COLUMN == "income"
    assert TARGET_COLUMN in EXPECTED_COLUMNS
    assert TARGET_COLUMN not in EXPECTED_FEATURES
    assert set(EXPECTED_FEATURES) == set(NUMERIC_FEATURES + CATEGORICAL_FEATURES)


def test_validate_dataframe_all_expected_columns_passes(synthetic_valid_df):
    """Verify valid synthetic dataset with all columns passes validation."""
    validated = validate_dataframe(synthetic_valid_df, require_target=True)
    assert isinstance(validated, pd.DataFrame)
    assert list(validated.columns) == EXPECTED_COLUMNS
    assert len(validated) == len(synthetic_valid_df)


def test_validate_dataframe_without_target_passes(synthetic_features_df):
    """Verify feature-only dataset passes when require_target=False."""
    validated = validate_dataframe(synthetic_features_df, require_target=False)
    assert list(validated.columns) == EXPECTED_FEATURES
    assert TARGET_COLUMN not in validated.columns


def test_validate_dataframe_missing_target_raises_error(synthetic_features_df):
    """Verify missing target column raises ValueError when require_target=True."""
    with pytest.raises(ValueError) as exc_info:
        validate_dataframe(synthetic_features_df, require_target=True)

    err = str(exc_info.value)
    assert "Missing required columns" in err
    assert TARGET_COLUMN in err


def test_validate_dataframe_missing_feature_column_raises_error(synthetic_valid_df):
    """Verify missing one or more required feature columns raises descriptive ValueError."""
    dropped_df = synthetic_valid_df.drop(columns=["age", "marital-status"])
    with pytest.raises(ValueError) as exc_info:
        validate_dataframe(dropped_df, require_target=True)

    err = str(exc_info.value)
    assert "Missing required columns" in err
    assert "age" in err
    assert "marital-status" in err


def test_production_splits_target_presence_and_classes():
    """Verify real train, validation, and test splits have valid target labels with zero nulls."""
    train_df = load_train_data()
    val_df = load_validation_data()
    test_df = load_test_data()

    for name, df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        assert TARGET_COLUMN in df.columns, f"Target missing from {name} split"
        assert df[TARGET_COLUMN].notna().all(), f"Found nulls in target for {name} split"
        unique_targets = set(df[TARGET_COLUMN].unique())
        assert unique_targets == {"<=50K", ">50K"}, f"Unexpected targets in {name}: {unique_targets}"


# --- 3. Duplicate Column Detection Tests ---

def test_csv_header_duplicate_column_detected(tmp_path):
    """Verify raw CSV header with duplicate column names raises descriptive ValueError."""
    dupe_csv = tmp_path / "dupe_header.csv"
    dupe_csv.write_text("age,workclass,age,income\n25,Private,25,<=50K\n", encoding="utf-8")

    with pytest.raises(ValueError) as exc_info:
        load_data(dupe_csv)

    err = str(exc_info.value)
    assert "duplicate column header" in err.lower()
    assert "age" in err


def test_csv_header_whitespace_duplicate_column_detected(tmp_path):
    """Verify CSV header duplicate with surrounding whitespace is caught."""
    dupe_csv = tmp_path / "whitespace_dupe.csv"
    dupe_csv.write_text("age, age ,workclass,income\n25,25,Private,<=50K\n", encoding="utf-8")

    with pytest.raises(ValueError) as exc_info:
        load_data(dupe_csv)

    err = str(exc_info.value)
    assert "duplicate column header" in err.lower()
    assert "age" in err


def test_validate_dataframe_duplicate_columns_raises_error(synthetic_valid_df):
    """Verify in-memory DataFrame with duplicate column names raises ValueError."""
    # Concatenate extra 'age' column to create duplicate
    dupe_df = pd.concat([synthetic_valid_df, synthetic_valid_df[["age"]]], axis=1)
    assert not dupe_df.columns.is_unique

    with pytest.raises(ValueError) as exc_info:
        validate_dataframe(dupe_df)

    err = str(exc_info.value)
    assert "duplicate column names detected" in err.lower()
    assert "age" in err


# --- 4. Missing Files and Malformed CSV Input Tests ---

def test_load_data_missing_file_raises_actionable_error():
    """Verify non-existent file path raises FileNotFoundError with helpful remediation."""
    non_existent = ROOT / "data" / "processed" / "does_not_exist_xyz.csv"
    with pytest.raises(FileNotFoundError) as exc_info:
        load_data(non_existent)

    err = str(exc_info.value)
    assert "Data file not found" in err
    assert "scripts/07_make_grouped_split.py" in err or "scripts/02_prepare_data.py" in err


def test_load_data_empty_file_raises_error(tmp_path):
    """Verify 0-byte CSV file raises ValueError."""
    empty_file = tmp_path / "empty_dataset.csv"
    empty_file.write_text("", encoding="utf-8")

    with pytest.raises(ValueError) as exc_info:
        load_data(empty_file)

    assert "empty" in str(exc_info.value).lower()


def test_load_data_whitespace_only_file_raises_error(tmp_path):
    """Verify file containing only whitespace or blank newlines raises ValueError."""
    blank_file = tmp_path / "blank.csv"
    blank_file.write_text("   \n\n\t  \n", encoding="utf-8")

    with pytest.raises(ValueError) as exc_info:
        load_data(blank_file)

    assert "empty" in str(exc_info.value).lower()


def test_load_data_malformed_csv_jagged_rows_raises_error(tmp_path):
    """Verify jagged rows with differing field counts raise ValueError."""
    jagged_file = tmp_path / "jagged.csv"
    jagged_file.write_text("age,workclass,income\n25,Private,<=50K\n30,Private,ExtraField,<=50K\n", encoding="utf-8")

    with pytest.raises(ValueError) as exc_info:
        load_data(jagged_file)

    assert "malformed" in str(exc_info.value).lower() or "error" in str(exc_info.value).lower()


def test_load_data_malformed_csv_unclosed_quotes_raises_error(tmp_path):
    """Verify CSV with unclosed quotes or syntax errors raises ValueError."""
    bad_quotes_file = tmp_path / "bad_quotes.csv"
    bad_quotes_file.write_text('age,workclass,income\n25,"Private,<=50K\n35,State-gov,<=50K\n', encoding="utf-8")

    with pytest.raises(ValueError) as exc_info:
        load_data(bad_quotes_file)

    assert "malformed" in str(exc_info.value).lower() or "error" in str(exc_info.value).lower()


# --- 5. Unexpected Data Types Tests ---

def test_validate_dataframe_non_dataframe_types_raise_typeerror():
    """Verify non-DataFrame input types raise TypeError."""
    for invalid in [None, "dataset", [1, 2, 3], {"age": [25]}, 42]:
        with pytest.raises(TypeError) as exc_info:
            validate_dataframe(invalid)
        assert "Expected pandas DataFrame" in str(exc_info.value)


def test_validate_dataframe_non_numeric_in_numeric_feature_raises_typeerror(synthetic_valid_df):
    """Verify passing string words in a numeric column raises TypeError."""
    bad_df = synthetic_valid_df.copy()
    bad_df["age"] = ["twenty", "thirty", "forty", "fifty"]

    with pytest.raises(TypeError) as exc_info:
        validate_dataframe(bad_df)

    err = str(exc_info.value)
    assert "Column 'age' has unexpected non-numeric data type" in err


def test_validate_dataframe_nested_collection_in_numeric_feature_raises_typeerror(synthetic_valid_df):
    """Verify nested objects/dicts in numeric columns raise TypeError."""
    bad_df = synthetic_valid_df.copy()
    bad_df["hours-per-week"] = [{"hours": 40}, {"hours": 35}, {"hours": 20}, {"hours": 50}]

    with pytest.raises(TypeError) as exc_info:
        validate_dataframe(bad_df)

    assert "hours-per-week" in str(exc_info.value)


def test_validate_dataframe_nested_collection_in_categorical_feature_raises_typeerror(synthetic_valid_df):
    """Verify nested collections in categorical columns raise TypeError."""
    bad_df = synthetic_valid_df.copy()
    bad_df["workclass"] = [{"class": "Private"}, {"class": "State-gov"}, {"class": "Self-emp"}, {"class": "Private"}]

    with pytest.raises(TypeError) as exc_info:
        validate_dataframe(bad_df)

    err = str(exc_info.value)
    assert "workclass" in err
    assert "unexpected nested collection" in err.lower()


def test_load_data_invalid_path_argument_type_raises_typeerror():
    """Verify passing non-path arguments to load_data raises TypeError."""
    with pytest.raises(TypeError) as exc_info:
        load_data(12345)
    assert "Expected path as str or Path" in str(exc_info.value)


# --- 6. Missing-Value Handling Tests ---

def test_missing_values_allowed_in_categorical_features(synthetic_valid_df):
    """Verify missing values in categorical predictors are preserved and accepted."""
    df_with_nulls = synthetic_valid_df.copy()
    df_with_nulls.loc[0, "workclass"] = np.nan
    df_with_nulls.loc[1, "occupation"] = None
    df_with_nulls.loc[2, "native-country"] = np.nan

    validated = validate_dataframe(df_with_nulls, require_target=True)
    assert len(validated) == len(synthetic_valid_df)
    assert pd.isna(validated.loc[0, "workclass"])
    assert pd.isna(validated.loc[1, "occupation"])
    assert pd.isna(validated.loc[2, "native-country"])


def test_missing_values_allowed_in_numeric_features(synthetic_valid_df):
    """Verify missing values in numeric predictors are preserved for downstream imputer."""
    df_with_nulls = synthetic_valid_df.copy()
    df_with_nulls.loc[0, "age"] = np.nan
    df_with_nulls.loc[1, "hours-per-week"] = np.nan

    validated = validate_dataframe(df_with_nulls, require_target=True)
    assert len(validated) == len(synthetic_valid_df)
    assert pd.isna(validated.loc[0, "age"])
    assert pd.isna(validated.loc[1, "hours-per-week"])


def test_target_missing_values_rejected_when_required(synthetic_valid_df):
    """Verify target column missing values raise ValueError when require_target=True."""
    df_with_null_target = synthetic_valid_df.copy()
    df_with_null_target.loc[0, TARGET_COLUMN] = np.nan

    with pytest.raises(ValueError) as exc_info:
        validate_dataframe(df_with_null_target, require_target=True)

    err = str(exc_info.value)
    assert f"Target column '{TARGET_COLUMN}' contains 1 missing value(s)" in err


def test_load_data_preserves_missing_values_from_csv(tmp_path, synthetic_valid_df):
    """Verify loading a CSV containing missing fields retains them without dropping rows."""
    df_with_nulls = synthetic_valid_df.copy()
    df_with_nulls.loc[0, "workclass"] = np.nan
    df_with_nulls.loc[1, "native-country"] = np.nan

    csv_path = tmp_path / "with_nulls.csv"
    df_with_nulls.to_csv(csv_path, index=False)

    loaded = load_data(csv_path, require_target=True)
    assert len(loaded) == len(synthetic_valid_df)
    assert loaded["workclass"].isna().sum() == 1
    assert loaded["native-country"].isna().sum() == 1


def test_get_missing_values_summary_utility(synthetic_valid_df):
    """Verify get_missing_values_summary returns accurate null counts per column."""
    test_df = synthetic_valid_df.copy()
    test_df.loc[0, "workclass"] = np.nan
    test_df.loc[1, "workclass"] = np.nan
    test_df.loc[0, "occupation"] = np.nan

    summary = get_missing_values_summary(test_df)
    assert isinstance(summary, pd.Series)
    assert summary["workclass"] == 2
    assert summary["occupation"] == 1
    assert summary["age"] == 0


def test_get_missing_values_summary_non_dataframe_raises_typeerror():
    """Verify get_missing_values_summary raises TypeError for non-DataFrame input."""
    with pytest.raises(TypeError) as exc_info:
        get_missing_values_summary("not_a_df")
    assert "Expected pandas DataFrame" in str(exc_info.value)


# --- 7. Train/Validation/Test Schema Consistency Tests ---

def test_train_validation_test_identical_columns():
    """Verify canonical train, validation, and test splits share identical columns and order."""
    train_df = load_train_data()
    val_df = load_validation_data()
    test_df = load_test_data()

    assert list(train_df.columns) == EXPECTED_COLUMNS
    assert list(val_df.columns) == EXPECTED_COLUMNS
    assert list(test_df.columns) == EXPECTED_COLUMNS
    assert list(train_df.columns) == list(val_df.columns) == list(test_df.columns)


def test_train_validation_test_compatible_dtypes():
    """Verify canonical splits have compatible data types across numeric and categorical features."""
    train_df = load_train_data()
    val_df = load_validation_data()
    test_df = load_test_data()

    assert check_schema_consistency(train_df, val_df, check_dtypes=True)
    assert check_schema_consistency(train_df, test_df, check_dtypes=True)

    for col in NUMERIC_FEATURES:
        assert pd.api.types.is_numeric_dtype(train_df[col])
        assert pd.api.types.is_numeric_dtype(val_df[col])
        assert pd.api.types.is_numeric_dtype(test_df[col])

    for col in CATEGORICAL_FEATURES:
        assert not pd.api.types.is_numeric_dtype(train_df[col])
        assert not pd.api.types.is_numeric_dtype(val_df[col])
        assert not pd.api.types.is_numeric_dtype(test_df[col])


def test_train_validation_test_target_distribution_and_labels():
    """Verify class balance is roughly consistent (~20-30% high-income) across all splits."""
    train_df = load_train_data()
    val_df = load_validation_data()
    test_df = load_test_data()

    for name, df in [("Train", train_df), ("Val", val_df), ("Test", test_df)]:
        pos_rate = (df[TARGET_COLUMN] == ">50K").mean()
        assert 0.20 <= pos_rate <= 0.30, f"{name} split positive class rate {pos_rate:.4f} outside [0.20, 0.30]"


def test_check_schema_consistency_utility_synthetic(synthetic_valid_df):
    """Verify check_schema_consistency detects column omissions and order mismatches."""
    df1 = synthetic_valid_df.copy()
    df2 = synthetic_valid_df.copy()

    # Identical schemas pass
    assert check_schema_consistency(df1, df2, check_dtypes=True) is True

    # Column order mismatch raises ValueError
    cols_reordered = list(df1.columns)
    cols_reordered[0], cols_reordered[1] = cols_reordered[1], cols_reordered[0]
    df_reordered = df2[cols_reordered]
    with pytest.raises(ValueError) as exc_info:
        check_schema_consistency(df1, df_reordered)
    assert "Column schema mismatch" in str(exc_info.value)

    # Missing column raises ValueError
    df_missing = df2.drop(columns=["age"])
    with pytest.raises(ValueError) as exc_info:
        check_schema_consistency(df1, df_missing)
    assert "Column schema mismatch" in str(exc_info.value)

    # Data type mismatch raises ValueError
    df_bad_type = df2.copy()
    df_bad_type["age"] = df_bad_type["age"].astype(str)
    with pytest.raises(ValueError) as exc_info:
        check_schema_consistency(df1, df_bad_type, check_dtypes=True)
    assert "Data type category mismatch" in str(exc_info.value)


def test_split_group_disjointness():
    """Verify train, validation, and test splits share zero feature groups (prevent leakage)."""
    train_df = load_train_data()
    val_df = load_validation_data()
    test_df = load_test_data()

    feature_cols = [c for c in train_df.columns if c != TARGET_COLUMN]

    train_groups = set(
        pd.util.hash_pandas_object(
            train_df[feature_cols].fillna("<MISSING>").astype(str), index=False
        )
    )
    val_groups = set(
        pd.util.hash_pandas_object(
            val_df[feature_cols].fillna("<MISSING>").astype(str), index=False
        )
    )
    test_groups = set(
        pd.util.hash_pandas_object(
            test_df[feature_cols].fillna("<MISSING>").astype(str), index=False
        )
    )

    assert train_groups.isdisjoint(val_groups), "Train and validation splits share predictor groups"
    assert train_groups.isdisjoint(test_groups), "Train and test splits share predictor groups"
    assert val_groups.isdisjoint(test_groups), "Validation and test splits share predictor groups"


# --- 8. No Unintended Mutation of Input Files Tests ---

def test_load_data_does_not_mutate_csv_file(synthetic_csv_file):
    """Verify load_data reads data without altering file content or disk hash."""
    def compute_sha256(filepath: Path) -> str:
        h = hashlib.sha256()
        with open(filepath, "rb") as f:
            while chunk := f.read(65536):
                h.update(chunk)
        return h.hexdigest()

    original_hash = compute_sha256(synthetic_csv_file)
    original_size = synthetic_csv_file.stat().st_size
    original_mtime = synthetic_csv_file.stat().st_mtime_ns

    # Load dataset
    df = load_data(synthetic_csv_file, require_target=True)

    # Perform in-memory mutations on returned DataFrame
    df.loc[0, "age"] = 999
    df["new_column"] = "mutation"
    df.drop(columns=[TARGET_COLUMN], inplace=True)

    # Verify file on disk is completely untouched
    post_hash = compute_sha256(synthetic_csv_file)
    post_size = synthetic_csv_file.stat().st_size
    post_mtime = synthetic_csv_file.stat().st_mtime_ns

    assert post_hash == original_hash, "CSV file hash on disk was modified by load_data or DataFrame mutation!"
    assert post_size == original_size, "CSV file size changed on disk!"
    assert post_mtime == original_mtime, "CSV file modification time changed on disk!"


def test_load_train_data_does_not_mutate_canonical_file():
    """Verify load_train_data preserves canonical train.csv file hash and content."""
    def compute_sha256(filepath: Path) -> str:
        h = hashlib.sha256()
        with open(filepath, "rb") as f:
            while chunk := f.read(65536):
                h.update(chunk)
        return h.hexdigest()

    original_hash = compute_sha256(TRAIN_DATA_PATH)
    original_size = TRAIN_DATA_PATH.stat().st_size

    # Load canonical training split
    train_df = load_train_data()
    assert len(train_df) == 29301

    # In-memory mutation
    train_df.iloc[0, 0] = -1

    post_hash = compute_sha256(TRAIN_DATA_PATH)
    post_size = TRAIN_DATA_PATH.stat().st_size

    assert post_hash == original_hash
    assert post_size == original_size
