"""Tests for incomelens.preprocessing module and pipeline compatibility."""

import sys
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from incomelens.data import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    TARGET_COLUMN,
    TRAIN_DATA_PATH,
    VALIDATION_DATA_PATH,
    TEST_DATA_PATH,
    load_test_data,
    load_train_data,
    load_validation_data,
)
from incomelens.preprocessing import (
    build_preprocessor,
    extract_features,
    get_feature_names,
    split_features_target,
)


@pytest.fixture(scope="module")
def train_features():
    if not TRAIN_DATA_PATH.is_file():
        pytest.skip("Training split data/processed/train.csv not found on disk.")
    train_df = load_train_data()
    X, y = split_features_target(train_df)
    return X, y


@pytest.fixture(scope="module")
def fitted_preprocessor(train_features):
    """Pre-fitted preprocessor strictly fitted on training data only."""
    X_train, _ = train_features
    preprocessor = build_preprocessor()
    preprocessor.fit(X_train)
    return preprocessor


def test_build_preprocessor_structure():
    """Verify ColumnTransformer architecture and sub-pipelines."""
    prep = build_preprocessor(sparse_output=True)
    assert prep.remainder == "drop"
    transformers = {name: (pipe, cols) for name, pipe, cols in prep.transformers}

    assert "numeric" in transformers
    num_pipe, num_cols = transformers["numeric"]
    assert num_cols == NUMERIC_FEATURES
    assert "imputer" in num_pipe.named_steps
    assert "scaler" in num_pipe.named_steps
    assert num_pipe.named_steps["imputer"].strategy == "median"

    assert "categorical" in transformers
    cat_pipe, cat_cols = transformers["categorical"]
    assert cat_cols == CATEGORICAL_FEATURES
    assert "imputer" in cat_pipe.named_steps
    assert "onehot" in cat_pipe.named_steps
    assert cat_pipe.named_steps["imputer"].strategy == "most_frequent"
    assert cat_pipe.named_steps["onehot"].handle_unknown == "ignore"
    assert cat_pipe.named_steps["onehot"].sparse_output is True


def test_train_only_fitting_and_transformation(fitted_preprocessor):
    """Verify preprocessor fitted on train transforms validation and test without refitting."""
    if not VALIDATION_DATA_PATH.is_file() or not TEST_DATA_PATH.is_file():
        pytest.skip("Validation or test split not found on disk.")
    val_df = load_validation_data()
    test_df = load_test_data()

    X_val = extract_features(val_df)
    X_test = extract_features(test_df)

    # Transform without fitting
    val_trans = fitted_preprocessor.transform(X_val)
    test_trans = fitted_preprocessor.transform(X_test)

    assert val_trans.shape == (len(X_val), 105)
    assert test_trans.shape == (len(X_test), 105)


def test_missing_value_imputation(fitted_preprocessor):
    """Verify missing values in both numeric and categorical columns are cleanly imputed."""
    sample = pd.DataFrame([{
        "age": np.nan,
        "fnlwgt": np.nan,
        "education-num": np.nan,
        "capital-gain": np.nan,
        "capital-loss": np.nan,
        "hours-per-week": np.nan,
        "workclass": np.nan,
        "education": np.nan,
        "marital-status": np.nan,
        "occupation": np.nan,
        "relationship": np.nan,
        "race": np.nan,
        "sex": np.nan,
        "native-country": np.nan,
    }])

    trans = fitted_preprocessor.transform(sample)
    if hasattr(trans, "toarray"):
        trans_dense = trans.toarray()
    else:
        trans_dense = np.asarray(trans)

    assert trans_dense.shape == (1, 105)
    assert not np.isnan(trans_dense).any()
    assert np.isfinite(trans_dense).all()


def test_unseen_categorical_levels_handled_gracefully(fitted_preprocessor):
    """Verify unseen/novel categorical categories do not raise error and yield zero-vectors."""
    sample = pd.DataFrame([{
        "age": 35,
        "fnlwgt": 180000,
        "education-num": 13,
        "capital-gain": 0,
        "capital-loss": 0,
        "hours-per-week": 40,
        "workclass": "UnknownEnterprise",
        "education": "PostPhDSpecialist",
        "marital-status": "Complicated",
        "occupation": "Cyberneticist",
        "relationship": "Roommate",
        "race": "Martian",
        "sex": "NonBinary",
        "native-country": "SolarSystem",
    }])

    trans = fitted_preprocessor.transform(sample)
    if hasattr(trans, "toarray"):
        trans_dense = trans.toarray()
    else:
        trans_dense = np.asarray(trans)

    assert trans_dense.shape == (1, 105)
    # The first 6 features are numeric, the remaining 99 are OHE dummy features.
    # Unseen categories should cause all categorical dummy columns to be 0.
    cat_dummies = trans_dense[0, 6:]
    assert (cat_dummies == 0.0).all()


def test_feature_names_preservation(fitted_preprocessor):
    """Verify feature name extraction produces 105 names matching production model."""
    feature_names = get_feature_names(fitted_preprocessor)
    assert len(feature_names) == 105

    # Check numeric prefix and order
    expected_numeric = [f"numeric__{col}" for col in NUMERIC_FEATURES]
    assert feature_names[:6] == expected_numeric

    # Check categorical prefix
    for name in feature_names[6:]:
        assert name.startswith("categorical__")

    # Check exact match against production model preprocessor
    prod_model_path = ROOT / "models" / "random_forest_final.joblib"
    if prod_model_path.exists():
        model = joblib.load(prod_model_path)
        prod_prep = model.named_steps["preprocessor"]
        assert feature_names == get_feature_names(prod_prep)


def test_split_features_target_clean_and_raw():
    """Verify split_features_target handles clean, raw, and numeric labels."""
    # 1. Clean labels
    df_clean = pd.DataFrame({
        **{col: [30, 40] for col in NUMERIC_FEATURES},
        **{col: ["Private", "Private"] for col in CATEGORICAL_FEATURES},
        TARGET_COLUMN: ["<=50K", ">50K"],
    })
    X1, y1 = split_features_target(df_clean)
    assert len(X1) == 2
    assert list(y1) == [0, 1]

    # 2. Raw labels with trailing periods (from UCI test split format)
    df_raw = pd.DataFrame({
        **{col: [30, 40] for col in NUMERIC_FEATURES},
        **{col: ["Private", "Private"] for col in CATEGORICAL_FEATURES},
        TARGET_COLUMN: ["<=50K.", ">50K."],
    })
    X2, y2 = split_features_target(df_raw)
    assert list(y2) == [0, 1]

    # 3. Numeric labels
    df_num = pd.DataFrame({
        **{col: [30, 40] for col in NUMERIC_FEATURES},
        **{col: ["Private", "Private"] for col in CATEGORICAL_FEATURES},
        TARGET_COLUMN: [0, 1],
    })
    X3, y3 = split_features_target(df_num)
    assert list(y3) == [0, 1]


def test_split_features_target_validation_errors():
    """Verify split_features_target raises descriptive errors on invalid inputs."""
    # Missing target
    df_no_target = pd.DataFrame({
        **{col: [30] for col in NUMERIC_FEATURES},
        **{col: ["Private"] for col in CATEGORICAL_FEATURES},
    })
    with pytest.raises(ValueError) as exc_info:
        split_features_target(df_no_target)
    assert "Target column 'income' not found" in str(exc_info.value)

    # Invalid target label
    df_bad_target = pd.DataFrame({
        **{col: [30] for col in NUMERIC_FEATURES},
        **{col: ["Private"] for col in CATEGORICAL_FEATURES},
        TARGET_COLUMN: ["UnderMillion"],
    })
    with pytest.raises(ValueError) as exc_info:
        split_features_target(df_bad_target)
    assert "Unexpected or missing income labels" in str(exc_info.value)


def test_extract_features_schema_mismatch():
    """Verify extract_features enforces required schema."""
    # Extra column is dropped cleanly
    df_extra = pd.DataFrame({
        **{col: [30] for col in NUMERIC_FEATURES},
        **{col: ["Private"] for col in CATEGORICAL_FEATURES},
        "extra_tracking_id": [12345],
        TARGET_COLUMN: ["<=50K"],
    })
    X = extract_features(df_extra)
    assert "extra_tracking_id" not in X.columns
    assert "income" not in X.columns
    assert list(X.columns) == NUMERIC_FEATURES + CATEGORICAL_FEATURES

    # Missing column raises ValueError
    df_incomplete = df_extra.drop(columns=["age"])
    with pytest.raises(ValueError) as exc_info:
        extract_features(df_incomplete)
    assert "missing required feature columns" in str(exc_info.value)
    assert "age" in str(exc_info.value)


def test_production_model_pipeline_compatibility():
    """Verify the production model pipeline works seamlessly with raw features."""
    prod_model_path = ROOT / "models" / "random_forest_final.joblib"
    if not prod_model_path.exists() or not TEST_DATA_PATH.is_file():
        pytest.skip("Production model or test data not found on disk.")

    model = joblib.load(prod_model_path)
    test_df = load_test_data()
    X_test, y_test = split_features_target(test_df)

    sample = X_test.head(10)
    preds = model.predict(sample)
    probs = model.predict_proba(sample)

    assert len(preds) == 10
    assert probs.shape == (10, 2)
    assert set(preds).issubset({0, 1})
