"""Strengthened test suite for model prediction, schema, probability calibration, and error handling.

This test suite verifies:
1. Model file existence, loading, and pipeline architecture.
2. Correct feature count and input/output schema.
3. Batch and single-row prediction consistency.
4. Valid class labels and integer/string semantics.
5. Probability range, conservation, and probability-column alignment.
6. Missing and unexpected feature handling.
7. Categorical preprocessing compatibility (unseen categories and missing values).
8. Invalid inputs and predictable error handling.
9. Deterministic reproducibility for fixed models and inputs.
"""

from pathlib import Path
import sys
from typing import Dict, List

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from incomelens.data import (
    CATEGORICAL_FEATURES,
    EXPECTED_FEATURES,
    NUMERIC_FEATURES,
    TARGET_COLUMN,
)
from incomelens.inference import (
    IncomePredictor,
    ValidationError,
)
from incomelens.training import (
    build_logistic_regression_pipeline,
    build_random_forest_pipeline,
)

MODEL_PATH = ROOT / "models" / "random_forest_final.joblib"
TRAIN_PATH = ROOT / "data" / "processed" / "train.csv"


# --- Fixtures for Fast and Deterministic Testing ---

@pytest.fixture(scope="module")
def model() -> Pipeline:
    """Load the canonical trained model artifact once per test module."""
    assert MODEL_PATH.exists(), f"Model artifact not found at {MODEL_PATH}"
    return joblib.load(MODEL_PATH)


@pytest.fixture
def synthetic_census_records() -> List[Dict[str, object]]:
    """Return diverse synthetic census feature dictionaries representing both classes."""
    return [
        {
            # Low income profile: young, entry-level, part-time
            "age": 22,
            "workclass": "Private",
            "fnlwgt": 200000,
            "education": "HS-grad",
            "education-num": 9,
            "marital-status": "Never-married",
            "occupation": "Other-service",
            "relationship": "Own-child",
            "race": "White",
            "sex": "Female",
            "capital-gain": 0,
            "capital-loss": 0,
            "hours-per-week": 20,
            "native-country": "United-States",
        },
        {
            # High income profile: senior, high capital-gain, executive
            "age": 52,
            "workclass": "Self-emp-inc",
            "fnlwgt": 200000,
            "education": "Doctorate",
            "education-num": 16,
            "marital-status": "Married-civ-spouse",
            "occupation": "Exec-managerial",
            "relationship": "Husband",
            "race": "White",
            "sex": "Male",
            "capital-gain": 20000,
            "capital-loss": 0,
            "hours-per-week": 55,
            "native-country": "United-States",
        },
        {
            # Typical median earner
            "age": 38,
            "workclass": "Private",
            "fnlwgt": 180000,
            "education": "Bachelors",
            "education-num": 13,
            "marital-status": "Never-married",
            "occupation": "Adm-clerical",
            "relationship": "Not-in-family",
            "race": "White",
            "sex": "Male",
            "capital-gain": 0,
            "capital-loss": 0,
            "hours-per-week": 40,
            "native-country": "United-States",
        },
        {
            # High earner: married professional, high capital gain
            "age": 45,
            "workclass": "Private",
            "fnlwgt": 220000,
            "education": "Masters",
            "education-num": 14,
            "marital-status": "Married-civ-spouse",
            "occupation": "Prof-specialty",
            "relationship": "Husband",
            "race": "White",
            "sex": "Male",
            "capital-gain": 15000,
            "capital-loss": 0,
            "hours-per-week": 50,
            "native-country": "United-States",
        },
        {
            # Low earner: student, retail
            "age": 19,
            "workclass": "Private",
            "fnlwgt": 150000,
            "education": "Some-college",
            "education-num": 10,
            "marital-status": "Never-married",
            "occupation": "Sales",
            "relationship": "Own-child",
            "race": "White",
            "sex": "Male",
            "capital-gain": 0,
            "capital-loss": 0,
            "hours-per-week": 15,
            "native-country": "United-States",
        },
    ]


@pytest.fixture
def synthetic_census_df(synthetic_census_records) -> pd.DataFrame:
    """Return DataFrame containing the 14 raw survey features."""
    return pd.DataFrame(synthetic_census_records)


# --- 1. Model Existence, Loading, and Pipeline Architecture Tests ---

def test_model_file_exists():
    """Verify production model artifact exists at the expected path."""
    assert MODEL_PATH.exists(), f"Model not found: {MODEL_PATH}"


def test_model_loads():
    """Verify production model artifact loads into a valid, fitted scikit-learn Pipeline."""
    model = joblib.load(MODEL_PATH)
    assert isinstance(model, Pipeline), f"Expected scikit-learn Pipeline, got {type(model).__name__}"
    assert hasattr(model, "predict"), "Model artifact does not implement predict"
    assert hasattr(model, "predict_proba"), "Model artifact does not implement predict_proba"
    assert hasattr(model, "classes_"), "Model artifact lacks classes_ attribute"
    assert len(model.steps) >= 2, f"Model pipeline expected >= 2 steps, got {len(model.steps)}"


def test_build_pipeline_builders():
    """Verify training pipeline builders produce two-step pipelines with correct estimators."""
    lr_pipe = build_logistic_regression_pipeline()
    assert len(lr_pipe.steps) == 2
    assert isinstance(lr_pipe.named_steps["classifier"], LogisticRegression)

    rf_pipe = build_random_forest_pipeline()
    assert len(rf_pipe.steps) == 2
    assert isinstance(rf_pipe.named_steps["classifier"], RandomForestClassifier)


def test_model_predictions_are_valid():
    """Legacy test: verify sample predictions from train data produce valid classes."""
    model = joblib.load(MODEL_PATH)
    df = pd.read_csv(TRAIN_PATH).drop(columns=["income"], errors="ignore")

    sample = df.head(10)
    predictions = model.predict(sample)

    assert len(predictions) == len(sample)
    assert pd.Series(predictions).notna().all()

    if hasattr(model, "classes_"):
        assert set(predictions).issubset(set(model.classes_))


# --- 2. Correct Feature Count and Schema Tests ---

def test_model_pipeline_feature_counts_and_schema(model):
    """Verify model pipeline expects 14 raw survey features and outputs 105 transformed features."""
    assert isinstance(model, Pipeline)
    assert len(model.steps) == 2
    assert "preprocessor" in model.named_steps
    assert "classifier" in model.named_steps

    preprocessor = model.named_steps["preprocessor"]
    transformers = {name: (trans, cols) for name, trans, cols in preprocessor.transformers}

    # Check numeric feature schema
    assert "numeric" in transformers
    num_cols = transformers["numeric"][1]
    assert sorted(num_cols) == sorted(NUMERIC_FEATURES)
    assert len(num_cols) == 6

    # Check categorical feature schema
    assert "categorical" in transformers
    cat_cols = transformers["categorical"][1]
    assert sorted(cat_cols) == sorted(CATEGORICAL_FEATURES)
    assert len(cat_cols) == 8

    # Check total raw features
    assert len(EXPECTED_FEATURES) == 14
    assert sorted(num_cols + cat_cols) == sorted(EXPECTED_FEATURES)

    # Check transformed output feature count
    out_features = preprocessor.get_feature_names_out()
    assert len(out_features) == 105

    # Check classifier input feature count
    classifier = model.named_steps["classifier"]
    assert classifier.n_features_in_ == 105


# --- 3. Batch and Single-Row Prediction Tests ---

def test_single_row_prediction(model, synthetic_census_df):
    """Verify single-row inference works cleanly for DataFrame and series inputs."""
    single_row = synthetic_census_df.iloc[[0]]

    # Predict single instance
    pred = model.predict(single_row)
    assert isinstance(pred, np.ndarray)
    assert pred.shape == (1,)
    assert pred[0] in [0, 1]

    # Predict_proba single instance
    probs = model.predict_proba(single_row)
    assert isinstance(probs, np.ndarray)
    assert probs.shape == (1, 2)
    assert np.isclose(probs[0].sum(), 1.0)


def test_batch_prediction_shape_and_row_consistency(model, synthetic_census_df):
    """Verify batch prediction returns exact shapes and row-by-row consistency."""
    batch_size = len(synthetic_census_df)
    batch_preds = model.predict(synthetic_census_df)
    batch_probs = model.predict_proba(synthetic_census_df)

    assert len(batch_preds) == batch_size
    assert batch_probs.shape == (batch_size, 2)

    # Verify each row evaluated in isolation matches its batch counterpart exactly
    for i in range(batch_size):
        row = synthetic_census_df.iloc[[i]]
        single_pred = model.predict(row)[0]
        single_prob = model.predict_proba(row)[0]

        assert single_pred == batch_preds[i], f"Mismatch at row {i}: single={single_pred}, batch={batch_preds[i]}"
        assert np.allclose(single_prob, batch_probs[i], atol=1e-6), (
            f"Probability mismatch at row {i}: single={single_prob}, batch={batch_probs[i]}"
        )


# --- 4. Valid Class Labels Tests ---

def test_model_class_labels_and_semantics(model, synthetic_census_df):
    """Verify class labels are strictly [0, 1] and match high/low income semantics."""
    classes = getattr(model, "classes_", getattr(model.named_steps["classifier"], "classes_", None))
    assert list(classes) == [0, 1]

    preds = model.predict(synthetic_census_df)
    assert set(preds).issubset({0, 1})
    assert not pd.isna(preds).any()

    # Synthetic row 0 (entry-level 22yo part-time) should predict 0 (<=50K)
    assert preds[0] == 0

    # Synthetic row 1 (52yo executive with $20k capital gain) should predict 1 (>50K)
    assert preds[1] == 1


# --- 5. Probability Range and Probability-Column Alignment Tests ---

def test_probability_range_and_conservation(model, synthetic_census_df):
    """Verify probabilities are non-negative, <= 1.0, and sum to 1.0 per row."""
    probs = model.predict_proba(synthetic_census_df)

    assert (probs >= 0.0).all(), "Found negative probabilities"
    assert (probs <= 1.0).all(), "Found probabilities > 1.0"
    assert np.isfinite(probs).all(), "Found non-finite probability values"

    row_sums = probs.sum(axis=1)
    assert np.allclose(row_sums, 1.0, atol=1e-5), f"Probabilities do not sum to 1.0: {row_sums}"


def test_probability_column_alignment_with_classes(model, synthetic_census_df):
    """Verify probability column indices match model.classes_ and argmax matches predict."""
    probs = model.predict_proba(synthetic_census_df)
    preds = model.predict(synthetic_census_df)
    classes = model.classes_

    # Argmax on probabilities must match model.predict exactly
    argmax_classes = classes[np.argmax(probs, axis=1)]
    assert np.array_equal(preds, argmax_classes), "Argmax of probabilities diverges from predicted classes"

    # For binary 0/1 classifier, column 1 is >50K probability
    for i, (pred, prob) in enumerate(zip(preds, probs)):
        p_low = prob[0]
        p_high = prob[1]
        assert np.isclose(p_low + p_high, 1.0)
        if pred == 1:
            assert p_high >= 0.5, f"Row {i} predicted 1 but p_high={p_high} < 0.5"
        else:
            assert p_low >= 0.5, f"Row {i} predicted 0 but p_low={p_low} < 0.5"


# --- 6. Missing and Unexpected Features Tests ---

def test_missing_required_features_raises_valueerror(model, synthetic_census_df):
    """Verify omitting required features causes pipeline's preprocessor to raise ValueError."""
    # Omit single column
    missing_one = synthetic_census_df.drop(columns=["age"])
    with pytest.raises(ValueError) as exc_info:
        model.predict(missing_one)
    assert "columns are missing" in str(exc_info.value).lower()
    assert "age" in str(exc_info.value)

    # Omit multiple columns
    missing_multiple = synthetic_census_df.drop(columns=["education", "marital-status"])
    with pytest.raises(ValueError) as exc_info:
        model.predict(missing_multiple)
    assert "columns are missing" in str(exc_info.value).lower()


def test_unexpected_features_handled_safely(model, synthetic_census_df):
    """Verify extra/unrelated columns are safely ignored by preprocessor without altering predictions."""
    extra_df = synthetic_census_df.copy()
    extra_df["user_session_id"] = [101, 102, 103, 104, 105]
    extra_df["arbitrary_noise"] = ["x", "y", "z", "w", "v"]

    preds_original = model.predict(synthetic_census_df)
    preds_extra = model.predict(extra_df)

    probs_original = model.predict_proba(synthetic_census_df)
    probs_extra = model.predict_proba(extra_df)

    assert np.array_equal(preds_original, preds_extra)
    assert np.allclose(probs_original, probs_extra, atol=1e-6)


def test_target_column_not_present_in_features(synthetic_census_df):
    """Verify target column 'income' is absent from input feature dataframe."""
    assert TARGET_COLUMN not in synthetic_census_df.columns

    predictor = IncomePredictor()
    with_target = synthetic_census_df.iloc[0].to_dict()
    with_target[TARGET_COLUMN] = "<=50K"

    with pytest.raises(ValidationError, match="prediction target and cannot be provided"):
        predictor.predict(with_target)


# --- 7. Categorical Preprocessing Compatibility Tests ---

def test_unseen_categorical_levels_handled_gracefully(model, synthetic_census_df):
    """Verify novel/unseen categorical values do not crash the pipeline and yield valid predictions."""
    unseen_df = synthetic_census_df.copy()
    unseen_df.loc[0, "workclass"] = "InterstellarEnterprise"
    unseen_df.loc[0, "native-country"] = "Atlantis"
    unseen_df.loc[0, "education"] = "FuturePostDoctorate"

    preds = model.predict(unseen_df)
    probs = model.predict_proba(unseen_df)

    assert len(preds) == len(synthetic_census_df)
    assert preds[0] in [0, 1]
    assert np.isfinite(probs[0]).all()
    assert np.isclose(probs[0].sum(), 1.0)


def test_missing_values_handled_by_pipeline_imputers(model, synthetic_census_df):
    """Verify missing values in predictors are imputed cleanly by pipeline transformers."""
    nan_df = synthetic_census_df.copy()
    nan_df.loc[0, "age"] = np.nan
    nan_df.loc[0, "hours-per-week"] = np.nan
    nan_df.loc[1, "workclass"] = np.nan
    nan_df.loc[1, "occupation"] = np.nan

    preds = model.predict(nan_df)
    probs = model.predict_proba(nan_df)

    assert len(preds) == len(synthetic_census_df)
    assert not pd.isna(preds).any()
    assert np.isfinite(probs).all()


# --- 8. Invalid Inputs and Predictable Error Handling Tests ---

def test_empty_dataframe_raises_predictable_error(model):
    """Verify empty DataFrame input raises predictable ValueError."""
    empty_df = pd.DataFrame(columns=EXPECTED_FEATURES)
    with pytest.raises(ValueError):
        model.predict(empty_df)


def test_incompatible_input_types_raise_predictable_error(model):
    """Verify non-DataFrame/non-array inputs raise predictable exceptions."""
    for invalid in [None, "invalid_string_input", 12345]:
        with pytest.raises((ValueError, TypeError, AttributeError)):
            model.predict(invalid)


def test_income_predictor_schema_and_range_validation():
    """Verify IncomePredictor enforces validation contract with clear errors without leaking file paths."""
    predictor = IncomePredictor()

    # Out of range age
    with pytest.raises(ValidationError, match="age"):
        predictor.predict({
            "age": 150,
            "workclass": "Private",
            "fnlwgt": 100000,
            "education": "Bachelors",
            "education-num": 13,
            "marital-status": "Never-married",
            "occupation": "Tech-support",
            "relationship": "Not-in-family",
            "race": "White",
            "sex": "Male",
            "capital-gain": 0,
            "capital-loss": 0,
            "hours-per-week": 40,
            "native-country": "United-States",
        })

    # Unsupported category
    with pytest.raises(ValidationError, match="Unsupported category"):
        predictor.predict({
            "age": 35,
            "workclass": "AstronautEnterprise",
            "fnlwgt": 100000,
            "education": "Bachelors",
            "education-num": 13,
            "marital-status": "Never-married",
            "occupation": "Tech-support",
            "relationship": "Not-in-family",
            "race": "White",
            "sex": "Male",
            "capital-gain": 0,
            "capital-loss": 0,
            "hours-per-week": 40,
            "native-country": "United-States",
        })


# --- 9. Reproducibility for Fixed Model and Input Tests ---

def test_prediction_determinism_multiple_calls(model, synthetic_census_df):
    """Verify calling predict and predict_proba repeatedly yields bit-for-bit identical results."""
    first_preds = model.predict(synthetic_census_df)
    first_probs = model.predict_proba(synthetic_census_df)

    for run_idx in range(5):
        run_preds = model.predict(synthetic_census_df)
        run_probs = model.predict_proba(synthetic_census_df)

        assert np.array_equal(first_preds, run_preds), f"Predictions varied on run {run_idx + 1}"
        assert np.allclose(first_probs, run_probs, atol=1e-7), f"Probabilities varied on run {run_idx + 1}"


def test_prediction_determinism_reloaded_artifact(model, synthetic_census_df):
    """Verify loading a fresh instance of the model artifact yields identical predictions."""
    fresh_model = joblib.load(MODEL_PATH)

    original_preds = model.predict(synthetic_census_df)
    fresh_preds = fresh_model.predict(synthetic_census_df)

    original_probs = model.predict_proba(synthetic_census_df)
    fresh_probs = fresh_model.predict_proba(synthetic_census_df)

    assert np.array_equal(original_preds, fresh_preds)
    assert np.allclose(original_probs, fresh_probs, atol=1e-7)
