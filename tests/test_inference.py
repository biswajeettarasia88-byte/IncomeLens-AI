"""Comprehensive tests for IncomePredictor inference interface and validation contract."""

import py_compile
import sys
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from incomelens.inference import (
    IncomePredictor,
    ValidationError,
    InferenceError,
    load_model,
    EXPECTED_FEATURES,
)


@pytest.fixture
def valid_input_dict():
    """Valid standard sample input dictionary."""
    return {
        "age": 42,
        "workclass": "Private",
        "fnlwgt": 178356.0,
        "education": "Bachelors",
        "education-num": 13,
        "marital-status": "Married-civ-spouse",
        "occupation": "Exec-managerial",
        "relationship": "Husband",
        "race": "White",
        "sex": "Male",
        "capital-gain": 5000.0,
        "capital-loss": 0.0,
        "hours-per-week": 45,
        "native-country": "United-States",
    }


def test_load_production_model_succeeds():
    """Verify loading the production model artifact without retraining."""
    model = load_model()
    assert model is not None
    assert hasattr(model, "predict")
    assert hasattr(model, "named_steps")


def test_income_predictor_valid_single_prediction(valid_input_dict):
    """Verify valid single input prediction output contract."""
    predictor = IncomePredictor()
    result = predictor.predict(valid_input_dict)

    # Check contract keys
    assert "prediction" in result
    assert "raw_prediction" in result
    assert "high_income_probability" in result
    assert "low_income_probability" in result
    assert "probabilities" in result
    assert "confidence_level" in result

    # Check types and domains
    assert result["prediction"] in ["<=50K", ">50K"]
    assert result["raw_prediction"] in [0, 1]
    assert isinstance(result["high_income_probability"], float)
    assert 0.0 <= result["high_income_probability"] <= 1.0
    assert 0.0 <= result["low_income_probability"] <= 1.0
    assert np.isclose(
        result["high_income_probability"] + result["low_income_probability"],
        1.0,
        atol=1e-3,
    )
    assert result["confidence_level"] in ["High", "Moderate", "Low (Boundary Case)"]


def test_income_predictor_valid_batch_prediction(valid_input_dict):
    """Verify prediction works on DataFrame input."""
    predictor = IncomePredictor()
    df_input = pd.DataFrame([valid_input_dict, valid_input_dict])
    result = predictor.predict(df_input)

    assert result["prediction"] in ["<=50K", ">50K"]


def test_target_column_rejected_as_input_feature(valid_input_dict):
    """Requirement 9: Never use the target column as a user input feature."""
    predictor = IncomePredictor()

    # Pass in dict with 'income'
    bad_dict = valid_input_dict.copy()
    bad_dict["income"] = ">50K"
    with pytest.raises(ValidationError, match="prediction target and cannot be provided"):
        predictor.predict(bad_dict)

    # Pass in DataFrame with 'income'
    bad_df = pd.DataFrame([bad_dict])
    with pytest.raises(ValidationError, match="prediction target and cannot be provided"):
        predictor.predict(bad_df)


def test_missing_required_fields_raises_validation_error(valid_input_dict):
    """Requirement 5: Validate missing fields with actionable errors."""
    predictor = IncomePredictor()

    bad_dict = valid_input_dict.copy()
    del bad_dict["age"]
    del bad_dict["workclass"]

    with pytest.raises(ValidationError) as exc_info:
        predictor.predict(bad_dict)

    msg = str(exc_info.value)
    assert "Missing required input fields" in msg
    assert "age" in msg
    assert "workclass" in msg


def test_invalid_numeric_values_raise_validation_error(valid_input_dict):
    """Requirement 5: Validate out-of-range or non-numeric values."""
    predictor = IncomePredictor()

    # Negative age
    bad_dict = valid_input_dict.copy()
    bad_dict["age"] = -1
    with pytest.raises(ValidationError, match="Field 'age' must be between 17 and 100"):
        predictor.predict(bad_dict)

    # Age > 100
    bad_dict["age"] = 105
    with pytest.raises(ValidationError, match="Field 'age' must be between 17 and 100"):
        predictor.predict(bad_dict)

    # Negative hours
    bad_dict = valid_input_dict.copy()
    bad_dict["hours-per-week"] = 0
    with pytest.raises(ValidationError, match="Field 'hours-per-week' must be between 1 and 99"):
        predictor.predict(bad_dict)

    # Negative capital gain
    bad_dict = valid_input_dict.copy()
    bad_dict["capital-gain"] = -500.0
    with pytest.raises(ValidationError, match="Field 'capital-gain' cannot be negative"):
        predictor.predict(bad_dict)

    # Non-numeric string in numeric field
    bad_dict = valid_input_dict.copy()
    bad_dict["age"] = "not_a_number"
    with pytest.raises(ValidationError, match="must be a valid numeric value"):
        predictor.predict(bad_dict)


def test_unsupported_categories_raise_validation_error(valid_input_dict):
    """Requirement 5: Validate unsupported categories with actionable choices."""
    predictor = IncomePredictor()

    bad_dict = valid_input_dict.copy()
    bad_dict["workclass"] = "InvalidWorkclass123"

    with pytest.raises(ValidationError) as exc_info:
        predictor.predict(bad_dict)

    msg = str(exc_info.value)
    assert "Unsupported category 'InvalidWorkclass123' for field 'workclass'" in msg
    assert "Private" in msg or "Federal-gov" in msg


def test_standalone_estimator_reuses_fitted_preprocessor(valid_input_dict):
    """Requirement 4: If artifact contains only estimator, reuse exact fitted preprocessing artifact."""
    model_path = ROOT / "models" / "random_forest_final.joblib"
    full_pipe = joblib.load(model_path)

    estimator_only = full_pipe.named_steps["classifier"]
    preprocessor_only = full_pipe.named_steps["preprocessor"]

    # Explicit preprocessor
    predictor_explicit = IncomePredictor(model=estimator_only, preprocessor=preprocessor_only)
    res_explicit = predictor_explicit.predict(valid_input_dict)

    # Automatic reuse from default pipeline
    predictor_auto = IncomePredictor(model=estimator_only)
    res_auto = predictor_auto.predict(valid_input_dict)

    assert res_explicit["prediction"] == res_auto["prediction"]
    assert np.isclose(
        res_explicit["high_income_probability"],
        res_auto["high_income_probability"],
    )


def test_model_without_predict_proba_returns_none(valid_input_dict):
    """Requirement 7: Return high-income probability only if model supports valid probability prediction."""
    model_path = ROOT / "models" / "random_forest_final.joblib"
    full_pipe = joblib.load(model_path)

    class DummyEstimatorNoProba:
        def __init__(self):
            self.classes_ = [0, 1]

        def predict(self, X):
            return np.array([1])

    predictor = IncomePredictor(
        model=DummyEstimatorNoProba(),
        preprocessor=full_pipe.named_steps["preprocessor"],
    )
    result = predictor.predict(valid_input_dict)

    assert result["prediction"] == ">50K"
    assert result["high_income_probability"] is None
    assert result["low_income_probability"] is None
    assert result["probabilities"] is None
    assert "Unknown" in result["confidence_level"]


def test_error_messages_do_not_leak_internal_paths(valid_input_dict):
    """Requirement 10: Do not expose internal file paths or stack traces to ordinary users."""
    predictor = IncomePredictor()
    bad_dict = valid_input_dict.copy()
    bad_dict["age"] = "invalid"

    try:
        predictor.predict(bad_dict)
    except ValidationError as e:
        msg = str(e)
        assert "D:\\" not in msg
        assert "C:\\" not in msg
        assert "/home" not in msg
        assert "Traceback" not in msg


def test_income_predictor_local_shap_explanation(valid_input_dict):
    """Verify local per-instance SHAP explanation structure, target class, and additivity."""
    predictor = IncomePredictor()
    result = predictor.predict(valid_input_dict, explain=True)

    assert result["explanation"] is not None
    exp = result["explanation"]

    # Target class must be >50K
    assert exp["target_class"] == ">50K"
    assert "base_value" in exp
    assert "sum_contributions" in exp
    assert "reconstructed_probability" in exp

    # Check mathematical additivity
    assert np.isclose(
        exp["base_value"] + exp["sum_contributions"],
        exp["reconstructed_probability"],
        atol=1e-3,
    )
    # Reconstructed probability matches high_income_probability
    assert np.isclose(
        exp["reconstructed_probability"],
        result["high_income_probability"],
        atol=1e-3,
    )

    # Check top contributions structure
    assert len(exp["top_contributions"]) > 0
    for item in exp["top_contributions"]:
        assert "feature" in item
        assert "display_name" in item
        assert "shap_value" in item
        assert "direction" in item
        assert item["direction"] in ["Positive (+)", "Negative (-)"]
        if item["direction"] == "Positive (+)":
            assert item["shap_value"] > 0
        else:
            assert item["shap_value"] < 0

    # Positive and negative separation
    for p in exp["top_positive"]:
        assert p["shap_value"] > 0
    for n in exp["top_negative"]:
        assert n["shap_value"] < 0


def test_local_shap_differs_by_profile_not_global(valid_input_dict):
    """Requirement 9: Do not display a global SHAP ranking as if it explained an individual prediction."""
    predictor = IncomePredictor()

    # Profile 1: High capital gain, exec-managerial
    res1 = predictor.predict(valid_input_dict, explain=True)

    # Profile 2: Low-income profile
    low_dict = {
        "age": 19,
        "workclass": "Private",
        "fnlwgt": 178356.0,
        "education": "10th",
        "education-num": 6,
        "marital-status": "Never-married",
        "occupation": "Handlers-cleaners",
        "relationship": "Own-child",
        "race": "White",
        "sex": "Female",
        "capital-gain": 0.0,
        "capital-loss": 0.0,
        "hours-per-week": 20,
        "native-country": "United-States",
    }
    res2 = predictor.predict(low_dict, explain=True)

    exp1 = res1["explanation"]
    exp2 = res2["explanation"]

    # Verify that different instances receive distinct per-instance contributions
    top1_feat = exp1["top_contributions"][0]["feature"]
    top2_feat = exp2["top_contributions"][0]["feature"]
    top1_val = exp1["top_contributions"][0]["shap_value"]
    top2_val = exp2["top_contributions"][0]["shap_value"]

    assert (top1_feat != top2_feat) or not np.isclose(top1_val, top2_val)


def test_explanation_failure_does_not_break_prediction(valid_input_dict, monkeypatch):
    """Requirement 8: Handle explanation failures without breaking ordinary prediction."""
    predictor = IncomePredictor()

    # Force explain_instance to raise an exception
    def broken_explain(*args, **kwargs):
        raise RuntimeError("Simulated explainer failure")

    monkeypatch.setattr(predictor, "explain_instance", broken_explain)

    result = predictor.predict(valid_input_dict, explain=True)

    # Ordinary prediction must still succeed completely
    assert result["prediction"] in ["<=50K", ">50K"]
    assert result["high_income_probability"] is not None
    assert result["explanation"] is None
    assert "explanation_error" in result


def test_explainer_initialization_is_cached():
    """Requirement 7: Cache expensive model initialization where appropriate."""
    predictor = IncomePredictor()

    exp1 = predictor.get_explainer()
    exp2 = predictor.get_explainer()

    assert exp1 is exp2


def test_streamlit_app_pages_compile():
    """Verify that all Streamlit app scripts compile cleanly without syntax errors."""
    app_files = [
        ROOT / "app" / "Home.py",
        ROOT / "app" / "pages" / "1_Predict_Income.py",
        ROOT / "app" / "pages" / "2_Model_Comparison.py",
        ROOT / "app" / "pages" / "3_Explainability.py",
        ROOT / "app" / "pages" / "4_Fairness_Audit.py",
    ]

    for py_file in app_files:
        assert py_file.is_file(), f"App file missing: {py_file}"
        py_compile.compile(str(py_file), doraise=True)
