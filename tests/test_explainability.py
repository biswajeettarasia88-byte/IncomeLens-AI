"""Tests for incomelens.explainability module and SHAP output validation."""

import sys
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import pytest
from scipy import sparse
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from incomelens.data import load_train_data
from incomelens.explainability import (
    extract_estimator_and_preprocessor,
    extract_positive_class_shap_values,
    generate_shap_table,
    save_shap_summary_plot,
    to_dense_numeric,
)


def test_to_dense_numeric_from_sparse():
    """Verify sparse matrix conversion to dense float64 matrix."""
    mat = sparse.csr_matrix([[1.0, 0.0, 3.5], [0.0, 2.0, 0.0]])
    dense = to_dense_numeric(mat)

    assert isinstance(dense, np.ndarray)
    assert dense.dtype == np.float64
    assert dense.shape == (2, 3)
    assert np.allclose(dense, [[1.0, 0.0, 3.5], [0.0, 2.0, 0.0]])


def test_to_dense_numeric_rejects_object_dtype_with_invalid_cast():
    """Verify non-numeric types that cannot be safely converted to float64 raise error."""
    arr = np.array([["apple", "banana"], ["cat", "dog"]], dtype=object)
    with pytest.raises(ValueError):
        to_dense_numeric(arr)


def test_to_dense_numeric_rejects_nan_and_inf():
    """Verify arrays containing NaN or inf raise ValueError."""
    arr_nan = np.array([[1.0, np.nan], [2.0, 3.0]])
    with pytest.raises(ValueError) as exc_info:
        to_dense_numeric(arr_nan)
    assert "contains NaN or infinity" in str(exc_info.value)

    arr_inf = np.array([[1.0, np.inf], [2.0, 3.0]])
    with pytest.raises(ValueError) as exc_info:
        to_dense_numeric(arr_inf)
    assert "contains NaN or infinity" in str(exc_info.value)


def test_to_dense_numeric_rejects_non_2d():
    """Verify 1D or 3D arrays raise ValueError."""
    arr_1d = np.array([1.0, 2.0, 3.0])
    with pytest.raises(ValueError) as exc_info:
        to_dense_numeric(arr_1d)
    assert "Expected 2D feature matrix" in str(exc_info.value)

    arr_3d = np.zeros((2, 2, 2))
    with pytest.raises(ValueError) as exc_info:
        to_dense_numeric(arr_3d)
    assert "Expected 2D feature matrix" in str(exc_info.value)


def test_shap_values_computation_regression():
    """Regression test ensuring SHAP calculation does not raise UFuncInputCastingError and outputs (N, 105)."""
    import shap

    model_path = ROOT / "models" / "random_forest_final.joblib"
    if not model_path.exists():
        pytest.skip("Production model not found.")

    model = joblib.load(model_path)
    preprocessor = model.named_steps["preprocessor"]
    estimator = model.named_steps["classifier"]

    # Sample small batch of training rows
    sample_df = load_train_data().drop(columns=["income"]).head(10)
    X_processed = preprocessor.transform(sample_df)
    X_dense = to_dense_numeric(X_processed)

    assert X_dense.dtype == np.float64
    assert X_dense.shape == (10, 105)

    # Explain tree estimator directly
    explainer = shap.TreeExplainer(estimator)
    shap_raw = explainer.shap_values(X_dense)

    if isinstance(shap_raw, list):
        shap_values = shap_raw[1] if len(shap_raw) > 1 else shap_raw[0]
    else:
        arr = np.asarray(shap_raw)
        if arr.ndim == 3:
            shap_values = arr[:, :, 1] if arr.shape[2] > 1 else arr[:, :, 0]
        else:
            shap_values = arr

    assert isinstance(shap_values, np.ndarray)
    assert shap_values.shape == (10, 105)
    assert not np.isnan(shap_values).any()
    assert np.isfinite(shap_values).all()


def test_saved_shap_artifacts_integrity():
    """Verify the saved reports/shap_values.csv and shap_summary.png on disk."""
    shap_csv = ROOT / "reports" / "shap_values.csv"
    if not shap_csv.exists():
        pytest.skip("reports/shap_values.csv not yet generated.")

    df = pd.read_csv(shap_csv)
    assert len(df) == 1000, f"Expected 1000 rows in SHAP table, got {len(df)}"
    assert len(df.columns) == 105, f"Expected 105 features in SHAP table, got {len(df.columns)}"
    assert not df.isna().any().any(), "SHAP values table contains NaNs."

    # Check plot
    summary_png = ROOT / "reports" / "figures" / "shap_summary.png"
    assert summary_png.exists(), "shap_summary.png not found."
    assert summary_png.stat().st_size > 50000, "shap_summary.png file size is suspiciously small."


def test_shap_mathematical_additivity():
    """Verify SHAP mathematical additivity: base_value + sum(shap_values) == predict_proba."""
    import shap

    model_path = ROOT / "models" / "random_forest_final.joblib"
    if not model_path.exists():
        pytest.skip("Production model not found.")

    model = joblib.load(model_path)
    preprocessor = model.named_steps["preprocessor"]
    estimator = model.named_steps["classifier"]

    # Sample small batch of training rows
    sample_df = load_train_data().drop(columns=["income"]).head(10)
    X_processed = preprocessor.transform(sample_df)
    X_dense = to_dense_numeric(X_processed)

    explainer = shap.TreeExplainer(estimator)
    shap_raw = explainer.shap_values(X_dense)
    shap_arr = np.asarray(shap_raw)

    probs = estimator.predict_proba(X_dense)

    # Class 0 (<=50K)
    base_val_0 = explainer.expected_value[0]
    shap_0 = shap_arr[:, :, 0]
    recon_0 = base_val_0 + shap_0.sum(axis=1)
    assert np.allclose(recon_0, probs[:, 0], atol=1e-5)

    # Class 1 (>50K)
    base_val_1 = explainer.expected_value[1]
    shap_1 = shap_arr[:, :, 1]
    recon_1 = base_val_1 + shap_1.sum(axis=1)
    assert np.allclose(recon_1, probs[:, 1], atol=1e-5)

    # Complementary property: shap_0 + shap_1 == 0
    assert np.allclose(shap_0 + shap_1, 0.0, atol=1e-5)


# --- New Focused Explainability Unit Tests ---

def test_extract_estimator_and_preprocessor_pipeline_and_standalone():
    """Verify correct tree estimator selection from Pipelines and standalone models."""
    rf = RandomForestClassifier(n_estimators=5, random_state=42)
    scaler = StandardScaler()

    # Two-step Pipeline
    pipe = Pipeline([("scaler", scaler), ("classifier", rf)])
    preprocessor, estimator = extract_estimator_and_preprocessor(pipe)
    assert estimator is rf
    assert preprocessor is not None

    # Standalone estimator
    prep_none, est_standalone = extract_estimator_and_preprocessor(rf)
    assert prep_none is None
    assert est_standalone is rf

    # Single-step Pipeline raises ValueError
    bad_pipe = Pipeline([("classifier", rf)])
    with pytest.raises(ValueError, match="Pipeline must contain preprocessing and a final estimator"):
        extract_estimator_and_preprocessor(bad_pipe)


def test_extract_positive_class_shap_values_3d_and_list_formats():
    """Verify SHAP output shape and correct positive-class slice extraction."""
    # 3D array: 5 samples x 3 features x 2 classes
    arr_3d = np.zeros((5, 3, 2), dtype=np.float64)
    arr_3d[:, :, 0] = -0.5  # Class 0 values
    arr_3d[:, :, 1] = 0.5   # Class 1 values

    # Extract class 1
    extracted_c1 = extract_positive_class_shap_values(arr_3d, positive_class_idx=1)
    assert extracted_c1.shape == (5, 3)
    assert np.allclose(extracted_c1, 0.5)

    # Extract class 0
    extracted_c0 = extract_positive_class_shap_values(arr_3d, positive_class_idx=0)
    assert extracted_c0.shape == (5, 3)
    assert np.allclose(extracted_c0, -0.5)

    # List format: [class_0_mat, class_1_mat]
    mat_0 = np.full((4, 2), -1.2)
    mat_1 = np.full((4, 2), 1.2)
    extracted_list = extract_positive_class_shap_values([mat_0, mat_1], positive_class_idx=1)
    assert extracted_list.shape == (4, 2)
    assert np.allclose(extracted_list, 1.2)

    # Invalid dimension raises ValueError
    with pytest.raises(ValueError, match="expected 2D matrix"):
        extract_positive_class_shap_values(np.zeros((5, 3, 2, 2)))


def test_generate_shap_table_feature_name_alignment():
    """Verify feature-name alignment and column count validation in SHAP DataFrame generation."""
    shap_vals = np.array([[0.1, -0.2, 0.3], [0.4, 0.0, -0.1]])
    feature_names = ["feat_age", "feat_education", "feat_hours"]

    df = generate_shap_table(shap_vals, feature_names)
    assert isinstance(df, pd.DataFrame)
    assert list(df.columns) == feature_names
    assert len(df) == 2
    assert df.loc[0, "feat_age"] == 0.1
    assert df.loc[1, "feat_hours"] == -0.1

    # Feature count mismatch raises ValueError
    with pytest.raises(ValueError, match="Feature count mismatch"):
        generate_shap_table(shap_vals, ["only_one_feature"])


def test_save_shap_summary_plot_nonempty_readable(tmp_path):
    """Verify nonempty, readable plot generation with valid PNG format."""
    rng = np.random.RandomState(42)
    values = rng.normal(size=(10, 3))
    X_sample = rng.normal(size=(10, 3))
    feature_names = ["f1", "f2", "f3"]
    plot_path = tmp_path / "test_shap_summary.png"

    saved = save_shap_summary_plot(values, X_sample, feature_names, plot_path)
    assert saved.is_file()
    assert saved.stat().st_size > 1000, "Generated plot file is suspiciously small or empty."

    # Validate standard PNG magic header
    with open(saved, "rb") as f:
        header = f.read(8)
    assert header == b"\x89PNG\r\n\x1a\n", "Generated file does not contain valid PNG magic header."


def test_synthetic_end_to_end_tree_explainability(tmp_path):
    """Verify complete explainability workflow using small synthetic tree model and dataset (~30ms)."""
    import shap

    # Create small synthetic numeric dataset (15 samples, 4 features)
    rng = np.random.RandomState(42)
    X_synthetic = rng.uniform(10, 80, size=(15, 4))
    y_synthetic = (X_synthetic[:, 0] > 40).astype(int)
    feature_names = ["num__age", "num__capital_gain", "num__hours", "num__education_num"]

    rf = RandomForestClassifier(n_estimators=3, max_depth=2, random_state=42)
    rf.fit(X_synthetic, y_synthetic)

    explainer = shap.TreeExplainer(rf)
    shap_raw = explainer.shap_values(X_synthetic)

    # 1. Correct class selection (class 1 for positive prediction)
    values = extract_positive_class_shap_values(shap_raw, positive_class_idx=1)
    assert values.shape == (15, 4)

    # 2. Valid saved CSV structure
    table_df = generate_shap_table(values, feature_names)
    assert list(table_df.columns) == feature_names
    assert not table_df.isna().any().any()

    csv_path = tmp_path / "synthetic_shap.csv"
    table_df.to_csv(csv_path, index=False)
    assert csv_path.is_file()
    reloaded = pd.read_csv(csv_path)
    assert reloaded.shape == (15, 4)

    # 3. Nonempty, readable plot generation
    plot_path = tmp_path / "synthetic_shap.png"
    save_shap_summary_plot(values, X_synthetic, feature_names, plot_path)
    assert plot_path.is_file()
    assert plot_path.stat().st_size > 2000
