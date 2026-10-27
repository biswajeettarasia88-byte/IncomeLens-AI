"""Tests for feature importance output structure and artifact integrity."""

import sys
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

CSV_PATH = ROOT / "reports" / "feature_importance.csv"
PNG_PATH = ROOT / "reports" / "figures" / "feature_importance.png"
MODEL_PATH = ROOT / "models" / "random_forest_final.joblib"


def test_feature_importance_csv_structure():
    """Verify reports/feature_importance.csv schema, dimensions, and properties."""
    assert CSV_PATH.is_file(), f"Expected artifact not found: {CSV_PATH}"

    df = pd.read_csv(CSV_PATH)

    # Required columns
    assert list(df.columns) == ["feature", "importance"], f"Unexpected columns: {df.columns}"

    # Dimensionality: 105 transformed features
    assert len(df) == 105, f"Expected 105 features, got {len(df)}"

    # Check for NaN, nulls, infs
    assert not df.isna().any().any(), "CSV contains NaN or null values"
    assert np.isfinite(df["importance"]).all(), "CSV contains infinite values"

    # Values must be non-negative probabilities/weights summing to 1.0
    assert (df["importance"] >= 0.0).all(), "Found negative importance values"
    assert np.isclose(df["importance"].sum(), 1.0, atol=1e-4), f"Importances sum to {df['importance'].sum()}, expected 1.0"

    # Must be sorted in descending order
    assert df["importance"].is_monotonic_decreasing, "Importance values are not sorted in descending order"


def test_feature_importance_names_match_preprocessor():
    """Verify feature names in the CSV match preprocessor.get_feature_names_out()."""
    if not MODEL_PATH.is_file():
        pytest.skip("Model artifact not found.")

    model = joblib.load(MODEL_PATH)
    preprocessor = model.named_steps["preprocessor"]
    expected_names = list(preprocessor.get_feature_names_out())

    df = pd.read_csv(CSV_PATH)
    actual_names = list(df["feature"])

    assert set(actual_names) == set(expected_names), "Feature names in CSV do not match preprocessor features"
    assert len(actual_names) == len(expected_names)


def test_feature_importance_figure_artifact():
    """Verify that reports/figures/feature_importance.png exists and is valid."""
    assert PNG_PATH.is_file(), f"Plot artifact not found: {PNG_PATH}"
    assert PNG_PATH.stat().st_size > 10000, f"Plot file size suspiciously small: {PNG_PATH.stat().st_size} bytes"
