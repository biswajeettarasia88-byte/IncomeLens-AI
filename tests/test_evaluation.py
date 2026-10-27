"""Tests for incomelens.evaluation module."""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from incomelens.evaluation import (
    NEGATIVE_CLASS_INT,
    POSITIVE_CLASS_INT,
    compute_metrics,
    evaluate_predictions,
    get_positive_class_index,
)


def test_get_positive_class_index_success():
    """Verify get_positive_class_index finds the correct column for various class array formats."""
    # Standard binary [0, 1]
    assert get_positive_class_index(np.array([0, 1]), positive_class=1) == 1
    # Inverted order [1, 0]
    assert get_positive_class_index(np.array([1, 0]), positive_class=1) == 0
    # String labels
    assert get_positive_class_index(["<=50K", ">50K"], positive_class=">50K") == 1
    # Fallback to string representation of int
    assert get_positive_class_index(["0", "1"], positive_class=1) == 1


def test_get_positive_class_index_missing_class_raises_error():
    """Verify get_positive_class_index raises ValueError when class is missing."""
    with pytest.raises(ValueError) as exc_info:
        get_positive_class_index(np.array([0, 2]), positive_class=1)
    assert "not found in model classes" in str(exc_info.value)


def test_compute_metrics_perfect_predictions():
    """Verify compute_metrics on perfect prediction scenario."""
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 0, 1, 1])
    y_prob = np.array([0.1, 0.2, 0.9, 0.85])

    metrics = compute_metrics(y_true, y_pred, y_prob=y_prob, positive_class=1)

    assert metrics["accuracy"] == 1.0
    assert metrics["balanced_accuracy"] == 1.0
    assert metrics["precision_high_income"] == 1.0
    assert metrics["recall_high_income"] == 1.0
    assert metrics["f1_high_income"] == 1.0
    assert metrics["roc_auc"] == 1.0


def test_compute_metrics_zero_positive_predictions():
    """Verify compute_metrics handles zero positive predictions without dividing by zero."""
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 0, 0, 0])
    y_prob = np.array([0.1, 0.2, 0.3, 0.4])

    metrics = compute_metrics(y_true, y_pred, y_prob=y_prob, positive_class=1)

    assert metrics["accuracy"] == 0.5
    assert metrics["balanced_accuracy"] == 0.5
    assert metrics["precision_high_income"] == 0.0
    assert metrics["recall_high_income"] == 0.0
    assert metrics["f1_high_income"] == 0.0
    assert metrics["roc_auc"] == 1.0


def test_compute_metrics_length_mismatch():
    """Verify compute_metrics raises ValueError on mismatched input lengths."""
    with pytest.raises(ValueError) as exc_info:
        compute_metrics([0, 1], [0])
    assert "Length mismatch" in str(exc_info.value)

    with pytest.raises(ValueError) as exc_info:
        compute_metrics([0, 1], [0, 1], y_prob=[0.5])
    assert "Length mismatch" in str(exc_info.value)


def test_compute_metrics_empty_inputs():
    """Verify compute_metrics raises ValueError on empty inputs."""
    with pytest.raises(ValueError) as exc_info:
        compute_metrics([], [])
    assert "cannot be empty" in str(exc_info.value)


def test_evaluate_predictions_outputs():
    """Verify evaluate_predictions returns metrics dict, formatted confusion matrix, and report."""
    y_true = [0, 0, 1, 1]
    y_pred = [0, 1, 0, 1]
    y_prob = [0.2, 0.6, 0.4, 0.8]

    metrics, cm_df, report = evaluate_predictions(y_true, y_pred, y_prob)

    assert isinstance(metrics, dict)
    assert "accuracy" in metrics
    assert "f1_high_income" in metrics

    assert isinstance(cm_df, pd.DataFrame)
    assert list(cm_df.index) == ["actual_<=50K", "actual_>50K"]
    assert list(cm_df.columns) == ["predicted_<=50K", "predicted_>50K"]
    assert cm_df.loc["actual_<=50K", "predicted_<=50K"] == 1
    assert cm_df.loc["actual_<=50K", "predicted_>50K"] == 1
    assert cm_df.loc["actual_>50K", "predicted_<=50K"] == 1
    assert cm_df.loc["actual_>50K", "predicted_>50K"] == 1

    assert isinstance(report, str)
    assert "<=50K" in report
    assert ">50K" in report


# --- New Focused Evaluation Unit Tests ---

def test_metrics_use_correct_positive_class_label():
    """Verify metrics calculate precision, recall, and F1 specifically for the high-income class (>50K = 1)."""
    # Asymmetric scenario:
    # 70 actual negatives (0), 30 actual positives (1)
    y_true = np.array([0] * 70 + [1] * 30)

    # Model predicts:
    # Negatives: 65 predicted as 0 (TN), 5 predicted as 1 (FP)
    # Positives: 10 predicted as 0 (FN), 20 predicted as 1 (TP)
    y_pred = np.array([0] * 65 + [1] * 5 + [0] * 10 + [1] * 20)
    y_prob = np.array([0.1] * 65 + [0.8] * 5 + [0.2] * 10 + [0.9] * 20)

    metrics = compute_metrics(y_true, y_pred, y_prob=y_prob, positive_class=POSITIVE_CLASS_INT)

    # Mathematical expectations for positive class (1):
    # TP = 20, FP = 5, FN = 10, TN = 65
    expected_precision = 20 / (20 + 5)  # 20 / 25 = 0.8000
    expected_recall = 20 / (20 + 10)    # 20 / 30 = 0.6667
    expected_f1 = 2 * (expected_precision * expected_recall) / (expected_precision + expected_recall)

    assert metrics["precision_high_income"] == pytest.approx(expected_precision, abs=1e-4)
    assert metrics["recall_high_income"] == pytest.approx(expected_recall, abs=1e-4)
    assert metrics["f1_high_income"] == pytest.approx(expected_f1, abs=1e-4)

    # If the function mistakenly evaluated class 0 as positive, precision would be 65 / (65 + 10) = 0.8667
    assert metrics["precision_high_income"] != pytest.approx(65 / 75, abs=1e-2)


def test_confusion_matrix_orientation_and_marginal_sums():
    """Verify confusion matrix orientation: rows are strictly Actual, columns are strictly Predicted."""
    # 70 actual negatives (0), 30 actual positives (1)
    y_true = np.array([0] * 70 + [1] * 30)
    # TN=65, FP=5, FN=10, TP=20
    y_pred = np.array([0] * 65 + [1] * 5 + [0] * 10 + [1] * 20)

    _, cm_df, _ = evaluate_predictions(y_true, y_pred)

    # Orientation verification
    assert list(cm_df.index) == ["actual_<=50K", "actual_>50K"]
    assert list(cm_df.columns) == ["predicted_<=50K", "predicted_>50K"]

    # Cell-level counts
    assert cm_df.loc["actual_<=50K", "predicted_<=50K"] == 65  # True Negatives (TN)
    assert cm_df.loc["actual_<=50K", "predicted_>50K"] == 5    # False Positives (FP)
    assert cm_df.loc["actual_>50K", "predicted_<=50K"] == 10   # False Negatives (FN)
    assert cm_df.loc["actual_>50K", "predicted_>50K"] == 20    # True Positives (TP)

    # Row marginal sums must equal ground truth class distributions
    assert cm_df.loc["actual_<=50K"].sum() == 70
    assert cm_df.loc["actual_>50K"].sum() == 30

    # Column marginal sums must equal predicted class distributions
    assert cm_df["predicted_<=50K"].sum() == 75
    assert cm_df["predicted_>50K"].sum() == 25


def test_undefined_metrics_single_class_ground_truth_safe_handling():
    """Verify undefined metrics (e.g. single-class ROC-AUC, zero division) fail safely without raising exceptions."""
    # Single class ground truth (all negatives: no positives exist)
    y_true_all_neg = np.array([0, 0, 0, 0, 0])
    y_pred_with_pos = np.array([0, 1, 0, 0, 0])
    y_prob = np.array([0.1, 0.7, 0.2, 0.3, 0.4])

    metrics_neg = compute_metrics(y_true_all_neg, y_pred_with_pos, y_prob=y_prob, positive_class=1)

    # ROC-AUC is undefined for single class; should safely return NaN
    assert np.isnan(metrics_neg["roc_auc"])
    # Recall for positive class is 0.0 without ZeroDivisionError
    assert metrics_neg["recall_high_income"] == 0.0

    # Single class ground truth (all positives: no negatives exist)
    y_true_all_pos = np.array([1, 1, 1, 1])
    y_pred_all_pos = np.array([1, 0, 1, 1])
    y_prob_pos = np.array([0.9, 0.3, 0.8, 0.85])

    metrics_pos = compute_metrics(y_true_all_pos, y_pred_all_pos, y_prob=y_prob_pos, positive_class=1)
    assert np.isnan(metrics_pos["roc_auc"])
    assert metrics_pos["recall_high_income"] == pytest.approx(3 / 4)


def test_saved_final_test_metrics_report_columns():
    """Verify reports/final_test_metrics.csv exists and contains canonical metrics columns."""
    report_path = ROOT / "reports" / "final_test_metrics.csv"
    assert report_path.is_file(), f"Expected metrics report not found at {report_path}"

    df = pd.read_csv(report_path)
    assert len(df) == 1

    expected_cols = [
        "accuracy",
        "balanced_accuracy",
        "precision_high_income",
        "recall_high_income",
        "f1_high_income",
        "roc_auc",
    ]
    assert list(df.columns) == expected_cols

    # Verify all metric values are valid probabilities / bounded rates
    for col in expected_cols:
        val = float(df[col].iloc[0])
        assert not np.isnan(val), f"Metric {col} is NaN"
        assert 0.0 <= val <= 1.0, f"Metric {col}={val} out of bounds [0, 1]"


def test_saved_confusion_matrix_report_structure():
    """Verify reports/confusion_matrix.csv exists and matches expected format and sample size."""
    cm_path = ROOT / "reports" / "confusion_matrix.csv"
    assert cm_path.is_file(), f"Expected confusion matrix not found at {cm_path}"

    df = pd.read_csv(cm_path, index_col=0)
    assert list(df.index) == ["actual_<=50K", "actual_>50K"]
    assert list(df.columns) == ["predicted_<=50K", "predicted_>50K"]

    # Verify counts are non-negative integers
    for r in df.index:
        for c in df.columns:
            count = df.loc[r, c]
            assert isinstance(count, (int, np.integer))
            assert count >= 0

    # Total sum must equal held-out test set size (9,771)
    total_test_samples = int(df.values.sum())
    assert total_test_samples == 9771


def test_saved_fairness_metrics_report_columns():
    """Verify reports/fairness_metrics.csv exists and contains audit schema columns."""
    fairness_path = ROOT / "reports" / "fairness_metrics.csv"
    assert fairness_path.is_file(), f"Expected fairness report not found at {fairness_path}"

    df = pd.read_csv(fairness_path)
    assert not df.empty

    expected_cols = [
        "sensitive_attribute",
        "subgroup",
        "reference_group",
        "sample_count",
        "selection_rate",
        "true_positive_rate",
        "false_positive_rate",
        "precision",
        "disparate_impact_ratio",
    ]
    for col in expected_cols:
        assert col in df.columns, f"Missing fairness column: {col}"
