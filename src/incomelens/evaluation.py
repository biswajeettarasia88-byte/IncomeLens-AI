"""Evaluation metrics, confusion matrices, and reporting for IncomeLens AI."""

import logging
from typing import Any, Dict, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

logger = logging.getLogger(__name__)

POSITIVE_LABEL_STR: str = ">50K"
NEGATIVE_LABEL_STR: str = "<=50K"
POSITIVE_CLASS_INT: int = 1
NEGATIVE_CLASS_INT: int = 0


def get_positive_class_index(
    classes: Union[np.ndarray, list],
    positive_class: Union[int, str] = 1,
) -> int:
    """Find and validate the probability column index for the positive class.

    Parameters
    ----------
    classes : array-like
        The classes_ attribute from a fitted classifier or pipeline.
    positive_class : int or str, default=1
        The identifier of the positive class (e.g. 1 or '>50K').

    Returns
    -------
    int
        The integer column index in predict_proba corresponding to positive_class.

    Raises
    ------
    ValueError
        If positive_class is not found in classes.
    """
    classes_list = list(classes)
    if positive_class in classes_list:
        return classes_list.index(positive_class)

    # Check string representation fallback
    str_classes = [str(c) for c in classes_list]
    if str(positive_class) in str_classes:
        return str_classes.index(str(positive_class))

    raise ValueError(
        f"Positive class '{positive_class}' not found in model classes: {classes_list}"
    )


def compute_metrics(
    y_true: Union[pd.Series, np.ndarray, list],
    y_pred: Union[pd.Series, np.ndarray, list],
    y_prob: Optional[Union[pd.Series, np.ndarray, list]] = None,
    positive_class: int = 1,
) -> Dict[str, float]:
    """Compute the canonical set of classification metrics for income prediction.

    Metrics reported:
    - accuracy
    - balanced_accuracy
    - precision_high_income (positive class precision)
    - recall_high_income (positive class recall / sensitivity)
    - f1_high_income (positive class F1)
    - roc_auc (ROC-AUC for positive class probabilities, if provided)

    Parameters
    ----------
    y_true : array-like
        Ground truth binary labels (0 and 1).
    y_pred : array-like
        Predicted class labels (0 and 1).
    y_prob : array-like, optional
        Predicted probabilities for the positive class.
    positive_class : int, default=1
        Positive class integer label (1 for >50K).

    Returns
    -------
    dict of str -> float
        Dictionary of computed metrics.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if len(y_true) != len(y_pred):
        raise ValueError(
            f"Length mismatch: len(y_true)={len(y_true)} vs len(y_pred)={len(y_pred)}"
        )

    if len(y_true) == 0:
        raise ValueError("Evaluation arrays cannot be empty.")

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision_high_income": float(
            precision_score(y_true, y_pred, pos_label=positive_class, zero_division=0)
        ),
        "recall_high_income": float(
            recall_score(y_true, y_pred, pos_label=positive_class, zero_division=0)
        ),
        "f1_high_income": float(
            f1_score(y_true, y_pred, pos_label=positive_class, zero_division=0)
        ),
    }

    if y_prob is not None:
        y_prob = np.asarray(y_prob)
        if len(y_prob) != len(y_true):
            raise ValueError(
                f"Length mismatch: len(y_true)={len(y_true)} vs len(y_prob)={len(y_prob)}"
            )
        try:
            metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob))
        except ValueError as e:
            logger.warning("ROC-AUC calculation skipped (undefined metric): %s", e)
            metrics["roc_auc"] = float("nan")

    return metrics


def evaluate_predictions(
    y_true: Union[pd.Series, np.ndarray, list],
    y_pred: Union[pd.Series, np.ndarray, list],
    y_prob: Optional[Union[pd.Series, np.ndarray, list]] = None,
    positive_class: int = 1,
) -> Tuple[Dict[str, float], pd.DataFrame, str]:
    """Compute comprehensive evaluation outputs: metrics dictionary, confusion matrix DataFrame, and classification report string.

    Parameters
    ----------
    y_true : array-like
        Ground truth binary labels.
    y_pred : array-like
        Predicted class labels.
    y_prob : array-like, optional
        Predicted probabilities for positive class.
    positive_class : int, default=1
        Label of positive class.

    Returns
    -------
    metrics : dict of str -> float
        Summary metrics.
    cm_df : pd.DataFrame
        Formatted confusion matrix with actual and predicted index/columns.
    report_str : str
        Scikit-learn classification report string.
    """
    logger.info("Evaluating predictions on %d instances (positive_class=%s)", len(y_true), positive_class)
    metrics = compute_metrics(
        y_true, y_pred, y_prob=y_prob, positive_class=positive_class
    )

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    cm_df = pd.DataFrame(
        cm,
        index=["actual_<=50K", "actual_>50K"],
        columns=["predicted_<=50K", "predicted_>50K"],
    )

    report_str = classification_report(
        y_true,
        y_pred,
        labels=[0, 1],
        target_names=["<=50K", ">50K"],
        zero_division=0,
    )

    return metrics, cm_df, report_str
