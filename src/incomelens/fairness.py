"""Fairness and group-level disparity auditing module for IncomeLens-AI.

This module provides functions to audit model performance across demographic
and sensitive sub-populations. It computes standard fair ML metrics including:
- Selection rate (demographic parity)
- True positive rate (equal opportunity)
- False positive rate (predictive equality)
- Precision / Positive predictive value (predictive parity)
- Disparate impact ratio (four-fifths rule assessment)
- Sample counts and uncertainty estimation (confidence intervals)

IMPORTANT METHODOLOGICAL AND ETHICAL NOTICE:
- Algorithmic fairness metrics are diagnostic tools, NOT proof of legal or ethical compliance.
- Removing sensitive features from input representations does NOT prevent indirect or proxy
  discrimination due to correlation with non-sensitive features (e.g., occupation, zip code).
- Disparities in selection rate often reflect underlying base-rate differences in the
  historical training corpus.
"""

from typing import Any, Dict, List, Optional, Union
import numpy as np
import pandas as pd


def _safe_divide(numerator: float, denominator: float, fill_value: float = np.nan) -> float:
    """Perform division handling zero denominators safely without raising ZeroDivisionError."""
    if denominator == 0.0 or np.isnan(denominator) or np.isnan(numerator):
        return fill_value
    return float(numerator / denominator)


def compute_wilson_score_interval(
    successes: int,
    total: int,
    confidence: float = 0.95,
) -> tuple[float, float]:
    """Compute the Wilson score confidence interval for a binomial proportion.

    Parameters
    ----------
    successes : int
        Number of positive events (e.g., positive predictions, true positives).
    total : int
        Total sample count.
    confidence : float, default=0.95
        Confidence level (0.95 corresponds to z=1.96).

    Returns
    -------
    tuple of (float, float)
        (lower_bound, upper_bound)
    """
    if total <= 0:
        return (np.nan, np.nan)

    # z-score for 95% is 1.95996
    z = 1.95996 if np.isclose(confidence, 0.95) else 1.96
    p_hat = successes / total

    denom = 1.0 + (z**2) / total
    center = (p_hat + (z**2) / (2.0 * total)) / denom
    margin = (z / denom) * np.sqrt((p_hat * (1.0 - p_hat) / total) + (z**2) / (4.0 * (total**2)))

    lower = max(0.0, center - margin)
    upper = min(1.0, center + margin)
    if successes <= 0:
        lower = 0.0
    if successes >= total:
        upper = 1.0
    return (float(lower), float(upper))


def evaluate_binary_group_metrics(
    y_true: Union[np.ndarray, pd.Series, List[Any]],
    y_pred: Union[np.ndarray, pd.Series, List[Any]],
    sensitive_features: Union[np.ndarray, pd.Series, List[Any]],
    positive_label: Any = 1,
    reference_group: Optional[Any] = None,
    min_sample_threshold: int = 100,
) -> pd.DataFrame:
    """Evaluate detailed group-level classification and fairness metrics across sub-populations.

    Parameters
    ----------
    y_true : array-like
        Ground truth target labels.
    y_pred : array-like
        Predicted binary class labels.
    sensitive_features : array-like
        Group memberships / sensitive attribute values for each instance.
    positive_label : any, default=1
        Value representing the positive class (e.g. 1 or '>50K').
    reference_group : any, optional
        Baseline or majority group against which parity differences and ratios are computed.
    min_sample_threshold : int, default=100
        Sample size below which a warning flag (`small_sample_flag=True`) is raised.

    Returns
    -------
    pd.DataFrame
        Table indexed by group value containing counts, base rates, selection rates,
        error rates, precision, recall, and parity ratios.

    Raises
    ------
    ValueError
        If inputs have mismatched lengths, are empty, or contain invalid groups.
    """
    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)
    sens_arr = np.asarray(sensitive_features)

    # Validation: empty inputs
    if len(y_true_arr) == 0 or len(y_pred_arr) == 0 or len(sens_arr) == 0:
        raise ValueError("Cannot evaluate fairness on empty inputs.")

    # Validation: length matching
    if not (len(y_true_arr) == len(y_pred_arr) == len(sens_arr)):
        raise ValueError(
            f"Input length mismatch: y_true ({len(y_true_arr)}), "
            f"y_pred ({len(y_pred_arr)}), sensitive_features ({len(sens_arr)})."
        )

    # Standardize binary classes to 0 and 1
    y_true_bin = (y_true_arr == positive_label).astype(int)
    y_pred_bin = (y_pred_arr == positive_label).astype(int)

    unique_groups = pd.unique(sens_arr)
    # Remove nulls if any
    unique_groups = [g for g in unique_groups if pd.notna(g)]

    if len(unique_groups) == 0:
        raise ValueError("sensitive_features contains no valid non-null group values.")

    if reference_group is not None and reference_group not in unique_groups:
        raise ValueError(
            f"Specified reference_group '{reference_group}' not found in sensitive_features. "
            f"Available groups: {unique_groups}"
        )

    rows = []
    for group in unique_groups:
        mask = sens_arr == group
        n_group = int(np.sum(mask))

        if n_group == 0:
            continue

        y_t = y_true_bin[mask]
        y_p = y_pred_bin[mask]

        pos_actual = int(np.sum(y_t == 1))
        neg_actual = int(np.sum(y_t == 0))
        pos_pred = int(np.sum(y_p == 1))
        neg_pred = int(np.sum(y_p == 0))

        tp = int(np.sum((y_t == 1) & (y_p == 1)))
        fp = int(np.sum((y_t == 0) & (y_p == 1)))
        tn = int(np.sum((y_t == 0) & (y_p == 0)))
        fn = int(np.sum((y_t == 1) & (y_p == 0)))

        base_rate = _safe_divide(pos_actual, n_group)
        selection_rate = _safe_divide(pos_pred, n_group)
        tpr = _safe_divide(tp, pos_actual)
        fpr = _safe_divide(fp, neg_actual)
        fnr = _safe_divide(fn, pos_actual)
        tnr = _safe_divide(tn, neg_actual)
        precision = _safe_divide(tp, pos_pred)
        accuracy = _safe_divide(tp + tn, n_group)

        balanced_accuracy = (
            0.5 * (tpr + tnr)
            if not np.isnan(tpr) and not np.isnan(tnr)
            else np.nan
        )

        sel_ci_low, sel_ci_high = compute_wilson_score_interval(pos_pred, n_group)
        tpr_ci_low, tpr_ci_high = compute_wilson_score_interval(tp, pos_actual)

        rows.append({
            "group": group,
            "sample_count": n_group,
            "small_sample_flag": n_group < min_sample_threshold,
            "actual_positives": pos_actual,
            "predicted_positives": pos_pred,
            "true_positives": tp,
            "false_positives": fp,
            "true_negatives": tn,
            "false_negatives": fn,
            "base_rate": base_rate,
            "selection_rate": selection_rate,
            "selection_rate_ci_lower": sel_ci_low,
            "selection_rate_ci_upper": sel_ci_high,
            "true_positive_rate": tpr,
            "tpr_ci_lower": tpr_ci_low,
            "tpr_ci_upper": tpr_ci_high,
            "false_positive_rate": fpr,
            "precision": precision,
            "accuracy": accuracy,
            "balanced_accuracy": balanced_accuracy,
        })

    metrics_df = pd.DataFrame(rows).set_index("group")

    # If reference group is provided, compute relative parity metrics
    if reference_group is not None and reference_group in metrics_df.index:
        ref_sel_rate = metrics_df.loc[reference_group, "selection_rate"]
        ref_tpr = metrics_df.loc[reference_group, "true_positive_rate"]
        ref_fpr = metrics_df.loc[reference_group, "false_positive_rate"]

        metrics_df["disparate_impact_ratio"] = metrics_df["selection_rate"].apply(
            lambda sr: _safe_divide(sr, ref_sel_rate)
        )
        metrics_df["demographic_parity_diff"] = metrics_df["selection_rate"] - ref_sel_rate
        metrics_df["equal_opportunity_diff"] = metrics_df["true_positive_rate"] - ref_tpr
        metrics_df["equalized_odds_diff"] = metrics_df.apply(
            lambda row: max(
                abs(row["true_positive_rate"] - ref_tpr),
                abs(row["false_positive_rate"] - ref_fpr),
            )
            if not np.isnan(row["true_positive_rate"]) and not np.isnan(row["false_positive_rate"])
            else np.nan,
            axis=1,
        )

    return metrics_df


def audit_model_fairness(
    model: Any,
    dataset: pd.DataFrame,
    sensitive_columns: Dict[str, Optional[Any]],
    target_column: str = "income",
    positive_label: Any = ">50K",
    min_sample_threshold: int = 100,
) -> Dict[str, pd.DataFrame]:
    """Execute a complete fairness audit across multiple sensitive attributes for a fitted model.

    Parameters
    ----------
    model : fitted estimator or Pipeline
        The scikit-learn model artifact.
    dataset : pd.DataFrame
        Dataset containing both features, sensitive attributes, and target.
    sensitive_columns : dict
        Mapping of `{column_name: reference_group}`.
        Example: `{"sex": "Male", "race": "White"}`.
    target_column : str, default='income'
        Name of the target column in `dataset`.
    positive_label : any, default='>50K'
        The representation of the positive high-income class.
    min_sample_threshold : int, default=100
        Sample size below which a group is flagged for uncertainty.

    Returns
    -------
    dict of {str: pd.DataFrame}
        Fairness metrics tables for each sensitive column.

    Raises
    ------
    KeyError
        If target_column or any sensitive_column is missing from dataset.
    """
    if target_column not in dataset.columns:
        raise KeyError(f"Target column '{target_column}' not found in dataset.")

    for col in sensitive_columns:
        if col not in dataset.columns:
            raise KeyError(f"Sensitive column '{col}' not found in dataset.")

    X = dataset.drop(columns=[target_column])
    y_true = dataset[target_column]

    raw_preds = model.predict(X)

    # Standardize predictions: handle models outputting 0/1 vs class labels
    if hasattr(model, "classes_") and len(model.classes_) == 2:
        classes = list(model.classes_)
        # If classes are [0, 1] and positive_label is '>50K', check if 1 corresponds to positive
        if positive_label not in classes and 1 in classes:
            # Map predictions: 1 is positive
            y_pred = raw_preds
            y_true_mapped = (y_true == positive_label).astype(int)
            pos_eval_label = 1
        else:
            y_pred = raw_preds
            y_true_mapped = y_true
            pos_eval_label = positive_label
    else:
        y_pred = raw_preds
        y_true_mapped = y_true
        pos_eval_label = positive_label

    results = {}
    for col, ref_group in sensitive_columns.items():
        sens_vals = dataset[col]
        metrics_df = evaluate_binary_group_metrics(
            y_true=y_true_mapped,
            y_pred=y_pred,
            sensitive_features=sens_vals,
            positive_label=pos_eval_label,
            reference_group=ref_group,
            min_sample_threshold=min_sample_threshold,
        )
        results[col] = metrics_df

    return results


def main() -> None:
    """Run standard fairness audit on test split with the production model."""
    from pathlib import Path
    import joblib
    from .data import load_test_data

    model_path = Path(__file__).resolve().parents[2] / "models" / "random_forest_final.joblib"
    if not model_path.is_file():
        raise FileNotFoundError(f"Model not found at {model_path}")

    model = joblib.load(model_path)
    df = load_test_data()

    # Create standard demographic audit cohorts
    df["age_group"] = pd.cut(
        df["age"],
        bins=[-np.inf, 24, 44, 64, np.inf],
        labels=["<25", "25-44", "45-64", "65+"],
    )
    df["country_group"] = np.where(
        df["native-country"] == "United-States", "US", "Non-US"
    )

    sensitive_specs = {
        "sex": "Male",
        "race": "White",
        "age_group": "25-44",
        "country_group": "US",
    }

    results = audit_model_fairness(
        model=model,
        dataset=df,
        sensitive_columns=sensitive_specs,
        target_column="income",
        positive_label=">50K",
    )

    print("\n==========================================")
    print("INCOMELENS-AI GROUP FAIRNESS AUDIT REPORT")
    print("==========================================")

    display_cols = [
        "sample_count",
        "base_rate",
        "selection_rate",
        "true_positive_rate",
        "false_positive_rate",
        "precision",
        "disparate_impact_ratio",
    ]

    all_rows = []
    for attr, metrics_table in results.items():
        print(f"\n--- SENSITIVE ATTRIBUTE: {attr.upper()} (Reference: {sensitive_specs[attr]}) ---")
        available_cols = [c for c in display_cols if c in metrics_table.columns]
        print(metrics_table[available_cols].round(4).to_string())

        df_attr = metrics_table.reset_index().rename(columns={"group": "subgroup"})
        df_attr["sensitive_attribute"] = attr
        df_attr["reference_group"] = sensitive_specs[attr]

        ref_prec = df_attr.loc[df_attr["subgroup"] == sensitive_specs[attr], "precision"].values[0] if (df_attr["subgroup"] == sensitive_specs[attr]).any() else np.nan
        ref_fpr = df_attr.loc[df_attr["subgroup"] == sensitive_specs[attr], "false_positive_rate"].values[0] if (df_attr["subgroup"] == sensitive_specs[attr]).any() else np.nan

        df_attr["predictive_equality_diff"] = df_attr["false_positive_rate"] - ref_fpr
        df_attr["predictive_parity_diff"] = df_attr["precision"] - ref_prec

        df_attr["unstable_estimate_flag"] = (df_attr["sample_count"] < 100) | (df_attr["actual_positives"] < 20)
        df_attr["unstable_estimate_note"] = "Stable"
        for idx, row in df_attr.iterrows():
            notes = []
            if row["sample_count"] < 100:
                notes.append(f"N={int(row['sample_count'])} < 100")
            if row["actual_positives"] < 20:
                notes.append(f"Actual positives={int(row['actual_positives'])} < 20")
            if notes:
                df_attr.at[idx, "unstable_estimate_note"] = "; ".join(notes)

        all_rows.append(df_attr)

    output_df = pd.concat(all_rows, ignore_index=True)
    canonical_cols = [
        "sensitive_attribute", "subgroup", "reference_group",
        "sample_count", "actual_positives", "predicted_positives",
        "base_rate", "selection_rate", "selection_rate_ci_lower", "selection_rate_ci_upper",
        "true_positive_rate", "tpr_ci_lower", "tpr_ci_upper",
        "false_positive_rate", "precision", "balanced_accuracy",
        "disparate_impact_ratio", "demographic_parity_diff", "equal_opportunity_diff",
        "predictive_equality_diff", "predictive_parity_diff",
        "unstable_estimate_flag", "unstable_estimate_note",
    ]
    available_canonical = [c for c in canonical_cols if c in output_df.columns]
    output_df = output_df[available_canonical]

    # Save to reports/fairness_metrics.csv
    report_dir = Path(__file__).resolve().parents[2] / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    csv_path = report_dir / "fairness_metrics.csv"
    output_df.round(4).to_csv(csv_path, index=False)

    # Post-generation verification
    if not csv_path.is_file() or csv_path.stat().st_size == 0:
        raise RuntimeError(f"Fairness metrics CSV missing or empty: {csv_path}")

    print(f"\nSaved fairness metrics: {csv_path}")
    print("Verified: Generated fairness report artifact is valid and non-empty.")
    print("Fairness audit completed successfully.")


if __name__ == "__main__":
    main()

