"""Tests for incomelens.fairness module and group-level disparity auditing."""

import sys
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from incomelens.fairness import (
    _safe_divide,
    audit_model_fairness,
    compute_wilson_score_interval,
    evaluate_binary_group_metrics,
)

EXPECTED_GROUP_METRIC_COLUMNS_WITH_REF = [
    "sample_count",
    "small_sample_flag",
    "actual_positives",
    "predicted_positives",
    "true_positives",
    "false_positives",
    "true_negatives",
    "false_negatives",
    "base_rate",
    "selection_rate",
    "selection_rate_ci_lower",
    "selection_rate_ci_upper",
    "true_positive_rate",
    "tpr_ci_lower",
    "tpr_ci_upper",
    "false_positive_rate",
    "precision",
    "accuracy",
    "balanced_accuracy",
    "disparate_impact_ratio",
    "demographic_parity_diff",
    "equal_opportunity_diff",
    "equalized_odds_diff",
]


def test_safe_divide():
    """Verify safe division with zero, nan, and valid denominators."""
    assert _safe_divide(10.0, 2.0) == 5.0
    assert np.isnan(_safe_divide(10.0, 0.0))
    assert _safe_divide(10.0, 0.0, fill_value=0.0) == 0.0
    assert np.isnan(_safe_divide(np.nan, 5.0))
    assert np.isnan(_safe_divide(5.0, np.nan))


def test_wilson_score_interval():
    """Verify Wilson score confidence interval calculation."""
    # 50 out of 100
    low, high = compute_wilson_score_interval(50, 100)
    assert 0.40 < low < 0.50
    assert 0.50 < high < 0.60
    assert low < high

    # Zero total
    low_zero, high_zero = compute_wilson_score_interval(0, 0)
    assert np.isnan(low_zero) and np.isnan(high_zero)

    # Extreme: 0 successes out of 100
    low_0, high_0 = compute_wilson_score_interval(0, 100)
    assert low_0 == 0.0
    assert high_0 > 0.0


def test_evaluate_binary_group_metrics_valid_calculations():
    """Verify exact calculation of selection rate, TPR, FPR, precision, and parity metrics."""
    # Group A: 10 samples
    # y_true: 4 positives, 6 negatives
    # y_pred: 3 positives, 7 negatives
    # TP=2, FP=1, FN=2, TN=5
    # Base rate = 0.4, Selection rate = 0.3, TPR = 0.5, FPR = 1/6, Precision = 2/3

    # Group B: 10 samples
    # y_true: 6 positives, 4 negatives
    # y_pred: 6 positives (TP=5, FP=1, FN=1, TN=3)
    # Base rate = 0.6, Selection rate = 0.6, TPR = 5/6, FPR = 0.25, Precision = 5/6

    y_true = [1, 1, 1, 1, 0, 0, 0, 0, 0, 0] + [1, 1, 1, 1, 1, 1, 0, 0, 0, 0]
    y_pred = [1, 1, 0, 0, 1, 0, 0, 0, 0, 0] + [1, 1, 1, 1, 1, 0, 1, 0, 0, 0]
    groups = ["A"] * 10 + ["B"] * 10

    df = evaluate_binary_group_metrics(
        y_true=y_true,
        y_pred=y_pred,
        sensitive_features=groups,
        positive_label=1,
        reference_group="B",
    )

    assert "A" in df.index and "B" in df.index
    # Group A metrics
    row_a = df.loc["A"]
    assert row_a["sample_count"] == 10
    assert row_a["actual_positives"] == 4
    assert row_a["predicted_positives"] == 3
    assert row_a["true_positives"] == 2
    assert row_a["false_positives"] == 1
    assert row_a["base_rate"] == 0.4
    assert row_a["selection_rate"] == 0.3
    assert np.isclose(row_a["true_positive_rate"], 0.5)
    assert np.isclose(row_a["false_positive_rate"], 1.0 / 6.0)
    assert np.isclose(row_a["precision"], 2.0 / 3.0)

    # Reference parity metrics (vs B)
    # DIR = 0.3 / 0.6 = 0.5
    assert np.isclose(row_a["disparate_impact_ratio"], 0.5)
    # Demographic Parity Diff = 0.3 - 0.6 = -0.3
    assert np.isclose(row_a["demographic_parity_diff"], -0.3)
    # Equal Opportunity Diff = 0.5 - (5/6) = -0.3333
    assert np.isclose(row_a["equal_opportunity_diff"], 0.5 - (5.0 / 6.0))


def test_evaluate_binary_group_metrics_empty_inputs_raises_error():
    """Verify empty inputs raise ValueError."""
    with pytest.raises(ValueError, match="empty inputs"):
        evaluate_binary_group_metrics([], [], [])


def test_evaluate_binary_group_metrics_mismatched_lengths_raises_error():
    """Verify mismatched input lengths raise ValueError."""
    with pytest.raises(ValueError, match="Input length mismatch"):
        evaluate_binary_group_metrics([1, 0], [1], ["A", "A"])


def test_evaluate_binary_group_metrics_invalid_reference_group_raises_error():
    """Verify non-existent reference group raises ValueError."""
    with pytest.raises(ValueError, match="Specified reference_group 'NonExistent' not found"):
        evaluate_binary_group_metrics([1, 0], [1, 0], ["A", "B"], reference_group="NonExistent")


def test_evaluate_binary_group_metrics_handles_zero_positives_gracefully():
    """Verify no ZeroDivisionError when actual positives or predicted positives are zero."""
    # Group with 0 actual positives and 0 predicted positives
    y_true = [0, 0, 0, 0]
    y_pred = [0, 0, 0, 0]
    groups = ["A", "A", "A", "A"]

    df = evaluate_binary_group_metrics(y_true, y_pred, groups, positive_label=1)
    row = df.loc["A"]

    assert row["sample_count"] == 4
    assert row["selection_rate"] == 0.0
    assert row["base_rate"] == 0.0
    # TPR and Precision should be NaN due to 0/0, not crash
    assert np.isnan(row["true_positive_rate"])
    assert np.isnan(row["precision"])
    assert row["false_positive_rate"] == 0.0


def test_audit_model_fairness_missing_columns_raises_error():
    """Verify missing target or sensitive column raises KeyError."""
    dummy_df = pd.DataFrame({"age": [25, 30], "sex": ["Male", "Female"]})

    # Missing target
    with pytest.raises(KeyError, match="Target column 'income' not found"):
        audit_model_fairness(model=None, dataset=dummy_df, sensitive_columns={"sex": "Male"}, target_column="income")

    # Missing sensitive column
    dummy_df["income"] = ["<=50K", ">50K"]
    with pytest.raises(KeyError, match="Sensitive column 'race' not found"):
        audit_model_fairness(model=None, dataset=dummy_df, sensitive_columns={"race": "White"}, target_column="income")


def test_audit_model_fairness_production_model_on_test_data():
    """End-to-end integration test with production model artifact and processed test set."""
    model_path = ROOT / "models" / "random_forest_final.joblib"
    test_path = ROOT / "data" / "processed" / "test.csv"

    if not model_path.is_file() or not test_path.is_file():
        pytest.skip("Production model or test data not found.")

    model = joblib.load(model_path)
    test_df = pd.read_csv(test_path)

    results = audit_model_fairness(
        model=model,
        dataset=test_df,
        sensitive_columns={"sex": "Male", "race": "White"},
        target_column="income",
        positive_label=">50K",
    )

    assert "sex" in results
    assert "race" in results

    sex_df = results["sex"]
    assert "Male" in sex_df.index and "Female" in sex_df.index
    assert sex_df.loc["Male", "sample_count"] == 6535
    assert sex_df.loc["Female", "sample_count"] == 3236

    # Verify female selection rate is markedly lower than male (disparate impact)
    assert sex_df.loc["Female", "selection_rate"] < sex_df.loc["Male", "selection_rate"]
    assert sex_df.loc["Female", "disparate_impact_ratio"] < 0.5


# --- New Focused Fairness Unit Tests ---

def test_fairness_group_counts_and_labels_multi_group():
    """Verify exact group counts, marginal sum conservation, and index labels across 4 groups."""
    # 4 groups with sizes: G1=20, G2=15, G3=30, G4=35 (total = 100)
    groups = ["G1"] * 20 + ["G2"] * 15 + ["G3"] * 30 + ["G4"] * 35
    rng = np.random.RandomState(42)
    y_true = rng.choice([0, 1], size=100, p=[0.7, 0.3])
    y_pred = rng.choice([0, 1], size=100, p=[0.75, 0.25])

    df = evaluate_binary_group_metrics(
        y_true=y_true,
        y_pred=y_pred,
        sensitive_features=groups,
        positive_label=1,
        reference_group="G1",
    )

    # 1. Labels match sensitive values exactly
    assert set(df.index) == {"G1", "G2", "G3", "G4"}

    # 2. Group counts match exactly
    assert df.loc["G1", "sample_count"] == 20
    assert df.loc["G2", "sample_count"] == 15
    assert df.loc["G3", "sample_count"] == 30
    assert df.loc["G4", "sample_count"] == 35

    # 3. Sum of sample counts equals total length
    assert df["sample_count"].sum() == 100

    # 4. Conservation of counts per group
    for g in ["G1", "G2", "G3", "G4"]:
        row = df.loc[g]
        assert row["true_positives"] + row["false_negatives"] == row["actual_positives"]
        assert row["true_positives"] + row["false_positives"] == row["predicted_positives"]
        assert (
            row["true_positives"] + row["false_positives"] + row["true_negatives"] + row["false_negatives"]
            == row["sample_count"]
        )


def test_metric_calculation_on_known_synthetic_dataset():
    """Verify exact calculation of selection rate, TPR, FPR, precision, and disparity differences on known data."""
    # Known 2-group scenario:
    # Reference Group (Ref): N=40, y_true has 20 pos (1) and 20 neg (0)
    # y_pred has: TP=16, FP=4, FN=4, TN=16
    #   base_rate = 20/40 = 0.50
    #   selection_rate = 20/40 = 0.50
    #   tpr = 16/20 = 0.80
    #   fpr = 4/20 = 0.20
    #   precision = 16/20 = 0.80

    # Subgroup (Sub): N=40, y_true has 10 pos (1) and 30 neg (0)
    # y_pred has: TP=6, FP=4, FN=4, TN=26
    #   base_rate = 10/40 = 0.25
    #   selection_rate = 10/40 = 0.25
    #   tpr = 6/10 = 0.60
    #   fpr = 4/30 = 0.13333
    #   precision = 6/10 = 0.60

    # Disparity vs Ref:
    #   disparate_impact_ratio = 0.25 / 0.50 = 0.50
    #   demographic_parity_diff = 0.25 - 0.50 = -0.25
    #   equal_opportunity_diff = 0.60 - 0.80 = -0.20
    #   equalized_odds_diff = max(|0.60 - 0.80|, |4/30 - 0.20|) = max(0.20, 0.0667) = 0.20

    y_true_ref = [1] * 20 + [0] * 20
    y_pred_ref = [1] * 16 + [0] * 4 + [1] * 4 + [0] * 16

    y_true_sub = [1] * 10 + [0] * 30
    y_pred_sub = [1] * 6 + [0] * 4 + [1] * 4 + [0] * 26

    y_true = y_true_ref + y_true_sub
    y_pred = y_pred_ref + y_pred_sub
    groups = ["Ref"] * 40 + ["Sub"] * 40

    df = evaluate_binary_group_metrics(
        y_true=y_true,
        y_pred=y_pred,
        sensitive_features=groups,
        positive_label=1,
        reference_group="Ref",
    )

    sub_row = df.loc["Sub"]
    assert sub_row["base_rate"] == 0.25
    assert sub_row["selection_rate"] == 0.25
    assert np.isclose(sub_row["true_positive_rate"], 0.60)
    assert np.isclose(sub_row["false_positive_rate"], 4.0 / 30.0)
    assert np.isclose(sub_row["precision"], 0.60)

    # Parity comparisons
    assert np.isclose(sub_row["disparate_impact_ratio"], 0.50)
    assert np.isclose(sub_row["demographic_parity_diff"], -0.25)
    assert np.isclose(sub_row["equal_opportunity_diff"], -0.20)
    assert np.isclose(sub_row["equalized_odds_diff"], 0.20)


def test_missing_and_empty_groups_handling():
    """Verify missing/null group values are filtered out safely, and all-null raises descriptive error."""
    # Group with nulls mixed in
    y_true = [1, 0, 1, 0, 1]
    y_pred = [1, 0, 0, 0, 1]
    groups_with_null = ["GroupA", "GroupB", None, np.nan, "GroupA"]

    # Nulls should be filtered out, evaluating only non-null groups (GroupA and GroupB)
    df = evaluate_binary_group_metrics(
        y_true=y_true,
        y_pred=y_pred,
        sensitive_features=groups_with_null,
        positive_label=1,
        reference_group="GroupA",
    )
    assert set(df.index) == {"GroupA", "GroupB"}
    assert df.loc["GroupA", "sample_count"] == 2
    assert df.loc["GroupB", "sample_count"] == 1

    # All-null sensitive features raise ValueError
    all_nulls = [None, np.nan, None]
    with pytest.raises(ValueError, match="sensitive_features contains no valid non-null group values"):
        evaluate_binary_group_metrics([1, 0, 1], [1, 0, 1], all_nulls)


def test_zero_denominator_safe_division_and_metrics():
    """Verify zero denominators across reference rates, TPR, and precision yield NaN without crashing."""
    # 1. Reference group has 0 selection rate (no positive predictions)
    y_true = [1, 0, 1, 0]
    y_pred = [0, 0, 1, 0]
    groups = ["RefZero", "RefZero", "Sub", "Sub"]

    df = evaluate_binary_group_metrics(
        y_true=y_true,
        y_pred=y_pred,
        sensitive_features=groups,
        positive_label=1,
        reference_group="RefZero",
    )

    assert df.loc["RefZero", "selection_rate"] == 0.0
    # Disparate impact ratio with zero denominator in reference selection rate produces NaN
    assert np.isnan(df.loc["Sub", "disparate_impact_ratio"])

    # 2. Group with zero actual negatives: FPR is NaN without ZeroDivisionError
    y_true_all_pos = [1, 1, 1]
    y_pred_pos = [1, 1, 0]
    df_all_pos = evaluate_binary_group_metrics(y_true_all_pos, y_pred_pos, ["AllPos"] * 3)
    assert np.isnan(df_all_pos.loc["AllPos", "false_positive_rate"])
    assert df_all_pos.loc["AllPos", "true_positive_rate"] == pytest.approx(2 / 3)


def test_fairness_output_table_schema_stability():
    """Verify evaluate_binary_group_metrics returns stable DataFrame schema with all expected columns."""
    y_true = [1, 0, 1, 0]
    y_pred = [1, 0, 0, 0]
    groups = ["A", "A", "B", "B"]

    # With reference group
    df_with_ref = evaluate_binary_group_metrics(
        y_true, y_pred, groups, positive_label=1, reference_group="A"
    )
    assert list(df_with_ref.columns) == EXPECTED_GROUP_METRIC_COLUMNS_WITH_REF

    # Verify column datatypes
    assert df_with_ref["sample_count"].dtype in [np.int64, int]
    assert df_with_ref["small_sample_flag"].dtype == bool
    assert df_with_ref["selection_rate"].dtype in [np.float64, float]

    # Without reference group: parity diff columns should be absent
    df_no_ref = evaluate_binary_group_metrics(y_true, y_pred, groups, positive_label=1)
    for parity_col in ["disparate_impact_ratio", "demographic_parity_diff", "equal_opportunity_diff"]:
        assert parity_col not in df_no_ref.columns


def test_audit_model_fairness_synthetic_model():
    """Verify audit_model_fairness end-to-end on small synthetic pipeline and multi-attribute dataset (~20ms)."""
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline

    # Create small synthetic dataset (20 samples, 2 sensitive columns)
    rng = np.random.RandomState(42)
    df_synthetic = pd.DataFrame({
        "feature_1": rng.uniform(0, 10, size=20),
        "feature_2": rng.uniform(0, 10, size=20),
        "gender": ["Male", "Female"] * 10,
        "region": ["North", "South", "East", "West"] * 5,
        "income": [">50K" if i % 3 == 0 else "<=50K" for i in range(20)],
    })

    # Fit small synthetic Pipeline with preprocessor that selects predictor features
    preprocessor = ColumnTransformer(
        transformers=[("num", "passthrough", ["feature_1", "feature_2"])],
        remainder="drop",
    )
    rf = RandomForestClassifier(n_estimators=3, max_depth=2, random_state=42)
    pipeline = Pipeline([("preprocessor", preprocessor), ("classifier", rf)])

    X_train = df_synthetic.drop(columns=["income"])
    y_train = (df_synthetic["income"] == ">50K").astype(int)
    pipeline.fit(X_train, y_train)

    # Run audit on both attributes
    results = audit_model_fairness(
        model=pipeline,
        dataset=df_synthetic,
        sensitive_columns={"gender": "Male", "region": "North"},
        target_column="income",
        positive_label=">50K",
    )

    assert "gender" in results
    assert "region" in results

    gender_df = results["gender"]
    assert set(gender_df.index) == {"Male", "Female"}
    assert gender_df.loc["Male", "sample_count"] == 10
    assert gender_df.loc["Female", "sample_count"] == 10

    region_df = results["region"]
    assert set(region_df.index) == {"North", "South", "East", "West"}
    assert region_df.loc["North", "sample_count"] == 5
