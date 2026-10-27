"""Focused unit tests for incomelens.training module.

Verifies:
1. Train/test separation is respected (training never accesses held-out test data).
2. Training uses only permitted training data (preprocessor statistics derive exclusively from train/val).
3. Evaluation does not refit or mutate the model.
4. Fixed random seeds produce reproducible behavior where expected (Random Forest and Logistic Regression).
5. Hyperparameter configuration and model serialization.
"""

from pathlib import Path
import sys
from unittest.mock import patch

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
    TRAIN_DATA_PATH,
    VALIDATION_DATA_PATH,
    TEST_DATA_PATH,
)
from incomelens.evaluation import evaluate_predictions
from incomelens.preprocessing import split_features_target
from incomelens.training import (
    build_logistic_regression_pipeline,
    build_random_forest_pipeline,
    train_final_model,
)


# --- Synthetic Fixtures for Fast Training ---

def _create_synthetic_batch(n_samples: int, seed: int, age_offset: int = 0) -> pd.DataFrame:
    """Generate reproducible small synthetic census DataFrames."""
    rng = np.random.RandomState(seed)
    return pd.DataFrame({
        "age": rng.randint(20, 60, size=n_samples) + age_offset,
        "workclass": rng.choice(["Private", "Self-emp-inc", "State-gov"], size=n_samples),
        "fnlwgt": rng.randint(50000, 250000, size=n_samples),
        "education": rng.choice(["HS-grad", "Bachelors", "Masters", "Doctorate"], size=n_samples),
        "education-num": rng.randint(9, 16, size=n_samples),
        "marital-status": rng.choice(["Never-married", "Married-civ-spouse", "Divorced"], size=n_samples),
        "occupation": rng.choice(["Exec-managerial", "Prof-specialty", "Sales", "Adm-clerical"], size=n_samples),
        "relationship": rng.choice(["Husband", "Wife", "Not-in-family", "Own-child"], size=n_samples),
        "race": rng.choice(["White", "Black", "Asian-Pac-Islander"], size=n_samples),
        "sex": rng.choice(["Male", "Female"], size=n_samples),
        "capital-gain": rng.choice([0, 0, 0, 5000, 14000], size=n_samples),
        "capital-loss": rng.choice([0, 0, 0, 1000], size=n_samples),
        "hours-per-week": rng.randint(20, 55, size=n_samples),
        "native-country": rng.choice(["United-States", "Canada", "Germany"], size=n_samples),
        "income": ["<=50K" if i % 2 == 0 else ">50K" for i in range(n_samples)],
    })


@pytest.fixture
def synthetic_train_df() -> pd.DataFrame:
    """Small 20-row synthetic training split."""
    return _create_synthetic_batch(n_samples=20, seed=101)


@pytest.fixture
def synthetic_val_df() -> pd.DataFrame:
    """Small 10-row synthetic validation split."""
    return _create_synthetic_batch(n_samples=10, seed=202)


@pytest.fixture
def synthetic_test_df() -> pd.DataFrame:
    """Small 10-row synthetic test split containing an extreme numeric outlier to detect leakage."""
    df = _create_synthetic_batch(n_samples=10, seed=303)
    # Inject extreme outlier into test set
    df.loc[0, "age"] = 999
    return df


# --- 1. Train/Test Separation Tests ---

def test_train_final_model_never_accesses_test_data(synthetic_train_df, synthetic_val_df):
    """Verify train_final_model never imports, loads, or touches the held-out test split."""
    with patch("incomelens.training.load_train_data", return_value=synthetic_train_df):
        with patch("incomelens.training.load_validation_data", return_value=synthetic_val_df):
            with patch("incomelens.data.load_test_data") as mock_load_test:
                model = train_final_model(save_path=None)

                # Training must NOT call load_test_data
                mock_load_test.assert_not_called()
                assert isinstance(model, Pipeline)
                assert hasattr(model, "classes_")


def test_train_module_does_not_reference_test_path():
    """Verify incomelens.training module source does not import or reference TEST_DATA_PATH."""
    import incomelens.training as training_mod

    # Confirm TEST_DATA_PATH and load_test_data are not exposed or imported
    assert not hasattr(training_mod, "TEST_DATA_PATH")
    assert not hasattr(training_mod, "load_test_data")


# --- 2. Permitted Training Data Usage Tests ---

def test_training_preprocessor_learns_only_from_dev_data(synthetic_train_df, synthetic_val_df, synthetic_test_df):
    """Verify preprocessing parameters (e.g. StandardScaler mean) derive strictly from train+val without test leakage."""
    dev_df = pd.concat([synthetic_train_df, synthetic_val_df], ignore_index=True)

    # Train model on synthetic development data
    model = train_final_model(train_df=synthetic_train_df, val_df=synthetic_val_df, save_path=None)

    # Extract fitted StandardScaler inside pipeline
    preprocessor = model.named_steps["preprocessor"]
    transformers_dict = dict([(name, (pipe, cols)) for name, pipe, cols in preprocessor.transformers_])
    scaler = transformers_dict["numeric"][0].named_steps["scaler"]

    # Calculate expected empirical mean from permitted dev data (imputer runs first with median)
    num_cols = transformers_dict["numeric"][1]
    expected_means = dev_df[num_cols].median().values  # After SimpleImputer(strategy='median')
    # Or check empirical mean directly from transformed numeric values
    imputer = transformers_dict["numeric"][0].named_steps["imputer"]
    dev_imputed = imputer.transform(dev_df[num_cols])
    expected_scaler_mean = np.mean(dev_imputed, axis=0)

    np.testing.assert_allclose(scaler.mean_, expected_scaler_mean, atol=1e-5)

    # Check that the test outlier (age=999) did NOT affect the fitted scaler
    age_idx = list(num_cols).index("age")
    assert scaler.mean_[age_idx] < 100.0, f"Scaler mean contaminated by test outlier: {scaler.mean_[age_idx]}"


# --- 3. Evaluation Does Not Refit the Model Tests ---

def test_evaluation_does_not_refit_or_mutate_model(synthetic_train_df, synthetic_test_df):
    """Verify running model evaluation and predict does NOT alter internal model parameters or tree pointers."""
    X_train, y_train = split_features_target(synthetic_train_df)
    X_test, y_test = split_features_target(synthetic_test_df)

    # Train lightweight random forest
    model = build_random_forest_pipeline(n_estimators=5, random_state=42)
    model.fit(X_train, y_train)

    clf = model.named_steps["classifier"]

    # Record internal estimator tree object memory IDs
    tree_ids_before = [id(tree) for tree in clf.estimators_]
    n_features_before = clf.n_features_in_
    classes_before = np.copy(clf.classes_)

    # Run predictions and evaluation
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    metrics, cm_df, report = evaluate_predictions(y_test, y_pred, y_prob)

    # Assert model internal state is strictly unmodified
    tree_ids_after = [id(tree) for tree in clf.estimators_]
    n_features_after = clf.n_features_in_
    classes_after = clf.classes_

    assert tree_ids_before == tree_ids_after, "Model trees were re-allocated / refit during evaluation!"
    assert n_features_before == n_features_after
    assert np.array_equal(classes_before, classes_after)


# --- 4. Fixed Random Seeds Produce Reproducible Behavior Tests ---

def test_fixed_random_seeds_reproducible_random_forest(synthetic_train_df, synthetic_test_df):
    """Verify identical random_state in RandomForest produces identical predictions, while differing seeds vary."""
    X_train, y_train = split_features_target(synthetic_train_df)
    X_test, _ = split_features_target(synthetic_test_df)

    # Model 1 with seed 42
    rf1 = build_random_forest_pipeline(n_estimators=10, random_state=42)
    rf1.fit(X_train, y_train)
    preds1 = rf1.predict(X_test)
    probs1 = rf1.predict_proba(X_test)

    # Model 2 with identical seed 42
    rf2 = build_random_forest_pipeline(n_estimators=10, random_state=42)
    rf2.fit(X_train, y_train)
    preds2 = rf2.predict(X_test)
    probs2 = rf2.predict_proba(X_test)

    # Assert bit-for-bit equivalence
    assert np.array_equal(preds1, preds2), "Random forest predictions varied with fixed random_state=42"
    assert np.allclose(probs1, probs2, atol=1e-7), "Random forest probabilities varied with fixed random_state=42"

    # Model 3 with differing seed 999
    rf3 = build_random_forest_pipeline(n_estimators=10, random_state=999)
    rf3.fit(X_train, y_train)
    probs3 = rf3.predict_proba(X_test)

    # Verify seed actually affected tree subspace sampling
    assert not np.array_equal(probs1, probs3), "Changing random_state did not alter random forest trees"


def test_fixed_random_seeds_reproducible_logistic_regression(synthetic_train_df, synthetic_test_df):
    """Verify LogisticRegression coefficients and predictions are reproducible with fixed seed."""
    X_train, y_train = split_features_target(synthetic_train_df)
    X_test, _ = split_features_target(synthetic_test_df)

    lr1 = build_logistic_regression_pipeline(max_iter=200, random_state=42)
    lr1.fit(X_train, y_train)

    lr2 = build_logistic_regression_pipeline(max_iter=200, random_state=42)
    lr2.fit(X_train, y_train)

    # Coefficients must match exactly
    clf1 = lr1.named_steps["classifier"]
    clf2 = lr2.named_steps["classifier"]

    assert np.array_equal(clf1.coef_, clf2.coef_)
    assert np.array_equal(clf1.intercept_, clf2.intercept_)

    # Predictions must match
    assert np.array_equal(lr1.predict(X_test), lr2.predict(X_test))


# --- 5. Hyperparameter Configuration and Model Serialization Tests ---

def test_pipeline_builders_hyperparameter_propagation():
    """Verify pipeline builders propagate hyperparameters directly to their underlying estimators."""
    rf_pipe = build_random_forest_pipeline(
        n_estimators=25,
        min_samples_leaf=4,
        random_state=123,
        n_jobs=2,
    )
    rf_clf = rf_pipe.named_steps["classifier"]
    assert rf_clf.n_estimators == 25
    assert rf_clf.min_samples_leaf == 4
    assert rf_clf.random_state == 123
    assert rf_clf.n_jobs == 2

    lr_pipe = build_logistic_regression_pipeline(
        max_iter=500,
        random_state=321,
        class_weight="balanced",
    )
    lr_clf = lr_pipe.named_steps["classifier"]
    assert lr_clf.max_iter == 500
    assert lr_clf.random_state == 321
    assert lr_clf.class_weight == "balanced"


def test_train_final_model_serialization(tmp_path, synthetic_train_df, synthetic_val_df):
    """Verify train_final_model serializes a functional pipeline that can be reloaded."""
    custom_save = tmp_path / "models" / "synthetic_model.joblib"

    # Fit and serialize
    model = train_final_model(
        train_df=synthetic_train_df,
        val_df=synthetic_val_df,
        save_path=custom_save,
    )

    assert custom_save.is_file()

    # Reload from disk
    loaded = joblib.load(custom_save)
    assert isinstance(loaded, Pipeline)
    assert list(loaded.classes_) == [0, 1]

    # Verify predict on sample
    sample_X, _ = split_features_target(synthetic_val_df)
    preds = loaded.predict(sample_X)
    assert len(preds) == len(synthetic_val_df)
    assert set(preds).issubset({0, 1})
