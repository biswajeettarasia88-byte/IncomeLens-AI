"""Model training pipelines and persistence for IncomeLens AI."""

import logging
from pathlib import Path
from typing import Optional, Union
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from .data import (
    TRAIN_DATA_PATH,
    VALIDATION_DATA_PATH,
    load_train_data,
    load_validation_data,
)
from .preprocessing import build_preprocessor, split_features_target

logger = logging.getLogger(__name__)

ROOT_DIR = Path(__file__).resolve().parents[2]
MODEL_DIR = ROOT_DIR / "models"
FINAL_MODEL_PATH = MODEL_DIR / "random_forest_final.joblib"


def build_logistic_regression_pipeline(
    max_iter: int = 1000,
    random_state: int = 42,
    class_weight: Optional[Union[dict, str]] = None,
) -> Pipeline:
    """Build the baseline Logistic Regression pipeline with preprocessing.

    Parameters
    ----------
    max_iter : int, default=1000
        Maximum solver iterations.
    random_state : int, default=42
        Reproducibility seed.
    class_weight : dict or 'balanced', optional
        Weights associated with classes.

    Returns
    -------
    Pipeline
        Unfitted Pipeline with ColumnTransformer preprocessor and LogisticRegression.
    """
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor(sparse_output=True)),
            (
                "classifier",
                LogisticRegression(
                    max_iter=max_iter,
                    random_state=random_state,
                    class_weight=class_weight,
                ),
            ),
        ]
    )


def build_random_forest_pipeline(
    n_estimators: int = 150,
    min_samples_leaf: int = 2,
    random_state: int = 42,
    n_jobs: int = -1,
) -> Pipeline:
    """Build the canonical Random Forest pipeline with preprocessing.

    Parameters
    ----------
    n_estimators : int, default=150
        Number of decision trees in forest.
    min_samples_leaf : int, default=2
        Minimum samples per leaf.
    random_state : int, default=42
        Reproducibility seed.
    n_jobs : int, default=-1
        Parallel CPU threads.

    Returns
    -------
    Pipeline
        Unfitted Pipeline with ColumnTransformer preprocessor and RandomForestClassifier.
    """
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor(sparse_output=True)),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=n_estimators,
                    min_samples_leaf=min_samples_leaf,
                    random_state=random_state,
                    n_jobs=n_jobs,
                ),
            ),
        ]
    )


def train_final_model(
    train_df: Optional[pd.DataFrame] = None,
    val_df: Optional[pd.DataFrame] = None,
    save_path: Optional[Path] = None,
) -> Pipeline:
    """Train the production model on the development set (train + val).

    Parameters
    ----------
    train_df : pd.DataFrame, optional
        Training split. If None, loads from data/processed/train.csv.
    val_df : pd.DataFrame, optional
        Validation split. If None, loads from data/processed/validation.csv.
    save_path : Path, optional
        Destination to save serialized model pipeline.

    Returns
    -------
    Pipeline
        Fitted model pipeline.
    """
    if train_df is not None and (not isinstance(train_df, pd.DataFrame) or train_df.empty):
        logger.error("Provided train_df is invalid or empty.")
        raise ValueError("Provided train_df must be a non-empty pandas DataFrame.")
    if val_df is not None and (not isinstance(val_df, pd.DataFrame) or val_df.empty):
        logger.error("Provided val_df is invalid or empty.")
        raise ValueError("Provided val_df must be a non-empty pandas DataFrame.")

    if train_df is None:
        train_df = load_train_data()
    if val_df is None:
        val_df = load_validation_data()

    dev_df = pd.concat([train_df, val_df], ignore_index=True)
    X_dev, y_dev = split_features_target(dev_df)

    logger.info(
        "Training Random Forest production pipeline on development set: %d rows (train=%d, val=%d), %d features",
        len(dev_df),
        len(train_df),
        len(val_df),
        X_dev.shape[1],
    )

    model = build_random_forest_pipeline()
    model.fit(X_dev, y_dev)
    logger.info("Random Forest pipeline fitted successfully.")

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, save_path)
        logger.info("Saved trained model artifact to: %s", save_path)

    return model
