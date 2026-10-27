"""Preprocessing pipelines and feature transformations for IncomeLens AI."""

import logging
from typing import List, Tuple
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

logger = logging.getLogger(__name__)

TARGET_COLUMN = "income"

NUMERIC_FEATURES = [
    "age",
    "fnlwgt",
    "education-num",
    "capital-gain",
    "capital-loss",
    "hours-per-week",
]

CATEGORICAL_FEATURES = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]


def build_preprocessor(sparse_output: bool = True) -> ColumnTransformer:
    """Create a preprocessing pipeline for mixed-type census data.

    Parameters
    ----------
    sparse_output : bool, default=True
        Whether the OneHotEncoder should return a sparse matrix.
        Matches the production model pipeline configuration.

    Returns
    -------
    ColumnTransformer
        Unfitted ColumnTransformer with numerical and categorical sub-pipelines.
    """
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=sparse_output,
                ),
            ),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, NUMERIC_FEATURES),
            ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )


def extract_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extract and validate the required feature matrix for inference or evaluation.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with census features.

    Returns
    -------
    pd.DataFrame
        DataFrame containing only the expected feature columns in canonical order.

    Raises
    ------
    TypeError
        If df is not a pandas DataFrame.
    ValueError
        If df is missing any required feature columns.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"Expected pandas DataFrame, got {type(df).__name__}")

    expected = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    missing = [col for col in expected if col not in df.columns]
    if missing:
        raise ValueError(
            f"DataFrame is missing required feature columns: {missing}. Expected: {expected}"
        )

    return df[expected].copy()


def split_features_target(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """Separate predictors from the binary income target.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing predictors and the target column.

    Returns
    -------
    X : pd.DataFrame
        Feature matrix containing NUMERIC_FEATURES + CATEGORICAL_FEATURES.
    y : pd.Series
        Binary integer labels (0 for <=50K, 1 for >50K).

    Raises
    ------
    ValueError
        If target column is missing, or contains unmappable / null values.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"Expected pandas DataFrame, got {type(df).__name__}")

    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Target column '{TARGET_COLUMN}' not found in DataFrame.")

    target_series = df[TARGET_COLUMN]

    # Support pre-encoded integer/boolean targets
    if pd.api.types.is_numeric_dtype(target_series):
        unique_vals = set(target_series.dropna().unique())
        if not unique_vals.issubset({0, 1}):
            raise ValueError(f"Unexpected numeric income labels: {unique_vals}")
        if target_series.isna().any():
            raise ValueError("Target column contains missing values.")
        y = target_series.astype(int)
    else:
        # String labels: strip whitespace and trailing dots (e.g. '<=50K.' -> '<=50K')
        y_str = target_series.astype(str).str.strip().str.rstrip(".")
        y = y_str.map({"<=50K": 0, ">50K": 1})

        if y.isna().any():
            invalid = set(target_series[y.isna()].unique())
            raise ValueError(f"Unexpected or missing income labels: {invalid}")
        y = y.astype(int)

    X = extract_features(df)
    return X, y


def get_feature_names(preprocessor: ColumnTransformer) -> List[str]:
    """Extract output feature names from a fitted ColumnTransformer.

    Parameters
    ----------
    preprocessor : ColumnTransformer
        Fitted ColumnTransformer.

    Returns
    -------
    list of str
        Names of all output features after transformation.
    """
    try:
        return [str(name) for name in preprocessor.get_feature_names_out()]
    except Exception as e:
        raise ValueError(f"Could not extract feature names from preprocessor: {e}") from e
