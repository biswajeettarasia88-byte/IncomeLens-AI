
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


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


def build_preprocessor():
    """Create a preprocessing pipeline for mixed-type census data."""

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
                    sparse_output=True,
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


def split_features_target(df: pd.DataFrame):
    """Separate predictors from the binary income target."""

    df = df.copy()

    # Prevent the target from entering the model inputs
    y = df[TARGET_COLUMN].str.strip()
    y = y.map({"<=50K": 0, ">50K": 1})

    if y.isna().any():
        raise ValueError("Unexpected or missing income labels.")

    X = df.drop(columns=[TARGET_COLUMN])

    # Retain only the explicitly defined predictors
    expected = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    X = X[expected]

    return X, y.astype(int)
