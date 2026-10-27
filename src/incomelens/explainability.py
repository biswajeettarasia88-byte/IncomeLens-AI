from pathlib import Path
from typing import Any, List, Optional, Tuple, Union

import joblib
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from scipy import sparse
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = ROOT / "models" / "random_forest_final.joblib"
DATA_PATH = ROOT / "train.csv"
REPORT_DIR = ROOT / "reports" / "figures"

REPORT_DIR.mkdir(parents=True, exist_ok=True)


def load_training_data():
    """Find the training CSV using the project's existing file layout."""
    candidates = [
        ROOT / "train.csv",
        ROOT / "data" / "train.csv",
        ROOT / "data" / "processed" / "train.csv",
    ]

    # Also check the project directories for an existing train.csv.
    candidates.extend([
        ROOT / "data" / "splits" / "train.csv",
        ROOT / "data" / "processed" / "splits" / "train.csv",
    ])

    for path in candidates:
        if path.is_file():
            print(f"Loading training data: {path}")
            return pd.read_csv(path)

    raise FileNotFoundError(
        "Could not find train.csv. Checked:\n"
        + "\n".join(str(p) for p in candidates)
        + "\nCheck OUTPUT_DIR in scripts/07_make_grouped_split.py."
    )


def to_dense_numeric(X):
    """Convert transformed features into a numeric matrix."""
    if sparse.issparse(X):
        X = X.toarray()

    X = np.asarray(X, dtype=np.float64)

    if X.ndim != 2:
        raise ValueError(f"Expected 2D feature matrix; got {X.shape}")

    if not np.isfinite(X).all():
        raise ValueError(
            "Transformed data contains NaN or infinity. "
            "Check the preprocessing pipeline."
        )

    return X


def extract_estimator_and_preprocessor(model: Any) -> Tuple[Optional[Any], Any]:
    """Extract fitted preprocessor and tree estimator from a Pipeline or estimator."""
    if isinstance(model, Pipeline):
        if len(model.steps) < 2:
            raise ValueError(
                "Pipeline must contain preprocessing and a final estimator."
            )
        preprocessor = model[:-1]
        estimator = model.steps[-1][1]
    else:
        preprocessor = None
        estimator = model

    return preprocessor, estimator


def extract_positive_class_shap_values(shap_raw: Any, positive_class_idx: int = 1) -> np.ndarray:
    """Extract 2D array of SHAP values for the specified positive class.

    Handles list of arrays (older SHAP API) and 3D ndarray (newer SHAP API).
    """
    if isinstance(shap_raw, list):
        if len(shap_raw) > positive_class_idx:
            values = shap_raw[positive_class_idx]
        else:
            values = shap_raw[0]
    else:
        values = np.asarray(shap_raw)
        if values.ndim == 3:
            if values.shape[2] > positive_class_idx:
                values = values[:, :, positive_class_idx]
            else:
                values = values[:, :, 0]

    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError(f"Unexpected SHAP output shape: {values.shape}, expected 2D matrix.")

    return values


def generate_shap_table(values: np.ndarray, feature_names: List[str]) -> pd.DataFrame:
    """Construct a validated DataFrame of SHAP values aligned with feature names."""
    if values.ndim != 2:
        raise ValueError(f"Expected 2D SHAP array, got {values.shape}")

    if values.shape[1] != len(feature_names):
        raise ValueError(
            f"Feature count mismatch: SHAP values have {values.shape[1]} columns, "
            f"but {len(feature_names)} feature names were provided."
        )

    return pd.DataFrame(values, columns=feature_names)


def save_shap_summary_plot(
    values: np.ndarray,
    X_sample: np.ndarray,
    feature_names: List[str],
    plot_path: Union[str, Path],
    max_display: int = 20,
) -> Path:
    """Generate and save a SHAP summary plot, verifying the output file is non-empty."""
    plot_path = Path(plot_path)
    plot_path.parent.mkdir(parents=True, exist_ok=True)

    shap.summary_plot(
        values,
        X_sample,
        feature_names=feature_names,
        max_display=max_display,
        show=False,
    )
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close("all")

    if not plot_path.is_file() or plot_path.stat().st_size == 0:
        raise RuntimeError(f"Failed to generate non-empty plot at {plot_path}")

    return plot_path


def main():
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    model = joblib.load(MODEL_PATH)
    train_df = load_training_data()

    # Remove target and prevent accidental target leakage.
    X_train = train_df.drop(columns=["income"], errors="ignore")

    # Apply the fitted preprocessing pipeline, if present.
    preprocessor, estimator = extract_estimator_and_preprocessor(model)
    if preprocessor is not None:
        X_processed = preprocessor.transform(X_train)
        try:
            feature_names = preprocessor.get_feature_names_out()
            feature_names = [str(name) for name in feature_names]
        except (AttributeError, ValueError):
            feature_names = [
                f"feature_{i}" for i in range(X_processed.shape[1])
            ]
    else:
        # This branch assumes the saved model accepts numeric features.
        X_processed = X_train
        feature_names = [str(c) for c in X_train.columns]

    X_processed = to_dense_numeric(X_processed)

    if X_processed.shape[1] != len(feature_names):
        feature_names = [
            f"feature_{i}" for i in range(X_processed.shape[1])
        ]

    # Use a manageable sample to reduce memory and execution time.
    sample_size = min(1000, len(X_processed))
    if sample_size == 0:
        raise ValueError("Training data contains no rows.")

    rng = np.random.default_rng(42)
    indices = rng.choice(
        len(X_processed),
        size=sample_size,
        replace=False,
    )
    X_sample = X_processed[indices]

    print(f"Explaining {sample_size} rows...")
    print(f"Estimator: {type(estimator).__name__}")
    print(f"Features: {X_sample.shape[1]}")

    # Explain the fitted tree estimator, not the whole Pipeline.
    explainer = shap.TreeExplainer(estimator)
    shap_raw = explainer.shap_values(X_sample)

    values = extract_positive_class_shap_values(shap_raw, positive_class_idx=1)

    # Save SHAP values as a table for reproducibility.
    shap_df = generate_shap_table(values, feature_names)
    shap_csv = REPORT_DIR.parent / "shap_values.csv"
    shap_csv.parent.mkdir(parents=True, exist_ok=True)
    shap_df.to_csv(shap_csv, index=False)

    # Global feature importance plot (publication-quality 300 DPI).
    plot_path = REPORT_DIR / "shap_summary.png"
    save_shap_summary_plot(values, X_sample, feature_names, plot_path, max_display=20)

    # Post-generation verification
    if not shap_csv.is_file() or shap_csv.stat().st_size == 0:
        raise RuntimeError(f"Generated SHAP CSV artifact is missing or empty: {shap_csv}")
    if not plot_path.is_file() or plot_path.stat().st_size == 0:
        raise RuntimeError(f"Generated SHAP plot artifact is missing or empty: {plot_path}")

    print(f"Saved SHAP summary (300 DPI): {plot_path}")
    print(f"Saved SHAP values: {shap_csv}")
    print("Verified: Generated SHAP explainability artifacts are valid and non-empty.")


if __name__ == "__main__":
    main()
