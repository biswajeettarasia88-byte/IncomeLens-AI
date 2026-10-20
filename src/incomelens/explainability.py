
from pathlib import Path

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


def main():
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    model = joblib.load(MODEL_PATH)
    train_df = load_training_data()

    # Remove target and prevent accidental target leakage.
    X_train = train_df.drop(columns=["income"], errors="ignore")

    # Apply the fitted preprocessing pipeline, if present.
    if isinstance(model, Pipeline):
        if len(model.steps) < 2:
            raise ValueError(
                "Pipeline must contain preprocessing and a final estimator."
            )

        preprocessor = model[:-1]
        estimator = model.steps[-1][1]

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
        estimator = model
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
    shap_values = explainer.shap_values(X_sample)

    # Support older and newer SHAP output formats.
    if isinstance(shap_values, list):
        values = (
            shap_values[1]
            if len(shap_values) > 1
            else shap_values[0]
        )
    else:
        values = np.asarray(shap_values)

        if values.ndim == 3:
            # Newer SHAP versions: samples x features x classes.
            values = values[:, :, 1] if values.shape[2] > 1 else values[:, :, 0]

    if values.ndim != 2:
        raise ValueError(
            f"Unexpected SHAP output shape: {values.shape}"
        )

    # Save SHAP values as a table for reproducibility.
    shap_df = pd.DataFrame(values, columns=feature_names)
    shap_csv = REPORT_DIR.parent / "shap_values.csv"
    shap_df.to_csv(shap_csv, index=False)

    # Global feature importance plot.
    shap.summary_plot(
        values,
        X_sample,
        feature_names=feature_names,
        max_display=20,
        show=False,
    )

    plt.tight_layout()
    plot_path = REPORT_DIR / "shap_summary.png"
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close("all")

    print(f"Saved SHAP summary: {plot_path}")
    print(f"Saved SHAP values: {shap_csv}")


if __name__ == "__main__":
    main()
