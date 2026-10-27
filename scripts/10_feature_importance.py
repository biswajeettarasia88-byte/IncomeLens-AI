"""Feature importance analysis for IncomeLens-AI production model.

This script extracts and visualizes the Mean Decrease in Impurity (MDI / Gini Importance)
from the fitted Random Forest estimator.

IMPORTANT METHODOLOGICAL NOTE:
- Feature importance extracted via `classifier.feature_importances_` is impurity-based (MDI).
- It measures the total reduction of Gini impurity brought by that feature across all splits
  in the training forest.
- It is NOT permutation-based, NOT SHAP-based, and MUST NOT be interpreted as causal evidence.
- MDI is known to exhibit selection bias towards continuous and high-cardinality features
  (e.g., fnlwgt), and collinear features (e.g., marital-status vs. relationship) dilute each other.
"""

from pathlib import Path
import sys

import joblib
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

MODEL_PATH = ROOT / "models" / "random_forest_final.joblib"
REPORT_DIR = ROOT / "reports"
FIGURE_DIR = REPORT_DIR / "figures"

FIGURE_DIR.mkdir(parents=True, exist_ok=True)


def main():
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

    # Load the final trained pipeline.
    model = joblib.load(MODEL_PATH)

    # Extract pipeline components
    if hasattr(model, "named_steps"):
        preprocessor = model.named_steps["preprocessor"]
        classifier = model.named_steps["classifier"]
    else:
        raise ValueError("Expected a scikit-learn Pipeline with 'preprocessor' and 'classifier'.")

    # Extract transformed feature names matching the 105 one-hot encoded inputs
    feature_names = preprocessor.get_feature_names_out()

    # Mean Decrease in Impurity (Gini Importance) directly from the fitted estimator
    importances = classifier.feature_importances_

    importance = pd.DataFrame({
        "feature": [str(f) for f in feature_names],
        "importance": importances,
    }).sort_values("importance", ascending=False)

    # Save complete feature importance table
    csv_path = REPORT_DIR / "feature_importance.csv"
    importance.to_csv(csv_path, index=False)

    # Plot top 20 features with accurate methodological labeling
    top = importance.head(20).sort_values("importance")

    plt.figure(figsize=(10, 8))
    plt.barh(top["feature"], top["importance"], color="#2b5c8f")
    plt.xlabel("Mean Decrease in Impurity (Gini Importance)")
    plt.title(
        "IncomeLens: Top 20 Model Features (Mean Decrease in Impurity)\n"
        "[Training Split Metric - Not Causal Evidence]",
        fontsize=12,
        pad=12,
    )
    plt.tight_layout()

    plot_path = FIGURE_DIR / "feature_importance.png"
    plt.savefig(
        plot_path,
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    # Verification of generated artifacts
    if not csv_path.is_file() or csv_path.stat().st_size == 0:
        raise RuntimeError(f"Feature importance CSV missing or empty: {csv_path}")
    if not plot_path.is_file() or plot_path.stat().st_size == 0:
        raise RuntimeError(f"Feature importance figure missing or empty: {plot_path}")

    print("\nTop 20 features (Mean Decrease in Impurity):")
    print(importance.head(20).to_string(index=False))
    print(f"\nSaved {csv_path}")
    print(f"Saved {plot_path} (300 DPI publication quality)")
    print("Verified: Generated feature importance artifacts are valid and non-empty.")
    print("Note: Impurity-based importance reflects training splits and is non-causal.")


if __name__ == "__main__":
    main()
