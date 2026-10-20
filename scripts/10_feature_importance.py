
from pathlib import Path

import joblib
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / "data" / "processed"
MODEL_PATH = ROOT / "models" / "random_forest_final.joblib"
REPORT_DIR = ROOT / "reports"
FIGURE_DIR = REPORT_DIR / "figures"

FIGURE_DIR.mkdir(parents=True, exist_ok=True)

# Load the final trained pipeline.
model = joblib.load(MODEL_PATH)

# Use development data, not the test set.
train = pd.read_csv(DATA_DIR / "train.csv")
validation = pd.read_csv(DATA_DIR / "validation.csv")
data = pd.concat([train, validation], ignore_index=True)

import sys
sys.path.insert(0, str(ROOT / "src"))

from incomelens.preprocessing import split_features_target

X, y = split_features_target(data)

# Apply the preprocessing already fitted by the saved pipeline.
preprocessor = model.named_steps["preprocessor"]
classifier = model.named_steps["classifier"]

X_transformed = preprocessor.transform(X)

feature_names = preprocessor.get_feature_names_out()

importance = pd.DataFrame({
    "feature": feature_names,
    "importance": classifier.feature_importances_,
}).sort_values("importance", ascending=False)

importance.to_csv(
    REPORT_DIR / "feature_importance.csv",
    index=False,
)

top = importance.head(20).sort_values("importance")

plt.figure(figsize=(10, 8))
plt.barh(top["feature"], top["importance"])
plt.xlabel("Random Forest feature importance")
plt.title("IncomeLens: Top 20 Model Features")
plt.tight_layout()

plt.savefig(
    FIGURE_DIR / "feature_importance.png",
    dpi=200,
    bbox_inches="tight",
)
plt.close()

print("\nTop 20 features:")
print(importance.head(20).to_string(index=False))
print("\nSaved reports/feature_importance.csv")
print("Saved reports/figures/feature_importance.png")
