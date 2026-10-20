
from pathlib import Path

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "random_forest_final.joblib"
TRAIN_PATH = ROOT / "data" / "processed" / "train.csv"


def test_model_file_exists():
    assert MODEL_PATH.exists(), f"Model not found: {MODEL_PATH}"


def test_model_loads():
    model = joblib.load(MODEL_PATH)
    assert model is not None


def test_model_predictions_are_valid():
    model = joblib.load(MODEL_PATH)
    df = pd.read_csv(TRAIN_PATH).drop(columns=["income"], errors="ignore")

    sample = df.head(10)
    predictions = model.predict(sample)

    assert len(predictions) == len(sample)
    assert pd.Series(predictions).notna().all()

    if hasattr(model, "classes_"):
        assert set(predictions).issubset(set(model.classes_))
