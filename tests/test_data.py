
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
TRAIN_PATH = ROOT / "data" / "processed" / "train.csv"


@pytest.fixture(scope="module")
def train_data():
    assert TRAIN_PATH.exists(), f"Training data not found: {TRAIN_PATH}"
    return pd.read_csv(TRAIN_PATH)


def test_training_data_exists():
    assert TRAIN_PATH.exists()


def test_training_data_loads(train_data):
    assert not train_data.empty
    assert len(train_data.columns) > 1


def test_training_data_has_no_duplicate_columns(train_data):
    assert train_data.columns.is_unique


def test_income_target_is_present(train_data):
    assert "income" in train_data.columns
    assert train_data["income"].notna().all()
