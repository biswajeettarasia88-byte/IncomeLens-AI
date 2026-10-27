"""Integration and contract tests for the IncomeLens-AI FastAPI backend."""

import sys
from pathlib import Path
import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from incomelens.api import app

client = TestClient(app)


def test_health_endpoint():
    """Verify health endpoint returns 200 and loads verified model artifact."""
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True
    assert data["model_name"] == "RandomForestClassifier"
    assert data["model_sha256"] == "f164a61183466b25186377e8946f4cb74b2df924f69e50224f5d8de72858a285"
    assert data["datasets_available"]["test"] is True


def test_model_metadata_endpoint():
    """Verify model metadata returns hyperparameters, pipeline stages, and input schema."""
    response = client.get("/api/model/metadata")
    assert response.status_code == 200
    data = response.json()
    assert "stages" in data
    assert "parameters" in data
    assert data["parameters"]["n_estimators"] == 150
    assert data["parameters"]["random_state"] == 42
    assert len(data["expected_features"]) == 14
    assert data["transformed_features_count"] == 105


def test_predict_endpoint_valid():
    """Verify single prediction with local SHAP feature attributions."""
    payload = {
        "age": 45,
        "workclass": "Private",
        "fnlwgt": 180000.0,
        "education": "Bachelors",
        "education_num": 13,
        "marital_status": "Married-civ-spouse",
        "occupation": "Exec-managerial",
        "relationship": "Husband",
        "race": "White",
        "sex": "Male",
        "capital_gain": 0.0,
        "capital_loss": 0.0,
        "hours_per_week": 50,
        "native_country": "United-States",
        "explain": True,
    }
    response = client.post("/api/predict", json=payload)
    assert response.status_code == 200
    res = response.json()
    assert res["status"] == "success"
    assert res["prediction"] in ["<=50K", ">50K"]
    assert 0.0 <= res["high_income_probability"] <= 1.0
    assert 0.0 <= res["low_income_probability"] <= 1.0
    assert abs((res["high_income_probability"] + res["low_income_probability"]) - 1.0) < 1e-4
    assert res["confidence_level"] in ["Low", "Moderate", "High"]
    assert "explanation" in res
    assert "top_contributions" in res["explanation"]
    assert len(res["explanation"]["top_contributions"]) >= 5


def test_predict_endpoint_invalid_missing_fields():
    """Verify validation error when required survey fields are omitted."""
    payload = {
        "age": 30,
        "workclass": "Private",
    }
    response = client.post("/api/predict", json=payload)
    assert response.status_code == 422


def test_metrics_test_endpoint():
    """Verify test metrics endpoint matches verified held-out evaluation values."""
    response = client.get("/api/metrics/test")
    assert response.status_code == 200
    data = response.json()
    assert pytest.approx(data["key_findings"]["accuracy"], rel=1e-3) == 0.8683
    assert pytest.approx(data["key_findings"]["balanced_accuracy"], rel=1e-3) == 0.7809
    assert pytest.approx(data["key_findings"]["roc_auc"], rel=1e-3) == 0.9181
    assert data["confusion_matrix"]["tn"] == 7080
    assert data["confusion_matrix"]["fp"] == 417
    assert data["confusion_matrix"]["fn"] == 870
    assert data["confusion_matrix"]["tp"] == 1404


def test_metrics_validation_endpoint():
    """Verify validation metrics endpoint returns model comparison data."""
    response = client.get("/api/metrics/validation")
    assert response.status_code == 200
    data = response.json()
    assert "comparison" in data
    assert len(data["comparison"]) >= 2
    assert data["split"] == "grouped_validation"


def test_feature_importance_endpoint():
    """Verify feature importance endpoint returns top MDI features sorted descending."""
    response = client.get("/api/feature-importance?top_n=20")
    assert response.status_code == 200
    data = response.json()
    assert data["total_features"] == 105
    assert data["top_n"] == 20
    features = data["features"]
    assert len(features) == 20
    # Verify sorted descending
    for i in range(len(features) - 1):
        assert features[i]["importance"] >= features[i + 1]["importance"]


def test_explainability_summary_endpoint():
    """Verify SHAP explainability summary endpoint returns top drivers and metadata."""
    response = client.get("/api/explainability/summary")
    assert response.status_code == 200
    data = response.json()
    assert data["method"] == "shap.TreeExplainer"
    assert "top_drivers" in data
    assert len(data["top_drivers"]) >= 5
    assert data["target_class"] == ">50K"


def test_fairness_endpoint():
    """Verify fairness endpoint returns demographic records and 80% rule compliance."""
    response = client.get("/api/fairness")
    assert response.status_code == 200
    data = response.json()
    assert "records" in data
    assert len(data["records"]) == 13
    assert data["summary_findings"]["gender_adverse_impact"]["eeoc_80_rule_violated"] is True
    assert data["summary_findings"]["gender_adverse_impact"]["disparate_impact_ratio"] < 0.80


def test_dataset_metadata_endpoint():
    """Verify dataset metadata matches verified record counts and schema."""
    response = client.get("/api/dataset/metadata")
    assert response.status_code == 200
    data = response.json()
    assert data["splits"]["train"]["rows"] == 29301
    assert data["splits"]["validation"]["rows"] == 9770
    assert data["splits"]["test"]["rows"] == 9771
    assert data["total_rows"] == 48842
    assert data["data_quality_audit"]["exact_duplicate_rows"] == 52


def test_experiments_endpoint():
    """Verify experiments endpoint lists runs and production model metadata."""
    response = client.get("/api/experiments")
    assert response.status_code == 200
    data = response.json()
    assert data["total_runs"] >= 1
    assert data["production_model"]["sha256"] == "f164a61183466b25186377e8946f4cb74b2df924f69e50224f5d8de72858a285"


def test_figures_static_mount():
    """Verify static figures are accessible via API mount."""
    resp1 = client.get("/api/figures/feature_importance.png")
    assert resp1.status_code == 200
    assert resp1.headers["content-type"] in ["image/png", "application/octet-stream"]

    resp2 = client.get("/api/figures/shap_summary.png")
    assert resp2.status_code == 200
    assert resp2.headers["content-type"] in ["image/png", "application/octet-stream"]
