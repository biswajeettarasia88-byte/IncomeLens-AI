"""Unit tests for provenance tracking and reports/artifact_manifest.json."""

import json
from pathlib import Path
import re
import sys
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

MANIFEST_PATH = ROOT / "reports" / "artifact_manifest.json"

from incomelens.provenance import (
    build_artifact_manifest,
    compute_file_sha256,
    get_dataset_provenance,
    get_git_info,
    get_model_provenance,
    get_runtime_environment,
    load_artifact_manifest,
    save_artifact_manifest,
)


def test_artifact_manifest_file_exists():
    """Verify that reports/artifact_manifest.json exists and is non-empty."""
    assert MANIFEST_PATH.is_file(), f"Artifact manifest not found at {MANIFEST_PATH}"
    assert MANIFEST_PATH.stat().st_size > 0, "Artifact manifest file is empty"


def test_artifact_manifest_is_parseable_json():
    """Verify that reports/artifact_manifest.json is valid, parseable JSON."""
    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert isinstance(data, dict)
    assert data.get("manifest_version") == "1.0.0"
    assert data.get("project_name") == "incomelens-ai"


def test_artifact_manifest_required_sections():
    """Verify all required top-level provenance schema sections are present."""
    manifest = load_artifact_manifest(MANIFEST_PATH)
    required_keys = [
        "$schema",
        "manifest_version",
        "project_name",
        "evaluation_timestamp",
        "random_seeds",
        "source_control",
        "runtime_environment",
        "model_artifact",
        "datasets",
        "test_evaluation_metrics",
        "generated_artifacts",
    ]
    for key in required_keys:
        assert key in manifest, f"Missing required section in manifest: {key}"


def test_artifact_manifest_model_details():
    """Verify model artifact metadata, hashes, and parameter recording."""
    manifest = load_artifact_manifest(MANIFEST_PATH)
    model_meta = manifest["model_artifact"]

    assert model_meta["relative_path"] == "models/random_forest_final.joblib"
    assert model_meta["model_class"] == "Pipeline"
    assert isinstance(model_meta["sha256"], str)
    assert len(model_meta["sha256"]) == 64
    assert model_meta["size_bytes"] > 10_000_000

    # Verify pipeline steps
    steps = model_meta.get("pipeline_steps", [])
    step_names = [s["name"] for s in steps]
    assert "preprocessor" in step_names
    assert "classifier" in step_names

    clf_step = next(s for s in steps if s["name"] == "classifier")
    params = clf_step.get("parameters", {})
    assert params.get("n_estimators") == 150
    assert params.get("min_samples_leaf") == 2
    assert params.get("random_state") == 42


def test_artifact_manifest_dataset_provenance():
    """Verify dataset splits, row counts, and cryptographic checksums."""
    manifest = load_artifact_manifest(MANIFEST_PATH)
    datasets = manifest["datasets"]

    expected_datasets = [
        "raw_adult",
        "processed_cleaned",
        "train_split",
        "validation_split",
        "test_split",
    ]
    for ds in expected_datasets:
        assert ds in datasets, f"Missing dataset in manifest: {ds}"
        meta = datasets[ds]
        assert "sha256" in meta and len(meta["sha256"]) == 64
        assert "rows" in meta and meta["rows"] > 0
        assert meta["columns"] == 15

    assert datasets["train_split"]["rows"] == 29301
    assert datasets["validation_split"]["rows"] == 9770
    assert datasets["test_split"]["rows"] == 9771
    assert datasets["raw_adult"]["rows"] == 48842


def test_artifact_manifest_random_seeds():
    """Verify documented random seeds for reproducibility."""
    manifest = load_artifact_manifest(MANIFEST_PATH)
    seeds = manifest["random_seeds"]

    assert seeds.get("holdout_split_seed") == 42
    assert seeds.get("validation_split_seed") == 43
    assert seeds.get("model_estimator_seed") == 42
    assert seeds.get("shap_sample_seed") == 42


def test_artifact_manifest_metrics_and_report_paths():
    """Verify canonical metrics and report artifact paths are recorded."""
    manifest = load_artifact_manifest(MANIFEST_PATH)
    metrics_meta = manifest["test_evaluation_metrics"]
    metrics = metrics_meta["metrics"]

    assert metrics_meta["split"] == "held_out_test"
    assert metrics_meta["sample_count"] == 9771
    assert metrics_meta["positive_class"] == ">50K"
    assert metrics["accuracy"] == pytest.approx(0.8683, abs=1e-3)
    assert metrics["precision_high_income"] == pytest.approx(0.7710, abs=1e-3)
    assert metrics["recall_high_income"] == pytest.approx(0.6174, abs=1e-3)
    assert metrics["roc_auc"] == pytest.approx(0.9181, abs=1e-3)

    artifacts = manifest["generated_artifacts"]
    expected_reports = [
        "final_test_metrics_csv",
        "confusion_matrix_csv",
        "feature_importance_csv",
        "feature_importance_figure",
        "shap_values_csv",
        "shap_summary_figure",
        "fairness_metrics_csv",
    ]
    for rep in expected_reports:
        assert rep in artifacts, f"Missing report in manifest: {rep}"
        assert "sha256" in artifacts[rep]
        assert len(artifacts[rep]["sha256"]) == 64


def test_artifact_manifest_no_sensitive_raw_records():
    """Verify that artifact_manifest.json contains zero raw personal record dumps or PII."""
    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        content = f.read()

    # Verify no raw individual row entries
    assert "fnlwgt:" not in content
    assert "capital-gain:" not in content
    # Ensure no passwords, api keys, or secrets tokens
    assert "password" not in content.lower()
    assert "secret" not in content.lower()
    assert "apikey" not in content.lower()


def test_compute_file_sha256_missing_file():
    """Verify compute_file_sha256 returns None for non-existent files."""
    assert compute_file_sha256("non_existent_file_path_12345.xyz") is None


def test_save_artifact_manifest_custom_path(tmp_path):
    """Verify saving manifest to custom destination without mutating production files."""
    custom_path = tmp_path / "custom_manifest.json"
    saved = save_artifact_manifest(custom_path)

    assert saved == custom_path
    assert saved.is_file()
    assert saved.stat().st_size > 0

    with open(saved, "r", encoding="utf-8") as f:
        reloaded = json.load(f)
    assert reloaded["project_name"] == "incomelens-ai"
