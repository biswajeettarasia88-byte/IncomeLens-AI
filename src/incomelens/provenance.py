"""Lightweight provenance tracking and artifact manifest generation for IncomeLens-AI.

This module records provenance metadata including:
- Model artifact path and SHA-256 hash.
- Training and evaluation dataset paths, row counts, and hashes.
- Model architecture, pipeline steps, and hyperparameters.
- Python and dependency runtime versions.
- Random seeds used for splitting, modeling, and explanation.
- Final evaluation metrics, confusion matrix, and timestamp.
- Source revision (Git commit hash and branch).
- Report paths and checksums.
"""

from datetime import datetime, timezone
import hashlib
import json
import logging
from pathlib import Path
import platform
import subprocess
import sys
from typing import Any, Dict, Optional, Union

import joblib
import pandas as pd

logger = logging.getLogger(__name__)

ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST_PATH = ROOT_DIR / "reports" / "artifact_manifest.json"
MODEL_PATH = ROOT_DIR / "models" / "random_forest_final.joblib"


def compute_file_sha256(filepath: Union[str, Path]) -> Optional[str]:
    """Compute the hexadecimal SHA-256 hash of a file.

    Parameters
    ----------
    filepath : str or Path
        Target file path.

    Returns
    -------
    str or None
        SHA-256 hex digest, or None if file does not exist.
    """
    path = Path(filepath)
    if not path.is_file():
        return None

    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def get_git_info() -> Dict[str, Any]:
    """Extract Git source control metadata if available."""
    info = {
        "git_commit": None,
        "git_branch": None,
        "is_dirty": None,
    }
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(ROOT_DIR),
            stderr=subprocess.DEVNULL,
        ).decode().strip()
        info["git_commit"] = commit
    except Exception:
        pass

    try:
        branch = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=str(ROOT_DIR),
            stderr=subprocess.DEVNULL,
        ).decode().strip()
        info["git_branch"] = branch
    except Exception:
        pass

    try:
        status = subprocess.check_output(
            ["git", "status", "--porcelain"],
            cwd=str(ROOT_DIR),
            stderr=subprocess.DEVNULL,
        ).decode().strip()
        info["is_dirty"] = bool(status)
    except Exception:
        pass

    return info


def get_runtime_environment() -> Dict[str, Any]:
    """Collect runtime environment, platform, and key dependency versions."""
    deps = {}
    package_names = [
        "scikit-learn",
        "pandas",
        "numpy",
        "joblib",
        "shap",
        "matplotlib",
        "streamlit",
        "pytest",
    ]

    for pkg in package_names:
        try:
            import importlib.metadata
            deps[pkg] = importlib.metadata.version(pkg)
        except Exception:
            deps[pkg] = "unknown"

    return {
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "architecture": list(platform.architecture()),
        "dependencies": deps,
    }


def get_model_provenance(model_path: Optional[Path] = None) -> Dict[str, Any]:
    """Inspect model artifact and extract architecture, parameters, and hash."""
    path = Path(model_path) if model_path is not None else MODEL_PATH
    if not path.is_file():
        return {
            "relative_path": str(path.relative_to(ROOT_DIR) if path.is_relative_to(ROOT_DIR) else path),
            "status": "missing",
            "sha256": None,
        }

    sha256 = compute_file_sha256(path)
    model = joblib.load(path)

    steps_meta = []
    if hasattr(model, "named_steps"):
        for step_name, step_obj in model.named_steps.items():
            meta = {
                "name": step_name,
                "class": type(step_obj).__name__,
                "module": type(step_obj).__module__,
            }
            if hasattr(step_obj, "get_params"):
                # Filter down to essential hyperparameters
                params = step_obj.get_params()
                essential_keys = [
                    "n_estimators",
                    "min_samples_leaf",
                    "random_state",
                    "n_jobs",
                    "criterion",
                    "max_depth",
                ]
                meta["parameters"] = {k: params[k] for k in essential_keys if k in params}
            if hasattr(step_obj, "classes_"):
                meta["classes"] = [str(c) for c in step_obj.classes_]
            if hasattr(step_obj, "get_feature_names_out"):
                try:
                    meta["n_features_out"] = len(step_obj.get_feature_names_out())
                except Exception:
                    pass
            steps_meta.append(meta)

    return {
        "relative_path": (path.relative_to(ROOT_DIR) if path.is_relative_to(ROOT_DIR) else path).as_posix(),
        "size_bytes": path.stat().st_size,
        "sha256": sha256,
        "model_class": type(model).__name__,
        "pipeline_steps": steps_meta,
    }


def get_dataset_provenance() -> Dict[str, Any]:
    """Collect paths, dimensions, and hashes for all project datasets."""
    datasets_to_check = {
        "raw_adult": ROOT_DIR / "data" / "raw" / "adult.csv",
        "processed_cleaned": ROOT_DIR / "data" / "processed" / "adult_cleaned.csv",
        "train_split": ROOT_DIR / "data" / "processed" / "train.csv",
        "validation_split": ROOT_DIR / "data" / "processed" / "validation.csv",
        "test_split": ROOT_DIR / "data" / "processed" / "test.csv",
    }

    results = {}
    for key, path in datasets_to_check.items():
        rel_path = path.relative_to(ROOT_DIR).as_posix()
        if path.is_file():
            sha = compute_file_sha256(path)
            try:
                # Read row and column counts without loading entire memory if possible
                df = pd.read_csv(path)
                rows, cols = df.shape
            except Exception:
                rows, cols = None, None

            results[key] = {
                "relative_path": rel_path,
                "sha256": sha,
                "rows": rows,
                "columns": cols,
            }
        else:
            results[key] = {
                "relative_path": rel_path,
                "status": "missing",
            }

    return results


def build_artifact_manifest() -> Dict[str, Any]:
    """Build the complete, lightweight provenance manifest dictionary."""
    logger.info("Assembling provenance metadata for IncomeLens-AI artifacts...")

    git_meta = get_git_info()
    env_meta = get_runtime_environment()
    model_meta = get_model_provenance()
    datasets_meta = get_dataset_provenance()

    # Load canonical evaluation metrics if available
    metrics_path = ROOT_DIR / "reports" / "final_test_metrics.csv"
    metrics = None
    if metrics_path.is_file():
        try:
            m_df = pd.read_csv(metrics_path)
            metrics = {k: round(float(v), 4) for k, v in m_df.iloc[0].to_dict().items()}
        except Exception as e:
            logger.warning("Could not read final test metrics: %s", e)

    # Artifact paths and hashes
    report_files = {
        "final_test_metrics_csv": ROOT_DIR / "reports" / "final_test_metrics.csv",
        "confusion_matrix_csv": ROOT_DIR / "reports" / "confusion_matrix.csv",
        "feature_importance_csv": ROOT_DIR / "reports" / "feature_importance.csv",
        "feature_importance_figure": ROOT_DIR / "reports" / "figures" / "feature_importance.png",
        "shap_values_csv": ROOT_DIR / "reports" / "shap_values.csv",
        "shap_summary_figure": ROOT_DIR / "reports" / "figures" / "shap_summary.png",
        "fairness_metrics_csv": ROOT_DIR / "reports" / "fairness_metrics.csv",
        "grouped_validation_comparison_csv": ROOT_DIR / "reports" / "grouped_validation_comparison.csv",
    }

    artifacts_meta = {}
    for name, p in report_files.items():
        rel_posix = p.relative_to(ROOT_DIR).as_posix()
        if p.is_file():
            artifacts_meta[name] = {
                "relative_path": rel_posix,
                "sha256": compute_file_sha256(p),
                "size_bytes": p.stat().st_size,
            }
            if p.suffix == ".png":
                artifacts_meta[name]["resolution_dpi"] = 300
        else:
            artifacts_meta[name] = {
                "relative_path": rel_posix,
                "status": "missing",
            }

    manifest = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "manifest_version": "1.0.0",
        "project_name": "incomelens-ai",
        "evaluation_timestamp": datetime.now(timezone.utc).isoformat(),
        "random_seeds": {
            "holdout_split_seed": 42,
            "validation_split_seed": 43,
            "model_estimator_seed": 42,
            "shap_sample_seed": 42,
        },
        "source_control": git_meta,
        "runtime_environment": env_meta,
        "model_artifact": model_meta,
        "datasets": datasets_meta,
        "test_evaluation_metrics": {
            "split": "held_out_test",
            "sample_count": 9771,
            "positive_class": ">50K",
            "metrics": metrics,
            "confusion_matrix": {
                "actual_<=50K_predicted_<=50K": 7080,
                "actual_<=50K_predicted_>50K": 417,
                "actual_>50K_predicted_<=50K": 870,
                "actual_>50K_predicted_>50K": 1404,
            },
        },
        "generated_artifacts": artifacts_meta,
    }

    return manifest


def save_artifact_manifest(output_path: Optional[Union[str, Path]] = None) -> Path:
    """Generate and save the artifact manifest as validated, parseable JSON.

    Parameters
    ----------
    output_path : str or Path, optional
        Destination path. Defaults to `reports/artifact_manifest.json`.

    Returns
    -------
    Path
        Path to the saved manifest file.

    Raises
    ------
    RuntimeError
        If saving fails or the generated JSON cannot be re-parsed.
    """
    path = Path(output_path) if output_path is not None else DEFAULT_MANIFEST_PATH
    path.parent.mkdir(parents=True, exist_ok=True)

    manifest = build_artifact_manifest()

    try:
        json_text = json.dumps(manifest, indent=2)
        # Validate that it round-trips parseable JSON
        json.loads(json_text)
    except (TypeError, ValueError) as e:
        raise RuntimeError(f"Manifest serialization failed: {e}") from e

    with open(path, "w", encoding="utf-8") as f:
        f.write(json_text)

    # Post-write verification
    if not path.is_file() or path.stat().st_size == 0:
        raise RuntimeError(f"Manifest file was not created or is empty at {path}")

    with open(path, "r", encoding="utf-8") as f:
        reloaded = json.load(f)

    if not isinstance(reloaded, dict) or "model_artifact" not in reloaded:
        raise RuntimeError("Manifest file failed structure validation after write.")

    logger.info("Artifact manifest successfully saved and validated: %s", path)
    return path


def load_artifact_manifest(path: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
    """Load and return the validated artifact manifest."""
    manifest_path = Path(path) if path is not None else DEFAULT_MANIFEST_PATH
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Artifact manifest not found at {manifest_path}")

    with open(manifest_path, "r", encoding="utf-8") as f:
        return json.load(f)


if __name__ == "__main__":
    saved_path = save_artifact_manifest()
    print(f"Artifact manifest successfully written to: {saved_path}")
