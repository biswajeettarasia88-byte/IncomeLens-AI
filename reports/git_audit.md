# IncomeLens AI — Git & Repository Audit Report

**Audit Date:** October 20, 2026  
**Auditor:** Senior Machine Learning Engineer  
**Project Path:** `D:\incomelens-ai`  
**Report Destination:** `reports/git_audit.md`

---

## 1. Executive Summary

An audit of the Git repository configuration, branches, working tree status, tracked vs. untracked artifacts, and ignore rules was conducted.

### Core Findings:
1. **Repository Lifecycle State:**
   - Active Branch: `master`
   - Commit History: **0 commits** (`fatal: your current branch 'master' does not have any commits yet`).
   - The repository has been initialized (`git init`) and contains 41 dangling git object blobs in `.git/objects/` (~26 MB) from prior staging operations, but no tree has ever been committed.
2. **Untracked File Risk:**
   - Root `.gitignore` is **0 bytes (completely empty)**.
   - If `git add .` were executed today, Git would stage **~103 MB of binaries, large datasets, and compiled bytecode**, including:
     - `models/random_forest_final.joblib` (50.1 MB)
     - `models/random_forest_grouped.joblib` (39.0 MB)
     - `data/raw/adult.csv` (5.3 MB)
     - `data/processed/adult_cleaned.csv` (5.3 MB)
     - `data/processed/train.csv` (3.2 MB)
     - `data/processed/validation.csv` (1.1 MB)
     - `data/processed/test.csv` (1.1 MB)
     - `reports/shap_values.csv` (2.3 MB)
     - 5 compiled Python bytecode files (`*.cpython-314.pyc`) in `src/` and `tests/`.
3. **Virtual Environments Status:**
   - `.venv/` (392.8 MB) and `.venv-1/` (8.1 MB) are currently ignored by Git only because each contains an internal `.gitignore` containing `*`. However, they are not listed in the root `.gitignore`.
4. **Credential Audit:**
   - No project credentials, private API keys, `.env` files, or SSH keys exist in the repository.
   - The only `.pem` files found are public CA bundles inside `site-packages/certifi`.
5. **Preservation Guarantee:**
   - No destructive commands (`git reset`, `git clean`, `git checkout`, `rm`) were executed.
   - All user files, data artifacts, and model weights remain intact.

---

## 2. Git Status & Working Tree Inventory

### 2.1 Branch & Commit Status
- **Current Branch:** `master`
- **Upstream / Remotes:** None configured (`git remote` is empty).
- **Commit Count:** `0` (Uncommitted initial repository).

### 2.2 Comprehensive Untracked File Inventory (58 Items)
Grouped by category and artifact size:

#### A. Large Binary Models (`models/` — 85.0 MB total)
- `models/random_forest_final.joblib` (50,084,507 bytes / 50.1 MB)
- `models/random_forest_grouped.joblib` (39,018,267 bytes / 39.0 MB)
- `models/logistic_regression.joblib` (8,106 bytes / 8.1 KB)
- `models/logistic_regression_grouped.joblib` (8,106 bytes / 8.1 KB)

#### B. Tabular Datasets (`data/` — 15.2 MB total)
- `data/raw/adult.csv` (5,336,181 bytes / 5.3 MB)
- `data/processed/adult_cleaned.csv` (5,319,900 bytes / 5.3 MB)
- `data/processed/train.csv` (3,190,538 bytes / 3.2 MB)
- `data/processed/test.csv` (1,064,895 bytes / 1.1 MB)
- `data/processed/validation.csv` (1,064,781 bytes / 1.1 MB)

#### C. Large Generated Output Tables (`reports/` — 2.3 MB)
- `reports/shap_values.csv` (2,341,543 bytes / 2.3 MB)

#### D. Compiled Bytecode Caches (`__pycache__/`)
- `src/incomelens/__pycache__/__init__.cpython-314.pyc` (142 bytes)
- `src/incomelens/__pycache__/preprocessing.cpython-314.pyc` (2,494 bytes)
- `src/incomelens/__pycache__/explainability.cpython-314.pyc` (7,427 bytes)
- `tests/__pycache__/test_data.cpython-314-pytest-9.1.1.pyc` (6,688 bytes)
- `tests/__pycache__/test_predictions.cpython-314-pytest-9.1.1.pyc` (7,485 bytes)

#### E. Code, Scripts, Reports & Configuration Files
- Configuration: `.gitignore`, `requirements.txt`
- Documentation: `README.md`, `docs/dataset_card.md`, `docs/experiment_protocol.md`, `docs/literature_review.csv`, `docs/problem_statement.md`, `notebooks/01_initial_eda.ipynb`
- Source code: `src/incomelens/__init__.py`, `data.py`, `preprocessing.py`, `training.py`, `evaluation.py`, `fairness.py`, `explainability.py`
- Procedural scripts: `scripts/01_audit_data.py` through `scripts/10_feature_importance.py` (10 files)
- Tests: `tests/test_data.py`, `tests/test_predictions.py`
- Streamlit application: `app/Home.py`, `app/pages/1_Predict_Income.py`, `2_Model_Comparison.py`, `3_Explainability.py`, `4_Fairness_Audit.py`
- Generated metrics & figures:
  - `reports/validation_model_comparison.csv`
  - `reports/grouped_validation_comparison.csv`
  - `reports/final_test_metrics.csv`
  - `reports/confusion_matrix.csv`
  - `reports/feature_importance.csv`
  - `reports/figures/feature_importance.png`
  - `reports/figures/shap_summary.png`
  - `reports/project_audit.md`
  - `reports/environment_audit.md`

---

## 3. Ignored Files & Git Internals Audit

### 3.1 Currently Ignored Directories
`git status --ignored` identifies:
- `.venv/` (392.8 MB) — ignored via `.venv/.gitignore` (`*`)
- `.venv-1/` (8.1 MB) — ignored via `.venv-1/.gitignore` (`*`)
- `.pytest_cache/` — ignored via `.pytest_cache/.gitignore` (`*`)

*Note:* While these subdirectories are temporarily shielded by tool-generated internal `.gitignore` files, the root repository provides zero ignore protection.

### 3.2 Git Object Store Analysis (`.git/objects/`)
- Total size of `.git`: **25.98 MB**.
- `git fsck --lost-found` reveals **41 dangling blob objects** dating from earlier uncommitted staging actions.
  - Examples: `9e68bc` (12.8 MB), `0b2a43` (10.0 MB).
- These dangling blobs reflect previously staged versions of datasets and model artifacts. Because no commit was finalized, they remain unreferenced.

---

## 4. Proposed Minimal `.gitignore` Corrections

The current root `.gitignore` is **0 bytes**. To ensure datasets, virtual environments, credentials, large binaries, and cache files are never accidentally tracked, the following minimal `.gitignore` is proposed:

```gitignore
# Byte-compiled / optimized / DLL files
__pycache__/
*.py[cod]
*$py.class

# Caches and test runners
.pytest_cache/
.coverage
htmlcov/

# Virtual Environments
.venv/
.venv*/
venv/
env/

# Serialized Model Artifacts (Track with DVC or Git LFS if required)
models/*.joblib
models/*.pkl
models/*.bin

# Data files (Preserve raw and processed data outside Git)
data/raw/*
!data/raw/.gitkeep
data/processed/*
!data/processed/.gitkeep

# Heavy generated tables
reports/shap_values.csv

# Environment variables & secrets
.env
.env.*
*.pem
*.key
*.token

# OS & Editor files
.DS_Store
Thumbs.db
.vscode/
.idea/
```

### Impact of Proposed `.gitignore`:
- **Before:** `git add .` stages ~103 MB of binary weights, datasets, and cache files.
- **After:** `git add .` stages only source code, documentation, lightweight scripts, summary reports/metrics, and figures (~1.5 MB total), keeping the Git history lightweight, clean, and fast.

---

## 5. Risk Assessment & Preservation Log

| Operation / Command | Execution Status | Risk Prevention |
| :--- | :--- | :--- |
| `git reset --hard` | **Never Executed** | No working tree modifications or data loss. |
| `git clean -fd` | **Never Executed** | All untracked artifacts, reports, and models preserved. |
| `git checkout` / `git restore` | **Never Executed** | Zero rollbacks of user work. |
| `git commit` | **Not Executed** | Awaiting explicit user direction. |
| File removals / overwrites | **None** | No existing files deleted or modified. |

---

## 6. Recommended Next Steps for the User

1. **Approve `.gitignore` update:**
   Allow the creation/update of `.gitignore` with the minimal rules specified in Section 4.
2. **Review tracked vs. ignored preferences:**
   Confirm whether you want:
   - Lightweight report CSVs (`final_test_metrics.csv`, `feature_importance.csv`, `confusion_matrix.csv`) and figures (`figures/*.png`) tracked in Git (recommended).
   - Datasets (`data/`) and model weights (`models/`) ignored by Git (recommended for standard repos unless Git LFS is configured).
3. **Execute Initial Commit:**
   Once `.gitignore` is verified, execute a clean initial commit to establish the baseline repository history.
