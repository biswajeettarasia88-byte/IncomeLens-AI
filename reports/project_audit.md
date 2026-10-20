# IncomeLens AI — Comprehensive Project Audit Report

**Audit Date:** October 20, 2026  
**Auditor:** Senior Machine Learning Engineer  
**Project Path:** `D:\incomelens-ai`  
**Target Goal:** Production-grade Census Income Classification (`<=50K` vs `>50K`) with rigorous evaluation, explainability, fairness audits, and interactive deployment.

---

## 1. Executive Summary

An exhaustive technical audit of the `incomelens-ai` repository was performed. The repository represents an active, partially completed machine learning project in the tabular classification domain.

### Confirmed Status Summary
- **Data Engineering & Splitting:** **Substantially Complete & Methodologically Sound.** Data extraction, label cleaning, duplicate/conflict analysis, and a leak-free grouped 60/20/20 train/validation/test split have been implemented and executed.
- **Model Training & Evaluation:** **Partially Complete.** Baseline dummy, Logistic Regression, and Random Forest pipelines were constructed and evaluated. The final selected model (`RandomForestClassifier`, 150 estimators, `min_samples_leaf=2`) achieves **86.83% accuracy** and **0.9181 ROC-AUC** on the held-out test set.
- **Explainability:** **Prototyped via Scripts.** Global TreeSHAP summary plots and Gini feature importances have been generated. However, the module is structured as a standalone script rather than a reusable service.
- **Software Architecture & Modularity:** **Incomplete / Fragmented.** Critical library modules (`data.py`, `training.py`, `evaluation.py`, `fairness.py`, `__init__.py`) are empty (0 bytes). Business logic is currently locked in numbered procedural scripts under `scripts/`.
- **Fairness & Bias Auditing:** **Completely Missing (0%).** Neither `src/incomelens/fairness.py` nor fairness audit scripts or metrics exist, despite sensitive demographic features (`sex`, `race`, `age`) being active predictors in the model.
- **Interactive UI (Streamlit):** **Completely Missing (0%).** All 5 application files under `app/` are 0-byte placeholders, and `streamlit` is absent from `requirements.txt`.
- **Packaging, Governance, & Testing:** **High Risk.** There is no `pyproject.toml` or `setup.py`, `.gitignore` is 0 bytes (risking multi-megabyte model and dataset commits), documentation files are blank, and test coverage is limited to basic data/model existence checks.

---

## 2. Repository Inventory & File Verification

Every file in the repository was inspected directly. The table below distinguishes confirmed facts verified from disk and commands from empty skeleton files.

| File / Directory | Size (Bytes) | Verification Status | Implementation State & Contents |
| :--- | :--- | :--- | :--- |
| **Root Files** | | | |
| `README.md` | 0 | Confirmed empty | Blank placeholder. No setup or usage instructions. |
| `.gitignore` | 0 | Confirmed empty | Blank placeholder. **Critical risk** for accidental git tracking of venvs, models, data. |
| `requirements.txt` | 72 | Confirmed | Contains 8 packages (`pandas`, `numpy`, `scikit-learn`, `joblib`, `matplotlib`, `seaborn`, `pytest`, `shap`). Missing `ucimlrepo`, `streamlit`. |
| `pyproject.toml` / `setup.py` | *Missing* | Confirmed absent | Package is not installable via `pip install -e .`. Scripts rely on `sys.path.insert`. |
| **Source Modules (`src/incomelens/`)** | | | |
| `src/incomelens/__init__.py` | 0 | Confirmed empty | Blank placeholder. Package root uninitialized. |
| `src/incomelens/data.py` | 0 | Confirmed empty | Blank placeholder. Data loading/splitting logic resides only in `scripts/`. |
| `src/incomelens/preprocessing.py` | 1,992 | Confirmed complete | Implements `TARGET_COLUMN`, `NUMERIC_FEATURES`, `CATEGORICAL_FEATURES`, `build_preprocessor()`, and `split_features_target()`. |
| `src/incomelens/training.py` | 0 | Confirmed empty | Blank placeholder. Training logic duplicated in scripts. |
| `src/incomelens/evaluation.py` | 0 | Confirmed empty | Blank placeholder. Evaluation metrics duplicated in scripts. |
| `src/incomelens/explainability.py` | 5,198 | Confirmed functional script | Standalone script with `main()`. Executes TreeSHAP sampling and saves PNG/CSV. Not modular. |
| `src/incomelens/fairness.py` | 0 | Confirmed empty | Blank placeholder. Zero fairness code. |
| **Scripts (`scripts/`)** | | | |
| `scripts/01_audit_data.py` | 1,560 | Confirmed functional | Fetches UCI Adult (id=2), cleans whitespace, maps `?` to `NA`, saves `data/raw/adult.csv`. |
| `scripts/02_prepare_data.py` | 2,162 | Confirmed functional | Normalizes target labels, audits missing values and types, saves `data/processed/adult_cleaned.csv`. |
| `scripts/03_train_baseline.py` | 2,696 | Confirmed functional | Fits DummyClassifier and baseline Logistic Regression on stratified 80/20 split. |
| `scripts/04_evaluate_model.py` | 3,195 | Confirmed functional | Compares LR and RF on random 60/20/20 train/val/test split. |
| `scripts/05_check_duplicates.py` | 1,299 | Confirmed functional | Audits exact duplicates (52) and identical feature vectors (111). |
| `scripts/06_inspect_conflicts.py` | 899 | Confirmed functional | Audits 5 distinct feature vectors with contradictory income labels. |
| `scripts/07_make_grouped_split.py` | 2,271 | Confirmed functional | Creates leak-free 60/20/20 grouped split using hashed feature groups. |
| `scripts/08_train_grouped_models.py` | 2,884 | Confirmed functional | Trains LR and RF on grouped train set, compares on grouped validation set. |
| `scripts/09_final_evaluate.py` | 3,377 | Confirmed functional | Trains final RF on dev set (train+val), evaluates on test set, serializes model and metrics. |
| `scripts/10_feature_importance.py` | 1,853 | Confirmed functional | Computes MDI feature importance from final model, generates CSV and plot. |
| **Tests (`tests/`)** | | | |
| `tests/test_data.py` | 773 | Confirmed (4 tests) | Tests existence, loading, uniqueness of columns, and target presence in `data/processed/train.csv`. |
| `tests/test_predictions.py` | 849 | Confirmed (3 tests) | Tests existence, loading, and batch inference of `models/random_forest_final.joblib`. |
| **Streamlit Application (`app/`)** | | | |
| `app/Home.py` | 0 | Confirmed empty | Blank placeholder. |
| `app/pages/1_Predict_Income.py` | 0 | Confirmed empty | Blank placeholder. |
| `app/pages/2_Model_Comparison.py` | 0 | Confirmed empty | Blank placeholder. |
| `app/pages/3_Explainability.py` | 0 | Confirmed empty | Blank placeholder. |
| `app/pages/4_Fairness_Audit.py` | 0 | Confirmed empty | Blank placeholder. |
| **Documentation & Notebooks** | | | |
| `docs/dataset_card.md` | 1,370 | Confirmed complete | High-quality dataset card documenting Adult census data, quality audit, and limitations. |
| `docs/experiment_protocol.md` | 0 | Confirmed empty | Blank placeholder. |
| `docs/literature_review.csv` | 0 | Confirmed empty | Blank placeholder. |
| `docs/problem_statement.md` | 0 | Confirmed empty | Blank placeholder. |
| `notebooks/01_initial_eda.ipynb` | 0 | Confirmed empty | Blank placeholder. |
| `reports/model_card.md` | 0 | Confirmed empty | Blank placeholder. |
| **Data Artifacts (`data/`)** | | | |
| `data/raw/adult.csv` | 5,336,181 | Confirmed (48,842 rows) | Raw UCI Adult dataset with 15 columns. |
| `data/processed/adult_cleaned.csv` | 5,319,900 | Confirmed (48,842 rows) | Standardized target labels (`<=50K`, `>50K`). |
| `data/processed/train.csv` | 3,190,538 | Confirmed (29,301 rows) | Grouped training split (~60%). |
| `data/processed/validation.csv` | 1,064,781 | Confirmed (9,770 rows) | Grouped validation split (~20%). |
| `data/processed/test.csv` | 1,064,895 | Confirmed (9,771 rows) | Grouped held-out test split (~20%). |
| **Model Artifacts (`models/`)** | | | |
| `models/logistic_regression.joblib` | 8,106 | Confirmed | Baseline LR from random split (script 03). |
| `models/logistic_regression_grouped.joblib` | 8,106 | Confirmed | Grouped validation LR (script 08). |
| `models/random_forest_grouped.joblib` | 39,018,267 | Confirmed | Grouped validation RF (script 08). |
| `models/random_forest_final.joblib` | 50,084,507 | Confirmed | Final production RF pipeline (preprocessor + classifier, script 09). |
| **Report Artifacts (`reports/`)** | | | |
| `reports/validation_model_comparison.csv` | 292 | Confirmed | LR vs RF comparison on random validation split. |
| `reports/grouped_validation_comparison.csv` | 291 | Confirmed | LR vs RF comparison on grouped validation split. |
| `reports/final_test_metrics.csv` | 205 | Confirmed | Final RF test set performance metrics. |
| `reports/confusion_matrix.csv` | 78 | Confirmed | Final RF test set confusion matrix. |
| `reports/feature_importance.csv` | 6,054 | Confirmed | Ranked MDI feature importances across all 105 transformed features. |
| `reports/shap_values.csv` | 2,341,543 | Confirmed | TreeSHAP values for 1,000 sampled instances across 105 features. |
| `reports/figures/feature_importance.png` | 151,883 | Confirmed | Bar chart of top 20 MDI features. |
| `reports/figures/shap_summary.png` | 372,816 | Confirmed | Beeswarm SHAP summary plot. |

---

## 3. Data Architecture, Quality, & Splitting Pipeline

### 3.1 Feature & Target Schema
- **Target Column:** `income`
  - Values normalized to binary classification: `<=50K` (mapped to `0`) and `>50K` (mapped to `1`).
  - Class distribution across entire dataset (48,842 rows):
    - `<=50K`: 37,155 (76.07%)
    - `>50K`: 11,687 (23.93%)
- **Numeric Features (6 columns):**
  1. `age`: int / float
  2. `fnlwgt` (final sampling weight): int / float
  3. `education-num` (years of education): int / float
  4. `capital-gain`: int / float
  5. `capital-loss`: int / float
  6. `hours-per-week`: int / float
- **Categorical Features (8 columns):**
  1. `workclass`: Private, Self-emp-not-inc, Self-emp-inc, Federal-gov, Local-gov, State-gov, Without-pay, Never-worked (contains missing values).
  2. `education`: Bachelors, Some-college, 11th, HS-grad, Prof-school, Assoc-acdm, Assoc-voc, 9th, 7th-8th, 12th, Masters, 1st-4th, 10th, Doctorate, 5th-6th, Preschool.
  3. `marital-status`: Married-civ-spouse, Divorced, Never-married, Separated, Widowed, Married-spouse-absent, Married-AF-spouse.
  4. `occupation`: Tech-support, Craft-repair, Other-service, Sales, Exec-managerial, Prof-specialty, Handlers-cleaners, Machine-op-inspct, Adm-clerical, Farming-fishing, Transport-moving, Priv-house-serv, Protective-serv, Armed-Forces (contains missing values).
  5. `relationship`: Wife, Own-child, Husband, Not-in-family, Other-relative, Unmarried.
  6. `race`: White, Asian-Pac-Islander, Amer-Indian-Eskimo, Other, Black.
  7. `sex`: Female, Male.
  8. `native-country`: 41 countries / regions (contains missing values).

### 3.2 Missing Value Audit
In `data/processed/adult_cleaned.csv`, missing values exist only in three categorical columns:
- `workclass`: 2,799 missing (5.73%)
- `occupation`: 2,809 missing (5.75%)
- `native-country`: 857 missing (1.75%)
All numeric features have zero missing values in the existing raw dataset.

### 3.3 Data Leakage Prevention & Grouped Splitting
A notable technical strength discovered during the audit is the rigorous duplicate and conflict analysis conducted in scripts `05`, `06`, and `07`:
- **Exact Duplicates:** 52 rows are identical across all 14 features and target.
- **Repeated Feature Vectors:** 111 rows have identical feature combinations.
- **Conflicting Labels:** 5 distinct feature combinations have contradictory target labels (i.e. identical demographics where one record earns `<=50K` and another earns `>50K`).
- **Grouped Split Solution (`scripts/07_make_grouped_split.py`):**
  To prevent identical individuals from landing in both train and test partitions (which would falsely inflate test metrics), the pipeline hashes the 14 feature columns to create group IDs and applies `GroupShuffleSplit`:
  - `data/processed/train.csv`: 29,301 rows (59.99%) — 22,288 `<=50K`, 7,013 `>50K` (23.93% positive)
  - `data/processed/validation.csv`: 9,770 rows (20.00%) — 7,370 `<=50K`, 2,400 `>50K` (24.56% positive)
  - `data/processed/test.csv`: 9,771 rows (20.00%) — 7,497 `<=50K`, 2,274 `>50K` (23.27% positive)
  Disjointness between train, validation, and test feature groups is verified and enforced via assertions.

---

## 4. Modeling Pipeline Architecture & Artifacts

### 4.1 Production Pipeline Specification (`models/random_forest_final.joblib`)
The serialised production model is a scikit-learn `Pipeline` composed of two stages:

```
[Input Data: 14 Columns]
          │
          ▼
┌─────────────────────────────────────────────────────────────┐
│ Step 1: ColumnTransformer ("preprocessor")                   │
│                                                             │
│ ├─ "numeric" (6 features):                                  │
│ │   SimpleImputer(strategy="median")                        │
│ │   StandardScaler()                                        │
│ │                                                           │
│ └─ "categorical" (8 features):                              │
│     SimpleImputer(strategy="most_frequent")                 │
│     OneHotEncoder(handle_unknown="ignore", sparse_output=T) │
│                                                             │
│ Output: 105 transformed sparse features                     │
└─────────────────────────────────────────────────────────────┘
          │
          ▼
┌─────────────────────────────────────────────────────────────┐
│ Step 2: RandomForestClassifier ("classifier")               │
│                                                             │
│   n_estimators = 150                                        │
│   min_samples_leaf = 2                                      │
│   random_state = 42                                         │
│   n_jobs = -1                                               │
│   classes_ = [0, 1]                                         │
└─────────────────────────────────────────────────────────────┘
```

### 4.2 Training Strategy
The final model was trained using `scripts/09_final_evaluate.py` on the combined **development set** (`train.csv` + `validation.csv` = 39,071 rows) and evaluated strictly once against the untouched held-out `test.csv` (9,771 rows).

---

## 5. Empirical Results & Verified Outputs

### 5.1 Validation Model Selection
Prior to final test evaluation, models were compared on validation splits:

**Ungrouped Validation Split (`reports/validation_model_comparison.csv`):**
| Model | Accuracy | Precision (>50K) | Recall (>50K) | F1 (>50K) | ROC-AUC |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Logistic Regression | 0.8520 | 0.7238 | 0.6166 | 0.6659 | 0.9037 |
| Random Forest | 0.8608 | 0.7554 | 0.6183 | 0.6800 | 0.9157 |

**Grouped Validation Split (`reports/grouped_validation_comparison.csv`):**
| Model | Accuracy | Precision (>50K) | Recall (>50K) | F1 (>50K) | ROC-AUC |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Logistic Regression | 0.8495 | 0.7412 | 0.5954 | 0.6604 | 0.9067 |
| Random Forest | **0.8663** | **0.7963** | **0.6125** | **0.6924** | **0.9188** |

### 5.2 Final Test Set Performance (`reports/final_test_metrics.csv`)
Evaluated strictly on `data/processed/test.csv` (9,771 rows):
- **Accuracy:** `0.8683` (86.83%)
- **Balanced Accuracy:** `0.7809` (78.09%)
- **Precision (>50K):** `0.7710` (77.10%)
- **Recall (>50K):** `0.6174` (61.74%)
- **F1 Score (>50K):** `0.6857` (68.57%)
- **ROC-AUC:** `0.9181` (91.81%)

### 5.3 Test Set Confusion Matrix (`reports/confusion_matrix.csv`)
| Ground Truth \ Predicted | Predicted `<=50K` (0) | Predicted `>50K` (1) | Total Actual |
| :--- | :--- | :--- | :--- |
| **Actual `<=50K` (0)** | **7,080** (TN) | **417** (FP) | 7,497 |
| **Actual `>50K` (1)** | **870** (FN) | **1,404** (TP) | 2,274 |
| **Total Predicted** | 7,950 | 1,821 | 9,771 |

- **True Negative Rate (Specificity):** $7080 / 7497 = 94.44\%$
- **False Positive Rate:** $417 / 7497 = 5.56\%$
- **False Negative Rate:** $870 / 2274 = 38.26\%$ (reflects conservative predictions for high earners)

### 5.4 Top Feature Importances (`reports/feature_importance.csv`)
Ranked by Random Forest Mean Decrease in Impurity (MDI):
1. `numeric__capital-gain`: 0.1403 (14.03%)
2. `categorical__marital-status_Married-civ-spouse`: 0.1010 (10.10%)
3. `numeric__age`: 0.0953 (9.53%)
4. `numeric__education-num`: 0.0893 (8.93%)
5. `categorical__relationship_Husband`: 0.0812 (8.12%)
6. `numeric__hours-per-week`: 0.0638 (6.38%)
7. `numeric__fnlwgt`: 0.0623 (6.23%)
8. `categorical__marital-status_Never-married`: 0.0399 (3.99%)
9. `numeric__capital-loss`: 0.0391 (3.91%)
10. `categorical__occupation_Exec-managerial`: 0.0249 (2.49%)
11. `categorical__education_Bachelors`: 0.0206 (2.06%)
12. `categorical__relationship_Wife`: 0.0154 (1.54%)
13. `categorical__relationship_Not-in-family`: 0.0153 (1.53%)
14. `categorical__education_Masters`: 0.0136 (1.36%)
15. `categorical__sex_Female`: 0.0134 (1.34%)
16. `categorical__sex_Male`: 0.0121 (1.21%)

---

## 6. Verified Test Suite Execution

Pytest was executed against the existing test suite in `.venv`:
```
platform win32 -- Python 3.14.7, pytest-9.1.1, pluggy-1.7.0
rootdir: D:\incomelens-ai

tests/test_data.py::test_training_data_exists PASSED                     [ 14%]
tests/test_data.py::test_training_data_loads PASSED                      [ 28%]
tests/test_data.py::test_training_data_has_no_duplicate_columns PASSED   [ 42%]
tests/test_data.py::test_income_target_is_present PASSED                 [ 57%]
tests/test_predictions.py::test_model_file_exists PASSED                 [ 71%]
tests/test_predictions.py::test_model_loads PASSED                       [ 85%]
tests/test_predictions.py::test_model_predictions_are_valid PASSED       [100%]

============================== 7 passed in 1.83s ==============================
```

### Test Quality & Scope Assessment:
- **Strengths:** Verifies that training data exists, is well-formed, and that the serialised model successfully generates non-null class predictions on sample tabular input.
- **Critical Weaknesses:**
  - Tests do not cover unit logic in `src/incomelens/preprocessing.py`.
  - Tests rely on pre-existing physical disk artifacts (`data/processed/train.csv` and `models/random_forest_final.joblib`). If run in a fresh clone, the test suite crashes immediately.
  - Zero tests exist for edge-case handling (missing values, invalid categories, unseen columns, type coercion).
  - Zero tests exist for explainability or fairness.

---

## 7. Critical Architecture & Technical Debt Analysis

### 7.1 Architecture Disconnect: Library vs. Procedural Scripts
The project was originally structured to be a modular Python package (`src/incomelens/`), but the core ML lifecycle was implemented procedurally inside 10 standalone scripts in `scripts/`.
- `src/incomelens/data.py` (0 bytes): Data extraction, cleaning, and grouped splitting code is trapped in `scripts/01`, `02`, and `07`.
- `src/incomelens/training.py` (0 bytes): Model training logic is duplicated across `scripts/03`, `04`, `08`, and `09`.
- `src/incomelens/evaluation.py` (0 bytes): Metric calculation and confusion matrix generation are duplicated across `scripts/04`, `08`, and `09`.
- `src/incomelens/fairness.py` (0 bytes): No implementation exists.
- `src/incomelens/explainability.py`: Implemented, but written as a standalone script executing a 1,000-row batch run via `main()`, with top-level directory creation side-effects on `import`. It lacks callable API methods (e.g. `explain_instance(model, row)`) required for serving or application integration.

### 7.2 Missing Packaging & Path Fragility
There is no `pyproject.toml` or `setup.py`. As a result:
- The package cannot be installed via `pip install -e .`.
- Every script has to manually inject `sys.path.insert(0, str(ROOT / "src"))`.
- Importing `incomelens` from notebooks, external workers, or test runners fails unless `PYTHONPATH` is manually configured.

### 7.3 Git Repository & Version Control Hazards
- **Git status:** `No commits yet`. The repository has never had an initial commit.
- **Empty `.gitignore` (0 bytes):** Running `git add .` would inadvertently track:
  - Virtual environments (`.venv/`, `.venv-1/` containing hundreds of megabytes).
  - Serialised model binaries (`models/random_forest_final.joblib` is ~50.1 MB, `random_forest_grouped.joblib` is ~39.0 MB).
  - Cache directories (`.pytest_cache/`, `__pycache__/`).
  - Large processed datasets and SHAP CSV tables.

### 7.4 Fairness & Ethical AI Gap
The dataset card (`docs/dataset_card.md`) explicitly lists fairness evaluation as an intended project objective. Furthermore, the model directly ingests sensitive protected attributes:
- `sex` (Female, Male)
- `race` (White, Asian-Pac-Islander, Amer-Indian-Eskimo, Other, Black)
- `age`
- Strong gender proxies (`relationship`: Husband vs. Wife, `marital-status`: Married-civ-spouse)
Despite this, zero fairness metrics (Demographic Parity Ratio, Disparate Impact, Equal Opportunity Difference, False Positive Rate parity) have been implemented, calculated, or reported.

### 7.5 Unimplemented User Interface
The Streamlit multi-page application structure under `app/` is completely empty (all 5 files are 0 bytes):
- `app/Home.py`: 0 bytes
- `app/pages/1_Predict_Income.py`: 0 bytes
- `app/pages/2_Model_Comparison.py`: 0 bytes
- `app/pages/3_Explainability.py`: 0 bytes
- `app/pages/4_Fairness_Audit.py`: 0 bytes
Furthermore, `streamlit` is not installed in the active virtual environment and is omitted from `requirements.txt`.

---

## 8. Prioritized Defects & Risks

### Severity 1: Critical (Blockers to deployment, reliability, and reproducibility)
1. **Empty `.gitignore` with large binaries and virtual environments:**
   Running `git add .` will commit ~100MB of model binaries and hundreds of MB of virtual environments. A comprehensive `.gitignore` is immediately required.
2. **Missing packaging (`pyproject.toml`):**
   The project cannot be installed as an editable module, causing `ModuleNotFoundError: No module named 'incomelens'` unless ad-hoc `sys.path` hacks are maintained.
3. **Missing dependencies in `requirements.txt`:**
   `ucimlrepo` (needed by data extraction) and `streamlit` (needed by the UI) are missing from `requirements.txt`.
4. **Complete absence of Fairness Auditing:**
   A census income classification model utilizing `sex`, `race`, and `age` cannot be deployed responsibly without demographic parity and equalized odds audits.

### Severity 2: High (Architectural integrity & functionality gaps)
5. **Empty core modules (`data.py`, `training.py`, `evaluation.py`, `fairness.py`):**
   Code in `scripts/` must be refactored into testable, reusable functions in `src/incomelens/`.
6. **Non-modular explainability implementation:**
   `src/incomelens/explainability.py` must provide callable functions for single-instance waterfall/force explanations and global summaries, decoupled from hardcoded file writes.
7. **Completely empty Streamlit application:**
   All 5 files in `app/` are empty placeholders.

### Severity 3: Medium (Quality assurance & documentation debt)
8. **Test suite brittleness:**
   Tests only test pre-computed disk files. There are no unit tests for transformers, encoders, or error handling.
9. **Blank documentation:**
   `README.md`, `reports/model_card.md`, `docs/experiment_protocol.md`, and `docs/problem_statement.md` are empty 0-byte files.
10. **Redundant virtual environment:**
    `.venv-1` exists in the repository root containing only bare `pip`, adding clutter.

---

## 9. Recommended Next Steps (Action Plan)

### Phase 1: Foundation, Packaging & Hygiene
1. **Create `.gitignore`:** Exclude `.venv*`, `__pycache__`, `.pytest_cache`, large model `.joblib` files (or manage with DVC/Git LFS), large CSVs in `reports/` and `data/`, and IDE configurations.
2. **Add `pyproject.toml`:** Configure `incomelens` as an editable package with standard build metadata and dependency lists.
3. **Update `requirements.txt`:** Include `streamlit`, `ucimlrepo`, and optional dev/test packages.
4. **Make initial Git commit:** Establish clean version control tracking.

### Phase 2: Refactoring Core Modules (`src/incomelens/`)
1. **Implement `src/incomelens/data.py`:** Refactor ingestion, cleaning, and grouped splitting from `scripts/01`, `02`, and `07` into reusable functions (`fetch_raw_data()`, `clean_data()`, `create_grouped_splits()`).
2. **Implement `src/incomelens/training.py`:** Create modular pipeline builders and training runners (`train_model()`, `save_model()`).
3. **Implement `src/incomelens/evaluation.py`:** Centralize multi-metric evaluation (`evaluate_predictions()`, `compute_confusion_matrix()`, `format_classification_metrics()`).
4. **Implement `src/incomelens/fairness.py`:** Build metric calculators for Disparate Impact, Demographic Parity, Equal Opportunity Difference, and FPR/FNR subgroup metrics across `sex` and `race`.
5. **Refactor `src/incomelens/explainability.py`:** Provide `explain_instance()` for real-time inference and `explain_dataset()` for batch SHAP calculations.

### Phase 3: Fairness Audit & Documentation
1. **Generate Fairness Report:** Execute fairness audit on the final Random Forest model and record disparity metrics across gender and racial groups.
2. **Populate `reports/model_card.md`:** Document model architecture, intended uses, performance benchmarks, limitations, and fairness evaluation results.
3. **Populate `README.md`:** Add architecture diagrams, setup instructions, CLI execution guides, and Streamlit launch commands.

### Phase 4: Streamlit Interactive Application
1. **Implement `app/Home.py`:** Project overview, problem framing, and navigation.
2. **Implement `app/pages/1_Predict_Income.py`:** Interactive form to input individual census profiles and receive real-time income probability predictions with single-instance SHAP waterfall explanations.
3. **Implement `app/pages/2_Model_Comparison.py`:** Interactive dashboard comparing baseline dummy, Logistic Regression, and Random Forest metrics on grouped validation and test sets.
4. **Implement `app/pages/3_Explainability.py`:** Global feature importance and interactive SHAP beeswarm/dependence plots.
5. **Implement `app/pages/4_Fairness_Audit.py`:** Interactive fairness dashboard displaying subgroup error rates, demographic parity, and disparate impact ratios.

### Phase 5: Test Suite Expansion
1. Add unit tests for preprocessing transformers (handling unseen categories, median imputation).
2. Add unit tests for data validation and grouped split disjointness.
3. Add unit tests for fairness metrics and explainability helper functions.
