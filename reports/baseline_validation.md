# IncomeLens AI — Baseline Validation Report

**Audit & Validation Date:** October 20, 2026  
**Auditor:** Senior Machine Learning Engineer  
**Project Path:** `D:\incomelens-ai`  
**Report Destination:** `reports/baseline_validation.md`

---

## 1. Executive Summary

This report establishes a rigorous, mathematically verified, and reproducible performance baseline for the `incomelens-ai` project. 

### Key Baseline Findings:
1. **Test Suite Health:** All 7 existing tests pass cleanly (`7 passed in 7.09s`).
2. **Dependency Consistency:** `python -m pip check` confirms zero dependency conflicts (`No broken requirements found`).
3. **Artifact Integrity:** `models/random_forest_final.joblib` exists, loads without warnings, and contains a fitted 2-step `Pipeline` (`ColumnTransformer` $\rightarrow$ `RandomForestClassifier`).
4. **Data & Report Verification:** All raw datasets, processed splits (`train.csv`, `validation.csv`, `test.csv`), and generated report artifacts are present on disk.
5. **Side-Effect Inspection:** Existing scripts (`04`, `08`, `09`) contain automatic retraining routines that overwrite existing model binaries and reports. In accordance with safety directives, these scripts were **not executed directly**.
6. **100% Mathematical Reproducibility:** A non-destructive in-memory evaluation of `models/random_forest_final.joblib` against `data/processed/test.csv` reproduced the stored test metrics and confusion matrix with **0.0 difference across all metrics**.

---

## 2. Test Suite & Dependency Verification

### 2.1 Test Suite Run
- **Command:** `.\.venv\Scripts\python.exe -m pytest tests -v`
- **Output:**
  ```text
  ============================= test session starts =============================
  platform win32 -- Python 3.14.7, pytest-9.1.1, pluggy-1.7.0 -- D:\incomelens-ai\.venv\Scripts\python.exe
  cachedir: .pytest_cache
  rootdir: D:\incomelens-ai
  collecting ... collected 7 items

  tests/test_data.py::test_training_data_exists PASSED                     [ 14%]
  tests/test_data.py::test_training_data_loads PASSED                      [ 28%]
  tests/test_data.py::test_training_data_has_no_duplicate_columns PASSED   [ 42%]
  tests/test_data.py::test_income_target_is_present PASSED                 [ 57%]
  tests/test_predictions.py::test_model_file_exists PASSED                 [ 71%]
  tests/test_predictions.py::test_model_loads PASSED                       [ 85%]
  tests/test_predictions.py::test_model_predictions_are_valid PASSED       [100%]

  ============================== 7 passed in 7.09s ==============================
  ```

### 2.2 Pip Dependency Check
- **Command:** `.\.venv\Scripts\python.exe -m pip check`
- **Output:**
  ```text
  No broken requirements found.
  ```

---

## 3. Production Model Artifact Verification

### 3.1 File Metadata
- **File Path:** `D:\incomelens-ai\models\random_forest_final.joblib`
- **Size:** 50,084,507 bytes (~50.08 MB)
- **Last Modified Timestamp:** `10/20/2026 1:26:57 PM`

### 3.2 Loaded Pipeline Architecture
- **Object Type:** `sklearn.pipeline.Pipeline`
- **Pipeline Steps:**
  1. `preprocessor`: `sklearn.compose.ColumnTransformer`
     - Numeric transformer (6 input columns $\rightarrow$ `SimpleImputer(strategy="median")` $\rightarrow$ `StandardScaler()`)
     - Categorical transformer (8 input columns $\rightarrow$ `SimpleImputer(strategy="most_frequent")` $\rightarrow$ `OneHotEncoder(handle_unknown="ignore", sparse_output=True)`)
     - Total transformed output features: **105 features**
  2. `classifier`: `sklearn.ensemble.RandomForestClassifier`
     - `n_estimators`: `150`
     - `min_samples_leaf`: `2`
     - `random_state`: `42`
     - `n_jobs`: `-1`
     - `classes_`: `[0, 1]` (0 = `<=50K`, 1 = `>50K`)

---

## 4. Existing Datasets & Evaluation Reports Inventory

### 4.1 Datasets Inventory
| File Path | Rows | Columns | File Size (Bytes) | Last Modified |
| :--- | :--- | :--- | :--- | :--- |
| `data/raw/adult.csv` | 48,842 | 15 | 5,336,181 | 10/20/2026 8:29:28 AM |
| `data/processed/adult_cleaned.csv` | 48,842 | 15 | 5,319,900 | 10/20/2026 8:30:38 AM |
| `data/processed/train.csv` | 29,301 | 15 | 3,190,538 | 10/20/2026 10:23:48 AM |
| `data/processed/validation.csv` | 9,770 | 15 | 1,064,781 | 10/20/2026 10:23:48 AM |
| `data/processed/test.csv` | 9,771 | 15 | 1,064,895 | 10/20/2026 10:23:48 AM |

### 4.2 Evaluation Reports Inventory
| File Path | Description | File Size (Bytes) | Last Modified |
| :--- | :--- | :--- | :--- |
| `reports/final_test_metrics.csv` | Final test metrics of selected model | 205 | 10/20/2026 1:26:57 PM |
| `reports/confusion_matrix.csv` | Confusion matrix on held-out test set | 78 | 10/20/2026 1:26:57 PM |
| `reports/grouped_validation_comparison.csv` | LR vs RF validation metrics on grouped split | 291 | 10/20/2026 10:25:01 AM |
| `reports/validation_model_comparison.csv` | LR vs RF validation metrics on random split | 292 | 10/20/2026 8:36:13 AM |
| `reports/feature_importance.csv` | All 105 MDI feature importances | 6,054 | 10/20/2026 1:25:55 PM |
| `reports/shap_values.csv` | 1,000-sample TreeSHAP value table | 2,341,543 | 10/20/2026 12:34:45 PM |
| `reports/figures/feature_importance.png` | Top 20 MDI features bar plot | 151,883 | 10/20/2026 1:25:56 PM |
| `reports/figures/shap_summary.png` | Global TreeSHAP beeswarm plot | 372,816 | 10/20/2026 12:34:48 PM |

---

## 5. Script Side-Effect & Safety Analysis

Before executing any evaluation scripts, their source code was audited for side effects:

1. **`scripts/04_evaluate_model.py`:**
   - *Behavior:* Splits `adult_cleaned.csv` into a random train/val/test split, fits new `LogisticRegression` and `RandomForestClassifier` pipelines, and overwrites `reports/validation_model_comparison.csv`.
   - *Verdict:* **Unsafe to run** (causes unintended retraining and overwrites historical report).
2. **`scripts/08_train_grouped_models.py`:**
   - *Behavior:* Trains new `LogisticRegression` and `RandomForestClassifier` pipelines on `train.csv`, overwrites `models/logistic_regression_grouped.joblib` and `models/random_forest_grouped.joblib`, and overwrites `reports/grouped_validation_comparison.csv`.
   - *Verdict:* **Unsafe to run** (causes unintended retraining and overwrites models/reports).
3. **`scripts/09_final_evaluate.py`:**
   - *Behavior:* Fits a new `RandomForestClassifier` on `train.csv + validation.csv`, overwrites `models/random_forest_final.joblib`, overwrites `reports/final_test_metrics.csv`, and overwrites `reports/confusion_matrix.csv`.
   - *Verdict:* **Unsafe to run** (retrains production model from scratch and overwrites outputs).
4. **`scripts/10_feature_importance.py`:**
   - *Behavior:* Loads `models/random_forest_final.joblib` without retraining, but recomputes importances and overwrites `reports/feature_importance.csv` and `reports/figures/feature_importance.png`.
   - *Verdict:* Overwrites existing figures and report CSV.

**Protocol Applied:** In strict adherence to the project instructions, destructive script execution was skipped. Instead, a verified read-only in-memory evaluation of the loaded production model was conducted.

---

## 6. Non-Destructive In-Memory Baseline Verification

### 6.1 Final Test Evaluation Reproduction (`models/random_forest_final.joblib` on `test.csv`)
Using the loaded `random_forest_final.joblib` model pipeline directly evaluated on `data/processed/test.csv` (9,771 instances):

| Metric | Computed from Loaded Model | Stored in `final_test_metrics.csv` | Absolute Difference |
| :--- | :--- | :--- | :--- |
| **Accuracy** | `0.8682836966533620` | `0.8682836966533620` | **0.0000000000000000** |
| **Balanced Accuracy** | `0.7808959995607743` | `0.7808959995607743` | **0.0000000000000000** |
| **Precision (>50K)** | `0.7710049423393740` | `0.7710049423393740` | **0.0000000000000000** |
| **Recall (>50K)** | `0.6174142480211082` | `0.6174142480211082` | **0.0000000000000000** |
| **F1 Score (>50K)** | `0.6857142857142857` | `0.6857142857142857` | **0.0000000000000000** |
| **ROC-AUC** | `0.9180519759941502` | `0.9180519759941502` | **0.0000000000000000** |

### 6.2 Confusion Matrix Reproduction
- **Computed Confusion Matrix:**
  ```text
                Predicted <=50K  Predicted >50K
  Actual <=50K             7080             417
  Actual >50K               870            1404
  ```
- **Stored Confusion Matrix (`reports/confusion_matrix.csv`):**
  ```text
                predicted_<=50K  predicted_>50K
  actual_<=50K             7080             417
  actual_>50K               870            1404
  ```
- **Match:** **Identical (100% agreement)**.

### 6.3 Validation Set Reproduction (`validation.csv`)
Evaluating the stored grouped validation models on `data/processed/validation.csv` (9,770 instances):

| Model | Accuracy | Precision (>50K) | Recall (>50K) | F1 (>50K) | ROC-AUC | Stored Match |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `logistic_regression_grouped.joblib` | `0.84953941` | `0.74118257` | `0.59541667` | `0.66035120` | `0.90668620` | **Exact** |
| `random_forest_grouped.joblib` | `0.86632549` | `0.79631636` | `0.61250000` | `0.69241639` | `0.91879365` | **Exact** |

---

## 7. Baseline Benchmark Summary for Tracking

The definitive baseline metrics to benchmark future architectural or feature engineering iterations against are:

```ini
[Production Baseline: Random Forest (n=150, min_samples_leaf=2)]
Dataset = UCI Adult Census Income (Grouped 60/20/20 Split)
Test Instances = 9771
Accuracy = 86.83%
Balanced Accuracy = 78.09%
Precision (>50K) = 77.10%
Recall (>50K) = 61.74%
F1-Score (>50K) = 68.57%
ROC-AUC = 91.81%
Specificity (TNR) = 94.44%
False Positive Rate = 5.56%
False Negative Rate = 38.26%
```

---

## 8. Conclusion & Sign-Off

The baseline state of `incomelens-ai` is **fully verified, uncorrupted, and mathematically reproducible**. The saved models faithfully generate the recorded report outputs, the test suite is green, dependencies are unbroken, and no artifacts or source files were modified or overwritten during this audit.
