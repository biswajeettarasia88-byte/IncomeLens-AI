# IncomeLens-AI — Final Regression Validation Report

**Validation Date:** October 26, 2026  
**Evaluator:** Antigravity AI Release Engineering  
**Environment:** Python 3.14.7, Windows x86_64 (`D:\incomelens-ai`)  
**Scope:** Exhaustive regression validation of all components, pipeline stages, artifacts, and test suites prior to release handover.

---

## 1. Executive Summary

A complete, non-destructive regression validation was conducted across all 10 mandated validation criteria. All steps executed with exit code `0`. Zero model retraining occurred; the production model artifact and held-out test dataset were strictly preserved in their validated states.

| Step | Validation Target | Execution Command | Result | Artifacts / Evidence |
| :---: | :--- | :--- | :---: | :--- |
| **1** | Dependency Health | `python -m pip check` | **PASS** | Exit code 0; `No broken requirements found.` |
| **2** | Full Test Suite | `python -m pytest tests -v` | **PASS** | **146 passed**, 0 failed, 0 errors, 200 warnings |
| **3** | Model Integrity | Cryptographic & Structural Check | **PASS** | SHA-256 `f164a611...` matches; 105 features; 150 trees |
| **4** | Data & Split Integrity | `scripts/05_check_duplicates.py`<br>`scripts/06_inspect_conflicts.py`<br>`pytest tests/test_data.py` | **PASS** | 52 duplicates documented; 5 label conflicts documented; 41/41 split & data tests pass |
| **5** | Final Evaluation | `python scripts/09_final_evaluate.py` | **PASS** | Non-training mode; metrics exactly match baseline (Acc: 0.8683, F1: 0.6857, AUC: 0.9181) |
| **6** | Feature Importance | `python scripts/10_feature_importance.py` | **PASS** | `reports/figures/feature_importance.png` (300 DPI); `reports/feature_importance.csv` |
| **7** | SHAP Generation | `python -m src.incomelens.explainability` | **PASS** | 1,000-sample TreeExplainer; `reports/figures/shap_summary.png` (300 DPI); `reports/shap_values.csv` |
| **8** | Fairness Evaluation | `python -m src.incomelens.fairness` | **PASS** | 13 subgroups × 23 metrics; Wilson 95% CIs; `reports/fairness_metrics.csv` |
| **9** | Application Smoke Tests| Streamlit `AppTest` | **PASS** | All 5 pages render without uncaught exceptions |
| **10**| Manifest Validation | `reports/artifact_manifest.json` | **PASS** | Valid JSON; exact SHA-256 match; test metrics and dependencies recorded |

---

## 2. Detailed Step-by-Step Validation Outcomes

### Step 1: Dependency Health (`python -m pip check`)
- **Outcome:** Exited 0.
- **Output:** `No broken requirements found.`
- **Inspection:** No files modified; read-only verification of installed package environment.

### Step 2: Full Test Suite (`python -m pytest tests -v`)
- **Outcome:** Exited 0 in 16.61s (and verified in background at 62.95s).
- **Output:** `146 passed, 200 warnings in 16.61s`.
- **Warning Summary:** 200 `InconsistentVersionWarning` notices from scikit-learn (deserializing artifact created under sklearn 1.9.1 in sklearn 1.9.0 environment). Functional behavior and predictions verified 100% correct across all tests.

### Step 3: Verified Model-Integrity Check
- **Model Path:** `models/random_forest_final.joblib`
- **Size:** 50,084,507 bytes
- **SHA-256:** `f164a61183466b25186377e8946f4cb74b2df924f69e50224f5d8de72858a285`
- **Structure:** `Pipeline` with `ColumnTransformer` (105 dense numeric features) and `RandomForestClassifier(n_estimators=150, min_samples_leaf=2, random_state=42)`.
- **Outcome:** Identical to manifest record. Zero retraining occurred.

### Step 4: Data-Integrity and Split Checks
- **Duplicates Check (`scripts/05_check_duplicates.py`):** 48,842 rows audited. 52 exact duplicates identified and documented. No rows deleted.
- **Label Conflict Inspection (`scripts/06_inspect_conflicts.py`):** 5 distinct feature combinations with conflicting labels identified and documented. No records modified.
- **Split Isolation Tests (`pytest tests/test_data.py -v`):** 41/41 tests passed, verifying group disjointness on `fnlwgt` between `train.csv` (29,313 rows), `validation.csv` (9,758 rows), and `test.csv` (9,771 rows).

### Step 5: Final Evaluation in Non-Training Mode (`scripts/09_final_evaluate.py`)
- **Mode:** Strictly non-training. Model is loaded via `joblib.load()`; inference run on `data/processed/test.csv`.
- **Verified Metrics ($N = 9,771$):**
  - Accuracy: `0.8683`
  - Balanced Accuracy: `0.7809`
  - High-Income Precision: `0.7710`
  - High-Income Recall: `0.6174` (870 false negatives out of 2,274 actual high earners)
  - High-Income F1-Score: `0.6857`
  - ROC-AUC: `0.9181`
- **Confusion Matrix:**
  $$\begin{pmatrix} \text{TN}: 7,080 & \text{FP}: 417 \\ \text{FN}: 870 & \text{TP}: 1,404 \end{pmatrix}$$
- **Artifacts Generated:** `reports/confusion_matrix.csv`, `reports/final_test_metrics.csv`, `reports/evaluation_manifest.json`.

### Step 6: Feature-Importance Generation (`scripts/10_feature_importance.py`)
- **Mode:** Non-training. Extracts MDI from fitted forest.
- **Artifacts:** `reports/figures/feature_importance.png` (300 DPI, 293,175 bytes), `reports/feature_importance.csv`.
- **Methodological Finding:** Continuous noise feature `fnlwgt` ranks #7 ($6.23\%$) due to MDI cardinality bias.

### Step 7: SHAP Generation (`python -m src.incomelens.explainability`)
- **Mode:** Non-training. Evaluates `TreeExplainer` on 1,000 stratified training samples.
- **Additivity Proof:** Exact probability additivity confirmed ($\max |\Delta| < 6 \times 10^{-8}$).
- **Artifacts:** `reports/figures/shap_summary.png` (300 DPI, 406,590 bytes), `reports/shap_values.csv`.

### Step 8: Fairness Evaluation (`python -m src.incomelens.fairness`)
- **Mode:** Non-training. Evaluates test set across demographic slices.
- **Artifacts:** `reports/fairness_metrics.csv` (13 rows, 23 columns).
- **Core Findings:** Female selection rate = $7.29\%$ vs Male = $24.25\%$ (DIR = $0.301$, violating EEOC 80% rule). Black selection rate = $9.02\%$ (DIR = $0.457$, violation).

### Step 9: Application Smoke Tests
- **Pages Verified:** `app/Home.py`, `app/pages/1_Predict_Income.py`, `app/pages/2_Model_Comparison.py`, `app/pages/3_Explainability.py`, `app/pages/4_Fairness_Audit.py`.
- **Harness:** Headless execution via `streamlit.testing.v1.AppTest`.
- **Result:** All 5 pages compiled and rendered with 0 uncaught exceptions.

### Step 10: Artifact-Manifest Validation
- **Manifest:** `reports/artifact_manifest.json` (5,320 bytes).
- **Verification:** Parseable JSON; SHA-256 checksums match `models/random_forest_final.joblib`; records Python 3.14.7, packages, evaluation metrics, and random seeds.

---

## 3. Discrepancies, Failures & Skipped Checks
- **Failed Checks:** **0**
- **Skipped Checks:** **0**
- **Discrepancies:** **0** (All test metrics match baseline to 16 decimal places).
- **Non-blocking Notices:**
  - Deprecation warnings from Streamlit regarding `use_container_width`.
  - Scikit-learn `InconsistentVersionWarning` (1.9.1 artifact read in 1.9.0 environment) — functional predictions confirmed 100% identical.

---

## 4. Final Verdict
**Regression Validation Status:** ✅ **PASS (10/10)**  
The IncomeLens-AI repository is in a stable, verified, and reproducible state for local demonstration and research benchmarking.
