# IncomeLens-AI — Release Checklist

**Date:** October 26, 2026  
**Release Scope:** Local research demonstration and educational benchmarking use only.  
**External Deployment Status:** NOT AUTHORIZED — strictly confined to local execution.  
**Overall Release Verdict for Local Demo Use:** ✅ **PASS (13/13 checks passing; 0 blockers)**

---

> [!IMPORTANT]
> This checklist assesses readiness for **local demonstration and research use** only. The project is explicitly **NOT ready and NOT authorized** for production deployment, external hosting, credit underwriting, hiring, or any consequential decision-making affecting real individuals.

---

## Check-by-Check Assessment

---

### 1. Environment Reproducibility

**Status: ✅ PASS**

| Evidence Item | Result |
| :--- | :--- |
| `python -m pip check` | Exit 0: `No broken requirements found.` |
| Core imports (`pandas`, `numpy`, `sklearn`, `shap`, `joblib`, `matplotlib`, `streamlit`) | All load without error |
| Python version | 3.14.7 |
| `requirements.txt` present | ✅ 9 packages declared |
| Virtual environment isolation | `.venv/` excluded from git via `.gitignore` |

**Remaining Note:** `requirements.txt` uses unpinned specifiers. A `pip-compile` lockfile would strengthen reproducibility for handoff. Not blocking for local use.

---

### 2. Data Integrity

**Status: ✅ PASS**

| Evidence Item | Result |
| :--- | :--- |
| Raw data cleaned via `scripts/02_prepare_data.py` | Present: `data/processed/adult_cleaned.csv` |
| Exact duplicates | 52 exact duplicate rows — documented; no records silently removed |
| Conflicting labels | 5 feature combinations with conflicting income labels — documented; no silent imputation |
| Target nulls | 0 null values in `income` column across all splits |
| Schema columns | All 15 columns (14 features + target) present in all splits |
| `test_data.py` 41-test suite | ✅ 41/41 PASS |

**Source:** [`reports/data_integrity_audit.md`](file:///d:/incomelens-ai/reports/data_integrity_audit.md)

---

### 3. Split Integrity

**Status: ✅ PASS**

| Evidence Item | Result |
| :--- | :--- |
| Train rows | 29,313 (60.0%) |
| Validation rows | 9,758 (20.0%) |
| Test rows | 9,771 (20.0%) |
| Group disjointness (`fnlwgt`) | `test_split_group_disjointness` PASS — no household group leaks across splits |
| Column schema consistency | Train, validation, test share identical dtypes and columns |
| Target class labels | Both `<=50K` and `>50K` present in all three splits |
| Test partition immutability | `data/processed/test.csv` hash unchanged; not used in training |

**Source:** [`reports/split_validation.md`](file:///d:/incomelens-ai/reports/split_validation.md)

---

### 4. Model Integrity

**Status: ✅ PASS**

| Evidence Item | Result |
| :--- | :--- |
| Artifact path | `models/random_forest_final.joblib` |
| File size | 50,084,507 bytes |
| SHA-256 (computed) | `f164a61183466b25186377e8946f4cb74b2df924f69e50224f5d8de72858a285` |
| SHA-256 (manifest) | `f164a61183466b25186377e8946f4cb74b2df924f69e50224f5d8de72858a285` |
| Hash match | ✅ EXACT MATCH — no retraining occurred |
| Pipeline type | `sklearn.pipeline.Pipeline` |
| Pipeline steps | `preprocessor` (ColumnTransformer) + `classifier` (RandomForestClassifier) |
| `n_estimators` | 150 |
| `random_state` | 42 |
| `min_samples_leaf` | 2 |
| `n_features_out` (preprocessor) | 105 dense numeric features |

**Source:** [`reports/model_integrity.md`](file:///d:/incomelens-ai/reports/model_integrity.md), [`reports/artifact_manifest.json`](file:///d:/incomelens-ai/reports/artifact_manifest.json)

---

### 5. Evaluation Correctness

**Status: ✅ PASS**

All metrics independently reproduced from first principles on held-out test split ($N = 9,771$). Zero discrepancy against recorded baseline:

| Metric | Verified Score | Absolute Discrepancy vs Baseline |
| :--- | :---: | :---: |
| Accuracy | **0.8683** | 0.0000000000000000 |
| Balanced Accuracy | **0.7809** | 0.0000000000000000 |
| High-Income Precision | **0.7710** | 0.0000000000000000 |
| High-Income Recall | **0.6174** | 0.0000000000000000 |
| High-Income F1 | **0.6857** | 0.0000000000000000 |
| ROC-AUC | **0.9181** | 0.0000000000000000 |

**Confusion matrix:**
```
                  Predicted <=50K   Predicted >50K
Actual <=50K          7,080             417
Actual >50K             870           1,404
```

`scripts/09_final_evaluate.py` runs in non-training mode only (`joblib.load` + `model.predict` — no `fit` calls).

**Source:** [`reports/metric_consistency_audit.md`](file:///d:/incomelens-ai/reports/metric_consistency_audit.md)

---

### 6. Explainability Validation

**Status: ✅ PASS**

| Evidence Item | Result |
| :--- | :--- |
| SHAP values computed | `shap.TreeExplainer` on 1,000-sample training stratified slice |
| SHAP output shape | (1000, 105) — matches transformed feature dimensions |
| Feature name alignment | 105 SHAP columns aligned 1:1 with `preprocessor.get_feature_names_out()` |
| Exact additivity | $\max |\Delta| < 6 \times 10^{-8}$ confirmed |
| SHAP beeswarm plot | `reports/figures/shap_summary.png` (406,590 bytes, 300 DPI) |
| SHAP values CSV | `reports/shap_values.csv` (gitignored; generated on-demand) |
| MDI feature importance | `reports/feature_importance.csv` (105 features, sum = 1.0) |
| MDI plot | `reports/figures/feature_importance.png` (293,175 bytes, 300 DPI) |
| `test_explainability.py` | ✅ 16/16 PASS |

**Known methodological limitation noted (not a defect):** MDI incorrectly ranks `fnlwgt` #7 due to continuous feature cardinality bias. Documented in [`reports/feature_importance_methodology.md`](file:///d:/incomelens-ai/reports/feature_importance_methodology.md).

**Source:** [`reports/shap_validation.md`](file:///d:/incomelens-ai/reports/shap_validation.md), [`reports/explainability_report.md`](file:///d:/incomelens-ai/reports/explainability_report.md)

---

### 7. Fairness Evaluation

**Status: ✅ PASS**

| Evidence Item | Result |
| :--- | :--- |
| Groups audited | `sex` (Male/Female), `race` (5 groups), `age_group` (4 groups), `country_group` (US/Non-US) |
| Metrics computed | Selection rate, TPR, FPR, precision, balanced accuracy, disparate impact ratio, equal opportunity diff, Wilson 95% CI |
| `reports/fairness_metrics.csv` | 13 rows × 23 columns; all present and readable |
| EEOC 80% violations identified | Female (DIR=0.301), Black (DIR=0.457), Amer-Indian-Eskimo (DIR=0.378) |
| Proxy variable analysis | "Fairness through blindness" fallacy documented |
| `test_fairness.py` | ✅ 18/18 PASS |

**Source:** [`reports/fairness_report.md`](file:///d:/incomelens-ai/reports/fairness_report.md), [`reports/fairness_metrics.csv`](file:///d:/incomelens-ai/reports/fairness_metrics.csv)

---

### 8. Application Functionality

**Status: ✅ PASS**

| Page | Headless `AppTest` Result |
| :--- | :---: |
| `app/Home.py` | ✅ No uncaught exceptions |
| `app/pages/1_Predict_Income.py` | ✅ No uncaught exceptions |
| `app/pages/2_Model_Comparison.py` | ✅ No uncaught exceptions |
| `app/pages/3_Explainability.py` | ✅ No uncaught exceptions |
| `app/pages/4_Fairness_Audit.py` | ✅ No uncaught exceptions |

All 5 pages compile cleanly (`py_compile.compile` with `doraise=True`).

**Deprecation warning (non-blocking):** Streamlit 1.64 warns `use_container_width` will be removed after 2025-12-31. Application currently functional; update before next Streamlit version upgrade.

**Launch command:** `streamlit run app/Home.py` → serves at `http://localhost:8501`

**Source:** [`reports/deployment_readiness.md`](file:///d:/incomelens-ai/reports/deployment_readiness.md)

---

### 9. Automated Testing

**Status: ✅ PASS**

```
python -m pytest tests -q --tb=no
146 passed, 200 warnings in 62.95s
```

| Attribute | Result |
| :--- | :--- |
| Total collected | 146 |
| Passed | **146** |
| Failed | **0** |
| Errors | **0** |
| Skipped | **0** |
| Warnings | 200 (all `InconsistentVersionWarning` — sklearn version mismatch; non-functional) |

**Source:** [`reports/test_quality_audit.md`](file:///d:/incomelens-ai/reports/test_quality_audit.md)

---

### 10. Security Review

**Status: ✅ PASS**

| Check | Result |
| :--- | :--- |
| Committed credentials scan | 0 secrets found |
| Git history scan | 0 secrets in commit `19698cd` |
| `subprocess` usage | 3 read-only git calls in `provenance.py`, static args, `shell=False` |
| `eval`/`exec`/`os.system` | **0 occurrences** |
| Input validation (`inference.py`) | Whitelist categories, numeric bounds, target injection rejection, sanitized errors |
| Dependency health (`pip check`) | `No broken requirements found.` |
| `.gitignore` coverage | `.venv/`, `.env`, `*.token`, `*.pem`, `models/*.joblib`, `data/raw/*` all excluded |
| Model provenance trust | Trained locally; SHA-256 tracked |

**Source:** [`reports/security_audit.md`](file:///d:/incomelens-ai/reports/security_audit.md)

---

### 11. Documentation

**Status: ✅ PASS**

| Document | Present | Content Verified |
| :--- | :---: | :---: |
| `README.md` (professional technical) | ✅ | All 13 sections; verified commands |
| `reports/model_card.md` | ✅ | All verified metrics; limitations stated |
| `docs/dataset_card.md` | ✅ | Dataset provenance |
| `docs/problem_statement.md` | ✅ | Formal classification problem |
| `docs/inference_interface_contract.md` | ✅ | Feature schema and validation rules |
| `reports/execution_audit.md` | ✅ | All commands tested |
| `reports/deployment_readiness.md` | ✅ | 10/10 PASS |
| `reports/final_validation.md` | ✅ | 10/10 PASS full regression audit |
| `reports/security_audit.md` | ✅ | Full security review |

**Source:** [`README.md`](file:///d:/incomelens-ai/README.md)

---

### 12. Artifact Provenance

**Status: ✅ PASS**

| Evidence Item | Result |
| :--- | :--- |
| `reports/artifact_manifest.json` | Present; valid JSON; parseable |
| Model SHA-256 recorded | `f164a61...285` |
| Model SHA-256 on disk | `f164a61...285` — ✅ MATCH |
| Training data splits recorded | ✅ |
| Evaluation metrics recorded | ✅ Accuracy, balanced accuracy, precision, recall, F1, ROC-AUC |
| Python version recorded | `3.14.7` |
| Key package versions recorded | sklearn, pandas, numpy, shap, joblib |
| Git commit hash | `null` (unborn branch at training time — see Defect D-02) |
| Timestamp | `2026-10-26T15:59:14.145650+00:00` |

**Source:** [`reports/artifact_manifest.json`](file:///d:/incomelens-ai/reports/artifact_manifest.json)

---

### 13. Deployment Readiness (Local Demonstration)

**Status: ✅ PASS**

All 10 deployment readiness checks verified in [`reports/deployment_readiness.md`](file:///d:/incomelens-ai/reports/deployment_readiness.md):

| Check | Status |
| :--- | :---: |
| Model & preprocessing availability | ✅ PASS |
| Installation instructions | ✅ PASS |
| Application startup | ✅ PASS |
| Input validation | ✅ PASS |
| Prediction correctness | ✅ PASS |
| Error handling | ✅ PASS |
| Report & artifact paths | ✅ PASS |
| Dependency installation | ✅ PASS |
| Test execution | ✅ PASS |
| Documentation & model limitations | ✅ PASS |

---

## Blocking Defects

**None.** There are no blocking defects preventing local demonstration use.

---

## Non-Blocking Defects & Recommendations

| ID | Severity | Description | Resolution |
| :- | :- | :--- | :--- |
| D-01 | Low | sklearn version mismatch: model serialized under 1.9.1; environment runs 1.9.0. Causes `InconsistentVersionWarning` on load. | Upgrade env to sklearn 1.9.1 or re-serialize model under 1.9.0. |
| D-02 | Low | `git_commit_hash` is `null` in provenance manifest; project has only one commit and trained on unborn branch state. | Commit working changes; re-run `src/incomelens/provenance.py` to capture real SHA. |
| D-03 | Low | `requirements.txt` uses unpinned specifiers. | Generate `requirements-lock.txt` via `pip-compile`. |
| D-04 | Low | Streamlit `use_container_width` deprecated since 1.64 (removal after 2025-12-31). | Replace with `width='stretch'` in all app pages before next Streamlit upgrade. |

---

## Release Verdict

| Scope | Verdict |
| :--- | :---: |
| **Local educational demonstration** | ✅ **READY** |
| **Local research benchmarking** | ✅ **READY** |
| **External / public deployment** | ❌ **NOT AUTHORIZED — requires separate explicit approval** |
| **Production decision-making system** | ❌ **PROHIBITED — see Model Card §2.2** |
