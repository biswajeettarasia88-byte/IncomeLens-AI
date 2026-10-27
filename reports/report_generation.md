# IncomeLens-AI — Evaluation & Explainability Artifact Generation Workflow

**Status:** Standardized & Verified  
**Date:** October 26, 2026  
**Auditor / Author:** Antigravity  
**Production Model Artifact:** [`models/random_forest_final.joblib`](file:///d:/incomelens-ai/models/random_forest_final.joblib)  

---

## 1. Principles & Governance

1. **Model Immutability:** Report generation and explainability analyses treat [`models/random_forest_final.joblib`](file:///d:/incomelens-ai/models/random_forest_final.joblib) strictly as **read-only**. None of the reporting or evaluation scripts retrain, re-fit, or overwrite the production model.
2. **Deterministic & Isolated Splits:**
   - Final evaluation (`scripts/09_final_evaluate.py`) and fairness audit (`src/incomelens/fairness.py`) run strictly on the held-out test partition ([`data/processed/test.csv`](file:///d:/incomelens-ai/data/processed/test.csv), $N = 9,771$).
   - Model selection comparison (`scripts/08_train_grouped_models.py`) evaluates exclusively on the validation partition ([`data/processed/validation.csv`](file:///d:/incomelens-ai/data/processed/validation.csv), $N = 9,770$).
   - Global SHAP explanation (`src/incomelens/explainability.py`) samples $N = 1,000$ instances from the training partition ([`data/processed/train.csv`](file:///d:/incomelens-ai/data/processed/train.csv), seed 42) to avoid test set snooping.
   - Feature importance (MDI) (`scripts/10_feature_importance.py`) directly extracts Gini importance from the fitted estimator trees.
3. **Publication-Quality Figures:** All plots are rendered with `matplotlib` at **300 DPI** resolution with `bbox_inches="tight"` and clear methodological caveats (distinguishing feature contribution from causation).
4. **Test Suite Independence:** Unit and integration tests in `tests/` execute independently using synthetic fixtures or temporary directories (`tmp_path`). Running `pytest` does not regenerate, overwrite, or mutate production report files.
5. **Output Verification:** All generation scripts automatically verify directory existence prior to writing and validate file non-emptiness and schema consistency after execution.

---

## 2. Canonical Artifact Registry

The table below catalogs every standardized evaluation and explainability artifact produced in `reports/`:

| Artifact Path | Generating Script / Module | Model Artifact | Data Split / Sample | Format & Schema | Resolution / Dimensions |
| :--- | :--- | :--- | :--- | :--- | :--- |
| [`reports/final_test_metrics.csv`](file:///d:/incomelens-ai/reports/final_test_metrics.csv) | `python scripts/09_final_evaluate.py` | `random_forest_final.joblib` | `data/processed/test.csv` ($N = 9,771$) | CSV: `accuracy, balanced_accuracy, precision_high_income, recall_high_income, f1_high_income, roc_auc` | 1 row, 6 metric columns |
| [`reports/confusion_matrix.csv`](file:///d:/incomelens-ai/reports/confusion_matrix.csv) | `python scripts/09_final_evaluate.py` | `random_forest_final.joblib` | `data/processed/test.csv` ($N = 9,771$) | CSV: `predicted_<=50K, predicted_>50K` indexed by `actual_<=50K, actual_>50K` | 2x2 matrix (sum = 9,771) |
| [`reports/evaluation_manifest.json`](file:///d:/incomelens-ai/reports/evaluation_manifest.json) | `python scripts/09_final_evaluate.py` | `random_forest_final.joblib` | `data/processed/test.csv` ($N = 9,771$) | JSON metadata recording model path, split name, sample count, positive class, and metrics | Structured JSON |
| [`reports/feature_importance.csv`](file:///d:/incomelens-ai/reports/feature_importance.csv) | `python scripts/10_feature_importance.py` | `random_forest_final.joblib` | Fitted forest trees (development split) | CSV: `feature, importance` | 105 rows (descending order, sum = 1.0) |
| [`reports/figures/feature_importance.png`](file:///d:/incomelens-ai/reports/figures/feature_importance.png) | `python scripts/10_feature_importance.py` | `random_forest_final.joblib` | Top 20 MDI features | PNG: Horizontal bar chart with non-causal disclaimer | **300 DPI** |
| [`reports/shap_values.csv`](file:///d:/incomelens-ai/reports/shap_values.csv) | `python -m src.incomelens.explainability` | `random_forest_final.joblib` | `data/processed/train.csv` ($N = 1,000$, seed 42) | CSV: 105 transformed feature columns | 1,000 rows × 105 columns |
| [`reports/figures/shap_summary.png`](file:///d:/incomelens-ai/reports/figures/shap_summary.png) | `python -m src.incomelens.explainability` | `random_forest_final.joblib` | $N = 1,000$ training sample | PNG: Bee-swarm summary plot of top 20 SHAP features | **300 DPI** |
| [`reports/fairness_metrics.csv`](file:///d:/incomelens-ai/reports/fairness_metrics.csv) | `python -m src.incomelens.fairness` | `random_forest_final.joblib` | `data/processed/test.csv` ($N = 9,771$) | CSV: 23 group parity & error rate columns across Sex, Race, Age, and Origin cohorts | 13 subgroup rows |
| [`reports/grouped_validation_comparison.csv`](file:///d:/incomelens-ai/reports/grouped_validation_comparison.csv) | `python scripts/08_train_grouped_models.py` | Grouped models | `data/processed/validation.csv` ($N = 9,770$) | CSV: `model, accuracy, precision_high_income, recall_high_income, f1_high_income, roc_auc` | 2 rows (LR vs RF) |
| [`reports/artifact_manifest.json`](file:///d:/incomelens-ai/reports/artifact_manifest.json) | `python -m src.incomelens.provenance` | `random_forest_final.joblib` | All datasets & reports | JSON: Provenance tracking recording SHA-256 hashes, hyperparameters, environment versions, random seeds, and artifact paths | Complete provenance schema |

---

## 3. Supported Report-Generation Workflow

To generate or update artifacts systematically, execute the following commands from the project root (`D:\incomelens-ai`):

### Step 1: Final Model Test Evaluation
Evaluates the frozen production model on the held-out test split ($N = 9,771$), producing the confusion matrix, summary metrics, and evaluation manifest:
```powershell
python scripts/09_final_evaluate.py
```
**Verification Check:**
- Confirms `reports/confusion_matrix.csv` exists and sums to 9,771.
- Confirms `reports/final_test_metrics.csv` has Accuracy = 0.8683, Precision = 0.7710, Recall = 0.6174, ROC-AUC = 0.9181.
- Confirms `reports/evaluation_manifest.json` is generated.

### Step 2: Feature Importance Analysis (MDI)
Extracts Mean Decrease in Impurity (Gini Importance) across the 105 transformed feature dimensions:
```powershell
python scripts/10_feature_importance.py
```
**Verification Check:**
- Confirms `reports/feature_importance.csv` has 105 features sorted in descending order and summing to 1.0.
- Confirms `reports/figures/feature_importance.png` is non-empty (>100 KB) and rendered at 300 DPI.

### Step 3: Global SHAP Explainability
Calculates exact probability-space TreeExplainer attributions on 1,000 training instances:
```powershell
python -m src.incomelens.explainability
```
**Verification Check:**
- Confirms `reports/shap_values.csv` exists (1,000 rows × 105 columns).
- Confirms `reports/figures/shap_summary.png` exists (>300 KB) and is rendered at 300 DPI.

### Step 4: Group Fairness & Demographic Disparity Audit
Audits selection rates, Wilson 95% confidence intervals, and disparate impact ratios across protected groups on the held-out test split:
```powershell
python -m src.incomelens.fairness
```
**Verification Check:**
- Confirms `reports/fairness_metrics.csv` exists with all 23 audit columns across Sex, Race, Age Group, and Country Group.
- Confirms Disparate Impact Ratio for Females is 0.3007 and TPR is 0.5286.

### Step 5: Provenance Manifest Generation
Generates the cryptographic provenance tracking record covering all datasets, models, parameters, seeds, environment versions, and report checksums:
```powershell
python -m src.incomelens.provenance
```
**Verification Check:**
- Confirms `reports/artifact_manifest.json` exists and is parseable JSON.
- Confirms SHA-256 digests are recorded for model artifact, data splits, and CSV/figure reports.
- Confirms zero sensitive raw records or credentials are leaked.

---

## 4. Post-Execution Verification Checklist

Every generation script incorporates an internal assertion or check that verifies:
1. Destination directories exist before file writes (`mkdir(parents=True, exist_ok=True)`).
2. Saved files are non-empty (`stat().st_size > 0`).
3. Images contain valid PNG magic headers (`\x89PNG\r\n\x1a\n`).
4. CSV tables contain non-null headers matching the exact expected column names.
5. In the event of any processing failure, explicit exceptions are raised and execution halts immediately without reporting false success.
