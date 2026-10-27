# IncomeLens AI — Model Selection Workflow & Candidate Audit Report

**Audit Date:** October 26, 2026  
**Auditor:** Senior Machine Learning Engineer  
**Project Path:** `D:\incomelens-ai`  
**Report Destination:** `reports/model_selection_audit.md`

---

## 1. Executive Summary & Audit Scope

This report audits the model-selection process, candidate algorithms, validation protocols, hyperparameter configurations, and class imbalance handling across the `incomelens-ai` repository.

### Key Audit Findings:
1. **Selection Integrity Confirmed:** Candidate model selection strictly utilized the **validation partition** (`validation.csv` = 9,770 rows), never the final test split (`test.csv` = 9,771 rows).
2. **Superiority of Selected Architecture:** The selected production model ([`RandomForestClassifier`](file:///D:/incomelens-ai/models/random_forest_final.joblib), 150 estimators, `min_samples_leaf=2`) strictly outperformed the baseline [`LogisticRegression`](file:///D:/incomelens-ai/models/logistic_regression_grouped.joblib) across every single validation metric:
   - Accuracy: **86.63%** vs. 84.95% (+1.68%)
   - High-Income Precision: **79.63%** vs. 74.12% (+5.51%)
   - High-Income Recall: **61.25%** vs. 59.54% (+1.71%)
   - High-Income F1: **69.24%** vs. 66.04% (+3.20%)
   - ROC-AUC: **0.9188** vs. 0.9067 (+1.21%)
3. **Consistent Preprocessing:** Both candidate models evaluated in [`scripts/08_train_grouped_models.py`](file:///D:/incomelens-ai/scripts/08_train_grouped_models.py) shared the exact same [`ColumnTransformer`](file:///D:/incomelens-ai/src/incomelens/preprocessing.py#L33) pipeline (105 transformed features).
4. **Key Defect Identified in Class Imbalance Handling:** Neither candidate configured `class_weight='balanced'` or applied decision-threshold calibration (`class_weight=None` across all models). As a result, both models are conservative on the minority high-income class (`>50K`), resulting in a 38.26% false negative rate.
5. **No Automated Hyperparameter Search:** Hyperparameters were chosen manually based on heuristic best practices (`min_samples_leaf=2`) rather than systematic cross-validated grid/random search.
6. **Artifact & Metric Preservation:** The existing production model (`models/random_forest_final.joblib`) and test metrics remain 100% preserved.

---

## 2. Candidate Models & Architectures

| Candidate Model | Estimator Type | Key Hyperparameters | Preprocessing Applied | Pipeline Composition |
| :--- | :--- | :--- | :--- | :--- |
| **Candidate 0: Dummy Baseline** | `DummyClassifier` | `strategy="most_frequent"` | None (Raw target) | Majority class rule (always predicts `<=50K`) |
| **Candidate 1: Logistic Regression** | `LogisticRegression` | `max_iter=1000`, `random_state=42`, $C=1.0$, `class_weight=None` | Imputation (median/mode) + Scaling + OneHotEncoder | `Pipeline([('preprocessor', ColumnTransformer), ('classifier', LogisticRegression)])` |
| **Candidate 2: Random Forest** | `RandomForestClassifier` | `n_estimators=150`, `min_samples_leaf=2`, `random_state=42`, `n_jobs=-1`, `class_weight=None` | Imputation (median/mode) + Scaling + OneHotEncoder | `Pipeline([('preprocessor', ColumnTransformer), ('classifier', RandomForestClassifier)])` |

---

## 3. Validation Comparison & Selection Evidence

### 3.1 Initial Random Validation Split (`reports/validation_model_comparison.csv`)
In [`scripts/04_evaluate_model.py`](file:///D:/incomelens-ai/scripts/04_evaluate_model.py), an initial stratified random split was evaluated:

| Model | Accuracy | Precision (>50K) | Recall (>50K) | F1-Score (>50K) | ROC-AUC |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Logistic Regression** | `0.8520` | `0.7238` | `0.6166` | `0.6659` | `0.9037` |
| **Random Forest** | `0.8608` | `0.7554` | `0.6183` | `0.6800` | `0.9157` |

### 3.2 Canonical Grouped Validation Split (`reports/grouped_validation_comparison.csv`)
In [`scripts/08_train_grouped_models.py`](file:///D:/incomelens-ai/scripts/08_train_grouped_models.py), models were evaluated on the leak-free grouped validation partition (`validation.csv`):

| Model | Accuracy | Precision (>50K) | Recall (>50K) | F1-Score (>50K) | ROC-AUC | Outperformance |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Logistic Regression** | `0.8495` | `0.7412` | `0.5954` | `0.6604` | `0.9067` | Baseline |
| **Random Forest** | **`0.8663`** | **`0.7963`** | **`0.6125`** | **`0.6924`** | **`0.9188`** | **+3.20% F1, +1.21% AUC** |

### 3.3 Selection Justification
- **Stated Objective:** Accurately classify annual census income (`<=50K` vs. `>50K`) with maximum discrimination and explainability.
- **Empirical Verdict:** Random Forest won across **all 5 tracked metrics** on the leak-free validation set. It demonstrated superior precision (+5.51%), better recall (+1.71%), higher overall discriminative ability (+1.21% ROC-AUC), and native compatibility with TreeSHAP explainability.

---

## 4. Methodological Strengths & Integrity Confirmations

1. **Test Set Isolation:**
   - In both evaluation scripts (`04` and `08`), the test set was strictly held out and never loaded.
   - Script 08 explicit directive: `"Test set was not loaded or evaluated. Choose a model using validation results."`
   - Only after Random Forest was chosen as the winning candidate was it fitted on the development set (`train + val`) and evaluated once on `test.csv`.
2. **Preprocessing Consistency:**
   - Both models ingested the exact same 14 raw features through identical `ColumnTransformer` instances.
   - Learned imputation statistics and category encodings were fitted strictly on the training partition.
3. **Leak-Free Partitioning:**
   - Feature-vector group hashing prevented near-duplicate demographic profiles from leaking across development and validation splits.

---

## 5. Identified Deficiencies & Technical Debt

### 5.1 Unaddressed Class Imbalance (`class_weight=None`)
- The income dataset exhibits a ~3.2:1 class imbalance (76.07% `<=50K` vs. 23.93% `>50K`).
- By running with default `class_weight=None`, the decision threshold defaults to $P(y=1) \ge 0.50$.
- **Consequence:** The model prioritizes specificity (94.44%) over sensitivity (61.74%), missing over 38% of high earners in the test set.

### 5.2 Feature Redundancy in Linear Models
- Both `education` (16 categorical levels) and `education-num` (ordinal integers 1–16) were included.
- For Logistic Regression, this creates perfect multicollinearity between the continuous scale and the 16 one-hot columns. While Random Forest handles redundant split candidates gracefully, linear model coefficients were destabilized.

### 5.3 Absence of Systematic Hyperparameter Optimization
- Random Forest hyperparameters were fixed heuristically (`n_estimators=150`, `min_samples_leaf=2`).
- Parameters such as `max_features`, `max_depth`, `min_samples_split`, and tree counts were never tuned systematically.

---

## 6. Proposed Controlled Improvement Experiments

Per the project directive, no automated retraining or expensive grid searches were executed during this audit. Instead, the following three targeted experiments are proposed for future iterations:

### Experiment 1: Cost-Sensitive & Threshold Tuning (Target: Minority Recall)
* **Hypothesis:** Adjusting class weights or calibrating the decision threshold will increase high-income recall from 61.7% to 70%+ while preserving an F1 score $\ge 0.68$.
* **Methodology:**
  1. Train `RandomForestClassifier` with `class_weight='balanced'` and `class_weight={0: 1.0, 1: 2.0}` on `train.csv`.
  2. Sweep classification threshold $\tau \in [0.25, 0.50]$ on `validation.csv` to maximize $F_1$ and $F_2$ scores.
  3. Validate chosen threshold $\tau^*$ on `validation.csv` before applying to test evaluation.

### Experiment 2: Feature Deduplication (Target: Model Efficiency)
* **Hypothesis:** Dropping the 16 one-hot encoded `education` categories while retaining ordinal continuous `education-num` will reduce dimensionality from 105 to 89 features without reducing ROC-AUC.
* **Methodology:**
  1. Modify preprocessor to exclude `education` from categorical pipelines.
  2. Evaluate feature-reduced Random Forest on `validation.csv`.
  3. Compare model size, training speed, and validation F1 against the 105-feature baseline.

### Experiment 3: Modern Gradient Boosted Trees Benchmark (LightGBM / XGBoost)
* **Hypothesis:** Histogram-based gradient boosting (e.g. LightGBM or `HistGradientBoostingClassifier`) will match or exceed Random Forest performance while producing a substantially smaller model artifact (current joblib is 50.1 MB).
* **Methodology:**
  1. Implement `build_hist_gradient_boosting_pipeline()` in `src/incomelens/training.py`.
  2. Fit on `train.csv`, evaluate on `validation.csv`.
  3. Compare accuracy, recall, F1, model serialization size, and inference latency.
