# SHAP Mathematical Consistency and Additivity Validation Report

**Author / Auditor:** Antigravity  
**Date:** October 26, 2026  
**Environment:** Python 3.14.7 | `shap` 0.54.1 | `scikit-learn` 1.9.1 | `numpy` 2.4.2  
**Target Artifacts Validated:**  
- Production Model: [`models/random_forest_final.joblib`](file:///d:/incomelens-ai/models/random_forest_final.joblib)  
- SHAP Values Dataset: [`reports/shap_values.csv`](file:///d:/incomelens-ai/reports/shap_values.csv)  
- Global Importance Visualization: [`reports/figures/shap_summary.png`](file:///d:/incomelens-ai/reports/figures/shap_summary.png)  
- Explainability Implementation: [`src/incomelens/explainability.py`](file:///d:/incomelens-ai/src/incomelens/explainability.py)

---

## 1. Executive Summary

This audit validates the mathematical rigor, exactness, and consistency of the SHAP (SHapley Additive exPlanations) explanations generated for the `IncomeLens-AI` production model.

Key conclusions:
1. **Explainer Architecture:** The project uses `shap.TreeExplainer` on the fitted `RandomForestClassifier` (`n_estimators=150`, `min_samples_leaf=2`).
2. **Explainer Output Space:** The explainer operates with `model_output="raw"`. For scikit-learn's `RandomForestClassifier`, the raw tree leaf predictions are class frequency proportions, and the forest aggregates them via a standard arithmetic mean without non-linear link functions (unlike gradient boosting log-odds). Consequently, **the raw output space is identical to class probability space**.
3. **Target Class Explained:** The explainer targets binary classification with `classes_ = ['<=50K', '>50K']`. The primary positive class (`>50K`, class index `1`) has a base expected value $\mathbb{E}[P] = 0.24087789$ (matching the positive class prevalence in the training set).
4. **Exact Mathematical Additivity Confirmed:** Across both a targeted exploratory sample (5 rows) and the full 1,000-row explanation set in [`reports/shap_values.csv`](file:///d:/incomelens-ai/reports/shap_values.csv):
   $$\mathbb{E}[P(y=\text{'>50K'})] + \sum_{j=1}^{105} \phi_j^{(1)}(x_i) = P(y=\text{'>50K'} \mid x_i)$$
   The maximum absolute discrepancy between the reconstructed probability and `model.predict_proba(X)[:, 1]` is **$5.99 \times 10^{-8}$**, with a mean absolute error of **$9.29 \times 10^{-10}$** across all 1,000 instances. There are zero instances exceeding $1 \times 10^{-7}$. The discrepancies are purely floating-point arithmetic rounding.
5. **Complementary Inverse Property:** Class 0 (`<=50K`) and Class 1 (`>50K`) SHAP values sum to zero feature-by-feature ($\phi_j^{(0)} + \phi_j^{(1)} = 0$) within machine epsilon ($1.41 \times 10^{-10}$ max error).
6. **Artifact Integrity:** [`reports/shap_values.csv`](file:///d:/incomelens-ai/reports/shap_values.csv) and [`reports/figures/shap_summary.png`](file:///d:/incomelens-ai/reports/figures/shap_summary.png) are strictly preserved as valid, reproducible artifacts.

---

## 2. SHAP Version, Explainer Configuration, and Output Space

### 2.1 Installed SHAP Version & Explainer Type
- **Package Version:** `shap==0.54.1`
- **Explainer Class:** `shap.TreeExplainer`
- **Underlying Estimator:** `sklearn.ensemble.RandomForestClassifier` (`min_samples_leaf=2`, `n_estimators=150`, `random_state=42`, `n_jobs=-1`)
- **Pipeline Context:** The saved artifact [`models/random_forest_final.joblib`](file:///d:/incomelens-ai/models/random_forest_final.joblib) is a scikit-learn `Pipeline` containing `ColumnTransformer` (scaling 6 numeric features + one-hot encoding 8 categorical features into 105 total features) followed by the `RandomForestClassifier`. Explanations are computed directly on the dense numeric output of the fitted preprocessor.

### 2.2 Output Space Mathematical Derivation
Unlike gradient-boosted decision trees (e.g., XGBoost, LightGBM, CatBoost) whose trees output unbounded margins in log-odds space ($\log(p / (1-p))$), scikit-learn's `RandomForestClassifier` computes leaf predictions as:
$$p_{t, k}(x) = \frac{n_{t, k}}{\sum_{c} n_{t, c}}$$
where $n_{t, k}$ is the number of training samples of class $k$ in the leaf node of tree $t$.

The forest prediction is the unweighted arithmetic mean of individual tree probabilities:
$$P(Y = k \mid x) = \frac{1}{T} \sum_{t=1}^{T} p_{t, k}(x)$$

Because the forest does not apply an external logistic or softmax link function:
$$\text{model\_output} = \text{"raw"} \equiv \text{predict\_proba}$$

`TreeExplainer` evaluates path contributions along each tree branch directly on these leaf probabilities. Therefore, each feature attribution $\phi_j^{(k)}$ represents the **exact additive contribution to the predicted probability** of class $k$.

---

## 3. Explainer Base Value and Class Label Alignment

### 3.1 Class Ordering
Inspection of `model.named_steps["classifier"].classes_`:
```python
['<=50K', '>50K']
```
- **Index 0:** `<=50K` (Low Income / Negative Class)
- **Index 1:** `>50K` (High Income / Positive Class)

### 3.2 Expected Values (Base Values)
The TreeExplainer computes background expected values based on the tree leaf weights:
- $\mathbb{E}[P(y=\text{'<=50K'})] = 0.7591221110286403$ (75.91%)
- $\mathbb{E}[P(y=\text{'>50K'})] = 0.24087788897135975$ (24.09%)
- **Base Value Sum:** $0.7591221110286403 + 0.24087788897135975 = 1.0000000000000000$

The base value for class 1 aligns with the training data class prevalence ($24.09\%$ high-income instances).

---

## 4. Additivity Validation Results

### 4.1 Reconstructed Probability Formula
For any row $x_i \in \mathbb{R}^{105}$:
$$\hat{P}_1(x_i) = \phi_0^{(1)} + \sum_{j=1}^{105} \phi_{i, j}^{(1)}$$
where:
- $\phi_0^{(1)} = \text{explainer.expected\_value}[1] = 0.24087789$
- $\phi_{i, j}^{(1)}$ is the SHAP value for feature $j$ on sample $i$ (stored in `reports/shap_values.csv`)

The error metric evaluated against model prediction $P(Y = \text{'>50K'} \mid x_i)$ is:
$$\Delta_i = \left| \hat{P}_1(x_i) - P(Y = \text{'>50K'} \mid x_i) \right|$$

### 4.2 Exploratory 5-Row Diagnostic
| Row Index | Model $P(>50\text{K} \mid x)$ | SHAP Base Value $\phi_0$ | $\sum \phi_j^{(1)}$ | Reconstructed $\hat{P}$ | Absolute Error $\Delta_i$ |
| :---: | :---: | :---: | :---: | :---: | :---: |
| 0 | 0.00000000 | 0.24087789 | -0.24087789 | 0.00000000 | $8.63 \times 10^{-14}$ |
| 1 | 0.04000000 | 0.24087789 | -0.20087789 | 0.04000000 | $6.94 \times 10^{-14}$ |
| 2 | 0.65333333 | 0.24087789 | +0.41245544 | 0.65333333 | $1.11 \times 10^{-13}$ |
| 3 | 0.00666667 | 0.24087789 | -0.23421122 | 0.00666667 | $9.63 \times 10^{-10}$ |
| 4 | 0.12666667 | 0.24087789 | -0.11421122 | 0.12666667 | $5.55 \times 10^{-14}$ |

### 4.3 Full 1,000-Row Sample Diagnostic ([`reports/shap_values.csv`](file:///d:/incomelens-ai/reports/shap_values.csv))
Statistical metrics across all $N = 1,000$ explained rows:

| Metric | Empirical Value | Acceptance Threshold | Status |
| :--- | :--- | :--- | :--- |
| **Max Absolute Error** ($\max \Delta_i$) | $5.9880 \times 10^{-8}$ | $< 1.0 \times 10^{-5}$ | PASS |
| **99th Percentile Error** | $1.8859 \times 10^{-8}$ | $< 1.0 \times 10^{-5}$ | PASS |
| **Mean Absolute Error** ($\text{MAE}$) | $9.2919 \times 10^{-10}$ | $< 1.0 \times 10^{-6}$ | PASS |
| **Median Absolute Error** | $1.0813 \times 10^{-10}$ | $< 1.0 \times 10^{-6}$ | PASS |
| **Min Absolute Error** | $8.6309 \times 10^{-14}$ | — | PASS |
| **Rows with $\Delta_i > 10^{-7}$** | 0 / 1,000 (0.00%) | 0 | PASS |
| **Rows with $\Delta_i > 10^{-5}$** | 0 / 1,000 (0.00%) | 0 | PASS |

### 4.4 Binary Complementarity Verification
Because the model outputs binary probabilities satisfying $P_0 + P_1 = 1$:
$$\sum_{j=1}^{105} \left( \phi_{i, j}^{(0)} + \phi_{i, j}^{(1)} \right) = 0$$
$$\max_{i, j} \left| \phi_{i, j}^{(0)} + \phi_{i, j}^{(1)} \right| = 1.4127 \times 10^{-10}$$

Both classes reflect exact additive symmetry.

---

## 5. Artifact Inspection & Feature Importance

### 5.1 Artifact Dimensionality and File Checks
- **CSV Path:** [`reports/shap_values.csv`](file:///d:/incomelens-ai/reports/shap_values.csv)
  - Dimensions: 1,000 rows $\times$ 105 feature columns.
  - Column Names: Identical match to `preprocessor.get_feature_names_out()` in sequence and naming.
  - Missing Values: 0 NaN, 0 Inf.
- **Plot Path:** [`reports/figures/shap_summary.png`](file:///d:/incomelens-ai/reports/figures/shap_summary.png)
  - Format: PNG, 372 KB, 300 DPI.
  - Type: SHAP Beeswarm Summary Plot displaying top 20 features with feature value color scale (low to high).

### 5.2 Top 10 Drivers of Income Prediction (>50K)
Ranked by global mean absolute SHAP value ($\frac{1}{N}\sum |\phi_{i, j}^{(1)}|$):

| Rank | Feature Name | Mean(\|SHAP\|) | Impact Direction |
| :---: | :--- | :---: | :--- |
| 1 | `categorical__marital-status_Married-civ-spouse` | 0.056853 | High value $\rightarrow$ Strongly increases $P(>50\text{K})$ |
| 2 | `categorical__relationship_Husband` | 0.045357 | High value $\rightarrow$ Increases $P(>50\text{K})$ |
| 3 | `numeric__education-num` | 0.043792 | Higher education years $\rightarrow$ Strongly increases $P(>50\text{K})$ |
| 4 | `numeric__capital-gain` | 0.042404 | Positive capital gain $\rightarrow$ Strongly increases $P(>50\text{K})$ |
| 5 | `numeric__age` | 0.035906 | Older age (up to retirement) $\rightarrow$ Increases $P(>50\text{K})$ |
| 6 | `numeric__hours-per-week` | 0.028160 | Higher work hours $\rightarrow$ Increases $P(>50\text{K})$ |
| 7 | `categorical__marital-status_Never-married` | 0.024533 | Never-married status $\rightarrow$ Decreases $P(>50\text{K})$ |
| 8 | `categorical__occupation_Exec-managerial` | 0.018935 | Executive / managerial role $\rightarrow$ Increases $P(>50\text{K})$ |
| 9 | `numeric__capital-loss` | 0.013194 | Higher capital loss $\rightarrow$ Increases $P(>50\text{K})$ |
| 10 | `categorical__education_Bachelors` | 0.011755 | Bachelor's degree $\rightarrow$ Increases $P(>50\text{K})$ |

---

## 6. Test Suite and Regression Verification

The test suite in [`tests/test_explainability.py`](file:///d:/incomelens-ai/tests/test_explainability.py) has been expanded with explicit mathematical additivity tests:
- `test_to_dense_numeric_from_sparse`: Confirms dense array conversion and float64 type casting.
- `test_to_dense_numeric_rejects_object_dtype_with_invalid_cast`: Prevents regression of `UFuncInputCastingError`.
- `test_to_dense_numeric_rejects_nan_and_inf`: Guard against unhandled imputations.
- `test_to_dense_numeric_rejects_non_2d`: Guard against malformed input arrays.
- `test_shap_values_computation_regression`: Verifies SHAP execution on production model without throwing ufunc errors.
- `test_saved_shap_artifacts_integrity`: Confirms shape `(1000, 105)` and file existence.
- `test_shap_mathematical_additivity`: Programmatically tests:
  1. Base value plus SHAP sum matches `predict_proba[:, 1]` within `1e-5`.
  2. Base value plus SHAP sum matches `predict_proba[:, 0]` within `1e-5`.
  3. Binary complementarity: `shap_0 + shap_1 == 0` within `1e-5`.

All 7 tests in `test_explainability.py` and all 41 project tests pass cleanly.
