# IncomeLens-AI Model Card

**Model Name:** IncomeLens-AI Production Classifier  
**Model Version:** 1.0.0 (Grouped Split Baseline Selection)  
**Model Date:** October 2026  
**Model Type:** Scikit-Learn Pipeline (`ColumnTransformer` + `RandomForestClassifier`)  
**Serialized Artifact:** [`models/random_forest_final.joblib`](file:///d:/incomelens-ai/models/random_forest_final.joblib)  
**Artifact SHA-256:** `f164a61183466b25186377e8946f4cb74b2df924f69e50224f5d8de72858a285`  
**License:** Open Educational / Research Benchmark  
**Production Readiness Status:** **NOT PRODUCTION-READY** (Restricted strictly to local research, benchmarking, and educational demonstration)

---

## 1. Model Overview & Architecture

`IncomeLens-AI` is a supervised machine learning classification pipeline engineered to predict whether an individual's reported annual gross income exceeded \$50,000 (`>50K`) versus less than or equal to \$50,000 (`<=50K`) based on demographic, household, and labor market attributes recorded in the 1994 United States Census database.

### 1.1 Technical Specifications
- **Pipeline Structure:** Single cohesive Scikit-Learn [`Pipeline`](file:///d:/incomelens-ai/models/random_forest_final.joblib) encapsulating both preprocessing and classification stages.
- **Preprocessor (`ColumnTransformer`):**
  - **Numeric Sub-pipeline (6 features):** `age`, `fnlwgt`, `education-num`, `capital-gain`, `capital-loss`, `hours-per-week` processed via `SimpleImputer(strategy='median')` followed by `StandardScaler()`.
  - **Categorical Sub-pipeline (8 features):** `workclass`, `education`, `marital-status`, `occupation`, `relationship`, `race`, `sex`, `native-country` processed via `SimpleImputer(strategy='most_frequent')` followed by `OneHotEncoder(handle_unknown='ignore', sparse_output=False)`.
  - **Transformed Feature Dimensionality:** Exactly **105 dense numeric features**.
- **Estimator (`RandomForestClassifier`):**
  - **Number of Trees:** $n\_estimators = 150$.
  - **Regularization:** $min\_samples\_leaf = 2$.
  - **Random Seed:** $random\_state = 42$ (fixed for deterministic reproducibility).
  - **Parallelization:** $n\_jobs = -1$.
  - **Decision Boundary:** Standard uniform threshold $\tau = 0.50$ ($P(\hat{Y}=1) \ge 0.50$).

---

## 2. Intended Use & Scope Boundaries

### 2.1 Intended & Supported Uses
- **Academic & Technical Benchmarking:** Evaluating tabular classification algorithms, grouped validation schemes, and hyperparameter stability.
- **Interpretability & Explainability Research:** Conducting comparative studies between global impurity-based heuristics (Gini MDI) and exact probability-space additive attributions (TreeExplainer SHAP).
- **Algorithmic Fairness Research:** Auditing demographic selection rate disparities, measuring equal opportunity gaps, and investigating proxy feature reconstruction under observational bias.
- **Software Engineering Demonstrations:** Demonstrating robust ML engineering practices, including cryptographic artifact provenance, schema validation contracts, and automated test suites.

### 2.2 Prohibited Operational Uses & Unsupported Interpretations
> [!CAUTION]
> The model is **strictly prohibited and unsuitable** for any operational deployment or decision-making system affecting human subjects.

#### Unsupported Interpretations:
- **Not a Measure of Current Earning Power:** The model predicts 1994 US dollar thresholds (\$50,000 in 1994 corresponds to over \$105,000 in 2026 inflation-adjusted purchasing power). It cannot predict contemporary wage levels.
- **Not a Continuous Earnings Estimator:** The model provides binary classification likelihoods, not granular salary predictions, liquid cash flow estimates, or household wealth calculations.
- **Not Causal:** Feature attributions (e.g., `marital-status_Married-civ-spouse` or `education-num`) represent observational associations present in 1994 census sampling, not causal determinants. Intervening on an individual's marital status does not causally increase earning potential.
- **Not an Assessment of Individual Competence:** The model assigns statistical likelihoods based on aggregate demographic categories; using it to judge individual character, intelligence, diligence, or creditworthiness is fundamentally invalid.

#### Strictly Prohibited Deployment Domains:
1. Automated credit underwriting, loan origination, interest rate setting, or insurance risk assessment.
2. Employment applicant screening, hiring, promotion, compensation, or termination gating.
3. Housing allocation, tenant screening, lease qualification, or mortgage eligibility.
4. Social welfare, healthcare benefit, or public assistance determinations.
5. Criminal justice risk assessment, bail determination, or sentencing support.
6. Solely automated decision-making without independent, authenticated human review.

---

## 3. Dataset Provenance & Data Limitations

### 3.1 Provenance & Partitions
- **Source:** 1994 United States Census Bureau Current Population Survey (CPS) Adult database, originally curated by Ronny Kohavi and Barry Becker.
- **Total Dataset Size:** 48,842 survey records.
- **Target Distribution:** Binary classification with substantial class imbalance:
  - Class 0 (`<=50K`): $75.91\%$ prevalence in development splits.
  - Class 1 (`>50K`): $24.09\%$ prevalence in development splits.
- **Grouped Partitions (Leak-Free Isolation):**
  - **Train Split ([`data/processed/train.csv`](file:///d:/incomelens-ai/data/processed/train.csv)):** 29,313 records ($60.0\%$), used exclusively for fitting preprocessing transformers and model estimators.
  - **Validation Split ([`data/processed/validation.csv`](file:///d:/incomelens-ai/data/processed/validation.csv)):** 9,758 records ($20.0\%$), reserved for model selection and threshold tuning.
  - **Test Split ([`data/processed/test.csv`](file:///d:/incomelens-ai/data/processed/test.csv)):** 9,771 records ($20.0\%$), held out strictly for frozen evaluation.
  - *Partitioning Strategy:* Partitioned via grouped splitting on `fnlwgt` (final sampling weight) to ensure identical household and demographic clusters do not leak across training, validation, and test sets.

### 3.2 Known Data Limitations & Biases
1. **Historical Labor Bias:** The 1994 survey reflects historical labor market barriers, lower female workforce participation rates in executive roles, and significant racial wage disparities prevalent in the early 1990s.
2. **Geographical Limitation:** Exclusively representative of the 1994 United States resident population. Inapplicable to non-US labor markets.
3. **Sampling Weight Artifact (`fnlwgt`):** The administrative survey weight `fnlwgt` reflects population socio-demographic balancing controls. It has zero economic or predictive utility for personal earning capacity, yet its high cardinality introduces noise in tree splits.
4. **Duplicate & Conflicting Records:** The raw data contains 52 exact duplicate rows, 111 repeated feature combinations, and 5 conflicting records where identical demographic profiles are assigned contradictory income labels (`<=50K` vs `>50K`).
5. **Top-Coding:** Capital gains and losses are artificially capped at \$99,999 in the raw Census data, truncating the true distribution of high-net-worth individuals.

---

## 4. Evaluation Methodology & Verified Performance

The production model was evaluated against the frozen held-out test split ([`data/processed/test.csv`](file:///d:/incomelens-ai/data/processed/test.csv), $N = 9,771$) without refitting or test-set tuning.

### 4.1 Verified Global Metrics
All reported metrics below were independently validated from first principles with zero discrepancy against scikit-learn standard evaluation functions (documented in [`reports/metric_consistency_audit.md`](file:///d:/incomelens-ai/reports/metric_consistency_audit.md)):

| Metric Name | Verified Test Score | 95% Confidence Interval | Benchmark Comparison (vs. Dummy) | Mathematical Formulation |
| :--- | :---: | :---: | :---: | :--- |
| **Accuracy** | **0.8683** | [0.8615, 0.8749] | +10.95% over Most Frequent (0.7588) | $\frac{TP + TN}{N} = \frac{1,404 + 7,080}{9,771} = \frac{8,484}{9,771}$ |
| **Balanced Accuracy** | **0.7809** | [0.7712, 0.7904] | +28.09% over Dummy (0.5000) | $\frac{1}{2}(\text{TPR} + \text{TNR}) = \frac{0.6174 + 0.9444}{2}$ |
| **High-Income Precision** | **0.7710** | [0.7512, 0.7898] | +53.00% over Base Rate (0.2410) | $\frac{TP}{TP + FP} = \frac{1,404}{1,404 + 417} = \frac{1,404}{1,821}$ |
| **High-Income Recall** | **0.6174** | [0.5971, 0.6373] | +61.74% over Constant Low (0.0000) | $\frac{TP}{TP + FN} = \frac{1,404}{1,404 + 870} = \frac{1,404}{2,274}$ |
| **High-Income F1-Score** | **0.6857** | [0.6702, 0.7008] | +68.57% over Zero Baseline | $\frac{2 \cdot TP}{2 \cdot TP + FP + FN} = \frac{2,808}{4,095}$ |
| **ROC-AUC** | **0.9181** | [0.9112, 0.9248] | +41.81% over Random Guess (0.5000) | $\int_0^1 \text{TPR}(FPR^{-1}(t)) \, dt$ |

> [!NOTE]
> The classification report text summary displays accuracy rounded to two decimals as **`0.87`** (`sklearn.metrics.classification_report(digits=2)`). When rendered with 4 decimal digits, it confirms **`0.8683`** with zero numerical divergence.

### 4.2 Class-Level Performance & Confusion Matrix

#### Per-Class Breakdown ($N = 9,771$)
| Class Label | Support ($N$) | Prevalence | Precision | Recall (TPR / TNR) | F1-Score | Specificity |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`<=50K` (Negative)** | 7,497 | 76.73% | 0.8906 (7,080 / 7,950) | 0.9444 (7,080 / 7,497) | 0.9167 | 0.6174 |
| **`>50K` (Positive)** | 2,274 | 23.27% | 0.7710 (1,404 / 1,821) | 0.6174 (1,404 / 2,274) | 0.6857 | 0.9444 |

#### Standardized Confusion Matrix ($Rows = Actual, Columns = Predicted$)
$$\begin{array}{c|cc|c}
& \mathbf{Predicted \le 50K} & \mathbf{Predicted > 50K} & \mathbf{Total} \\
\hline
\mathbf{Actual \le 50K} & 7,080 \text{ (TN)} & 417 \text{ (FP)} & 7,497 \\
\mathbf{Actual > 50K} & 870 \text{ (FN)} & 1,404 \text{ (TP)} & 2,274 \\
\hline
\mathbf{Total} & 7,950 & 1,821 & 9,771
\end{array}$$

---

## 5. Explainability Methodology & Interpretability Findings

The pipeline incorporates dual global explainability mechanisms and per-instance local attribution:

### 5.1 Gini Impurity (MDI) Methodology
- **Extraction:** Feature importances are computed directly from the forest's variance reduction across all 150 trees via [`scripts/10_feature_importance.py`](file:///d:/incomelens-ai/scripts/10_feature_importance.py).
- **Known Flaw (Cardinality Selection Bias):** MDI is known to overvalue numerical attributes with continuous, fine-grained splits. In our audit ([`reports/feature_importance_methodology.md`](file:///d:/incomelens-ai/reports/feature_importance_methodology.md)), `fnlwgt` (pure survey sampling noise) emerged as the **#7 most important feature** ($6.23\%$ of total split importance), exposing the danger of relying naively on MDI for feature selection.

### 5.2 TreeExplainer SHAP Methodology
- **Computation:** Exact probability-space Shapley values are calculated via `shap.TreeExplainer` on a representative 1,000-instance training split using [`src/incomelens/explainability.py`](file:///d:/incomelens-ai/src/incomelens/explainability.py).
- **Exact Additivity Proof:** For Random Forest classifiers, TreeExplainer computes attributions directly in class probability units. The sum of SHAP values plus the base expected value ($E[f(x)] = 0.2372$) reconstructs the model's predicted probability of `>50K` within machine precision ($\max |\Delta| < 6 \times 10^{-8}$). See [`reports/shap_validation.md`](file:///d:/incomelens-ai/reports/shap_validation.md).
- **Top Predictive Drivers of High Income:**
  1. `capital-gain`: Dominant positive driver when non-zero.
  2. `education-num`: Strong monotonic positive relationship with high income.
  3. `age`: Non-linear quadratic effect peaking between ages 45 and 55.
  4. `relationship_Husband` / `marital-status_Married-civ-spouse`: Substantial positive attribution reflecting 1994 joint household tax filing structures.
  5. `hours-per-week`: Positive attribution above 40 hours/week; strong negative penalty below 35 hours/week.

---

## 6. Algorithmic Fairness Audit & Missing Analyses

Demographic disparity analyses were conducted on the held-out test split ($N = 9,771$) across `sex` and `race` using [`src/incomelens/fairness.py`](file:///d:/incomelens-ai/src/incomelens/fairness.py). Full documentation is provided in [`reports/fairness_report.md`](file:///d:/incomelens-ai/reports/fairness_report.md).

### 6.1 Verified Subgroup Disparity Table
| Sensitive Attribute | Subgroup | Support ($N$) | Base Rate $P(Y=1)$ | Selection Rate $P(\hat{Y}=1)$ | Disparate Impact Ratio | Group Recall (TPR) | Equal Opportunity Difference | 80% Rule Compliance |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Sex** | **Male** *(Reference)* | 6,535 | 29.44% | **24.25%** | **1.000** | **63.36%** | **0.0000** | Reference |
| | **Female** | 3,236 | 10.82% | **7.29%** | **0.301** | **52.86%** | **-0.1050** | **VIOLATION (< 0.80)** |
| **Race** | **White** *(Reference)* | 8,312 | 24.70% | **19.77%** | **1.000** | **61.96%** | **0.0000** | Reference |
| | **Black** | 942 | 12.00% | **9.02%** | **0.456** | **54.87%** | **-0.0709** | **VIOLATION (< 0.80)** |
| | **Asian-Pac-Islander** | 326 | 25.46% | **23.62%** | **1.195** | **69.88%** | **+0.0792** | Compliant |
| | **Amer-Indian-Eskimo** | 107 | 12.15% | **7.48%** | **0.378** | **38.46%** | **-0.2350** | **VIOLATION (< 0.80)** |

### 6.2 Key Disparity Findings
1. **Severe Gender Selection Disparity:** The disparate impact ratio for females is **0.301**, severely below the EEOC four-fifths (0.80) guideline. Females are selected for high income at less than one-third the rate of males.
2. **Opportunity Deficit (Recall Gap):** High-earning females suffer a $10.50$ percentage point recall deficit ($52.86\%$ vs. $63.36\%$), meaning high-earning women are substantially more likely to be misclassified as low-income than high-earning men.
3. **Proxy Variable Leakage ("Fairness Through Blindness" Fallacy):** Removing explicit protected attributes (`sex` or `race`) does not eradicate disparate impact. Highly correlated proxies (`relationship_Husband`, `marital-status`, and gender-stratified `occupation` categories) enable the forest to implicitly reconstruct protected classes.

### 6.3 Missing Analyses & Audit Gaps (Unverified Evaluations)
The following analyses have **not** been performed and are explicitly declared as missing:
- **Intersectional Fairness Audits:** Combined intersectional cohorts (e.g. Black female, Hispanic female, elderly minority workers) were not evaluated due to severe sample size shrinkage ($N < 50$) in sub-cohorts leading to statistical instability.
- **Counterfactual Fairness:** Causal counterfactual interventions (e.g. flipping `sex` while holding all structural proxies fixed) were not evaluated.
- **Post-Processing Fairness Interventions:** The final model was **not** adjusted with group-specific classification thresholds or rejection-option classification to enforce equalized odds or demographic parity; it operates at a uniform threshold of $\tau = 0.50$.
- **External Dataset Validation:** The model has **never been validated** on any contemporary dataset (e.g. American Community Survey 2020+, modern CPS, or international census records).

---

## 7. Known Failure Modes & Deployment Limitations

### 7.1 Specific Failure Modes Identified
1. **High-Income Recall Ceiling (38.26% Miss Rate):** The model produces 870 false negatives out of 2,274 true high earners. For individuals near the \$50,000 threshold or those without substantial capital gains, the model default is strongly conservative (`<=50K`).
2. **Boundary Profile Instability:** Survey profiles receiving probabilities near the decision boundary ($0.40 \le P \le 0.60$) exhibit volatility under minor shifts in continuous inputs (such as $\pm 2$ hours worked or small fluctuations in sampling weights).
3. **Imputation Artifacts for Infrequent Categories:** Rare categorical levels unseen during training are imputed to mode and assigned all-zero one-hot vectors via `handle_unknown='ignore'`, causing them to default to baseline class expectations.
4. **Proxy Feature Encoding of Marital Status:** The model heavily penalizes unmarried and single-parent households due to joint filing proxies embedded in the 1994 survey format.

### 7.2 Local Deployment & Demonstration Boundary
As verified in [`reports/deployment_readiness.md`](file:///d:/incomelens-ai/reports/deployment_readiness.md):
- **Local Demonstration Status:** **READY (10/10 PASS)** for local interactive demonstration (`streamlit run app/Home.py`) and offline automated testing.
- **Production Status:** **UNREADY FOR REAL-WORLD APPLICATION**.
- **External Guardrail:** The model must remain confined to local sandbox execution. No cloud endpoints, external databases, or third-party web services have been or should be provisioned.

---

## 8. Summary Checklist of Verified Facts

| Criterion | Verified Status | Evidence Location |
| :--- | :---: | :--- |
| **Model Version & Digest** | Verified (SHA-256 `f164a611...`) | [`reports/artifact_manifest.json`](file:///d:/incomelens-ai/reports/artifact_manifest.json) |
| **Data Partition Sizes** | Verified (Train: 29,313; Val: 9,758; Test: 9,771) | [`reports/split_validation.md`](file:///d:/incomelens-ai/reports/split_validation.md) |
| **Accuracy Score** | Verified (0.8683 on Test) | [`reports/metric_consistency_audit.md`](file:///d:/incomelens-ai/reports/metric_consistency_audit.md) |
| **Recall / False Negative Rate** | Verified (0.6174 Recall, 38.26% FN rate) | [`reports/metric_consistency_audit.md`](file:///d:/incomelens-ai/reports/metric_consistency_audit.md) |
| **Gender Disparate Impact** | Verified (0.301 DIR, < 0.80 violation) | [`reports/fairness_metrics.csv`](file:///d:/incomelens-ai/reports/fairness_metrics.csv) |
| **Exact SHAP Additivity** | Verified ($\max \Delta < 6 \times 10^{-8}$) | [`reports/shap_validation.md`](file:///d:/incomelens-ai/reports/shap_validation.md) |
| **Automated Test Suite** | Verified (146/146 PASS in 16.15s) | [`tests/`](file:///d:/incomelens-ai/tests/) |
