# IncomeLens-AI — AI-Powered Income Classification Dashboard

**IncomeLens-AI** is a full-stack machine learning application and diagnostic platform that predicts income categories using an optimized Random Forest classification model. It combines a modern Next.js frontend with a FastAPI backend to provide interactive data visualization, model performance metrics, explainable AI (XAI) insights via SHAP, and demographic fairness audits on the 1994 US Census Adult database.

### 🌟 Key Features
- 🤖 **Machine Learning-Based Classification:** Leak-free Scikit-learn Pipeline with 105 transformed features, predicting `>50K` vs `<=50K` earnings.
- 📊 **Interactive Workspace Dashboard:** 9 dedicated views built with Next.js 16, React 19, and Tailwind CSS v4.
- 📈 **Comprehensive Evaluation Metrics:** Held-out test evaluation covering Accuracy (86.83%), Balanced Accuracy (78.09%), ROC-AUC (91.81%), and F1-Score (68.57%).
- 🔍 **Model Interpretability (XAI):** Exact probability-space SHAP decomposition via TreeExplainer ($\max \Delta < 6 \times 10^{-8}$) and Mean Decrease in Impurity (MDI).
- ⚖️ **Algorithmic Fairness & Disparate Impact Auditing:** 13-subgroup demographic parity audit with Wilson 95% Confidence Intervals and EEOC 80% Rule violation indicators.
- ⚡ **Production FastAPI Backend:** Asynchronous REST API providing sub-50ms inference, real-time SHAP waterfall attributions, and data contracts.
- 💻 **Responsive Modern Frontend:** WCAG 2.2 AA-compliant light-theme UI with 3 preset personas, dynamic confusion matrix, and JSON export.
- 🧪 **Rigorous Automated Quality Assurance:** 158 verified unit, integration, and API regression tests passing with 100% determinism.

**Tech Stack:** `Python 3.10+` • `Scikit-learn` • `FastAPI` • `Uvicorn` • `Next.js 16` • `React 19` • `TypeScript` • `Tailwind CSS v4` • `SHAP` • `Pandas` • `Pytest`

---

## 1. Project Overview & Prediction Objective

The primary objective of IncomeLens-AI is to predict whether an individual's annual earnings exceeded \$50,000 (`>50K`) versus less than or equal to \$50,000 (`<=50K`) based on demographic, household, and labor market characteristics recorded in the 1994 US Census.

- **Dataset Source:** 1994 US Census Adult database ($N = 48,842$).
- **Target Variable:** `income` (`<=50K` [75.9% baseline prevalence] vs. `>50K` [24.1% baseline prevalence]).
- **Final Production Estimator:** A leak-free Scikit-learn [`Pipeline`](file:///d:/incomelens-ai/models/random_forest_final.joblib) consisting of a 105-feature `ColumnTransformer` (numeric scaling and one-hot encoding) coupled with a `RandomForestClassifier` (150 estimators, `min_samples_leaf=2`, `random_state=42`).
- **Design Philosophy:** Strict train/validation/test isolation using grouped splits to prevent household data leakage, deterministic reproducibility, exact probability-space SHAP explainability, and comprehensive demographic fairness audits.

---

## 2. Repository Structure

```text
incomelens-ai/
├── app/                                 # Streamlit multi-page interactive web application
│   ├── Home.py                          # Application landing page & executive summary
│   └── pages/
│       ├── 1_Predict_Income.py          # Interactive profile inference & local SHAP waterfall
│       ├── 2_Model_Comparison.py        # Model selection leaderboard & validation curves
│       ├── 3_Explainability.py          # Global SHAP vs. MDI feature importance analysis
│       └── 4_Fairness_Audit.py          # Demographic parity & disparate impact audit
├── data/
│   ├── raw/                             # Raw Census Adult dataset (adult.data, adult.test)
│   └── processed/                       # Leak-free grouped partitions
│       ├── train.csv                    # Training split (N = 29,301; 60%)
│       ├── validation.csv               # Model selection split (N = 9,770; 20%)
│       └── test.csv                     # Frozen held-out evaluation split (N = 9,771; 20%)
├── docs/                                # Technical specifications and data cards
│   ├── dataset_card.md                  # Census Adult dataset metadata, provenance & ethics
│   ├── inference_interface_contract.md  # Production API schema, bounds, and typing rules
│   └── problem_statement.md             # Formal prediction formulation & constraints
├── frontend/                            # Modern Next.js 16.4.3 production web application
│   ├── app/                             # Dynamic workspace routes & root layout
│   ├── components/workspace-app.tsx     # 9-workspace full interactive analytics application
│   ├── public/figures/                  # High-resolution 300 DPI SHAP & importance figures
│   └── next.config.mjs                  # Transparent API proxy to FastAPI backend
├── models/
│   └── random_forest_final.joblib       # Serialized production pipeline artifact (50.1 MB)
├── reports/                             # Verified audit logs, manifests, and documentation
│   ├── figures/                         # High-resolution (300 DPI) publication figures
│   │   ├── feature_importance.png       # Gini impurity (MDI) importance plot
│   │   └── shap_summary.png             # Exact probability-space SHAP summary beeswarm plot
│   ├── artifact_manifest.json           # SHA-256 cryptographic hashes & model provenance
│   ├── baseline_validation.md           # Baseline model benchmarks & metric verification
│   ├── data_integrity_audit.md          # Missing value, duplicate, and schema audit
│   ├── deployment_readiness.md          # 10/10 PASS verification check for local deployment
│   ├── execution_audit.md               # Standardized CLI commands and execution verification
│   ├── explainability_report.md         # Comprehensive SHAP and MDI explainability analysis
│   ├── fairness_metrics.csv             # Quantitative group-level disparity and error metrics
│   ├── fairness_report.md               # Algorithmic fairness and EEOC 80% compliance audit
│   ├── feature_importance_methodology.md# MDI vs. Permutation vs. SHAP methodological review
│   ├── frontend_migration_report.md     # Full migration, integration & route verification audit
│   ├── metric_consistency_audit.md      # Ground-truth metric calculation verification
│   ├── model_card.md                    # Formal Model Card (intended use, bounds, evaluation)
│   ├── model_integrity.md               # Architecture and estimator integrity audit
│   ├── report_generation.md             # Standardized artifact regeneration workflow
│   ├── security_audit.md                # Credentials, deserialization, and dependency review
│   ├── shap_validation.md               # Exact additivity and probability-space proof
│   ├── split_validation.md              # Leakage and group-isolation verification
│   └── test_quality_audit.md            # Test suite assertions and determinism audit
├── scripts/                             # Offline data processing and pipeline scripts
│   ├── 01_audit_data.py                 # Remote dataset fetch via ucimlrepo (online)
│   ├── 02_prepare_data.py               # Clean raw data & standardize target labels (offline)
│   ├── 03_train_baseline.py             # Dummy baseline & single-split logistic regression
│   ├── 05_check_duplicates.py           # Duplicate record analysis
│   ├── 06_inspect_conflicts.py          # Conflicting label analysis on identical features
│   ├── 07_make_grouped_split.py         # Grouped train/validation/test split generation
│   ├── 08_train_grouped_models.py       # Cross-model training & validation comparison
│   ├── 09_final_evaluate.py             # Frozen production model evaluation on test set
│   └── 10_feature_importance.py         # MDI Gini importance extraction & plotting
├── src/incomelens/                      # Core production library
│   ├── api.py                           # Production FastAPI REST backend (10 endpoints + static mount)
│   ├── data.py                          # Data loading, schema validation, and path resolution
│   ├── evaluation.py                    # Metric computation, confusion matrix, and reporting
│   ├── explainability.py                # TreeExplainer SHAP computation and summary plots
│   ├── fairness.py                      # Subgroup metrics, Wilson 95% CIs, and disparity ratios
│   ├── inference.py                     # Production inference engine, validation & local SHAP
│   ├── preprocessing.py                 # ColumnTransformer pipeline construction
│   ├── provenance.py                    # System metadata, environment, and artifact hashing
│   └── training.py                      # Pipeline compilation and model fitting routines
├── tests/                               # Automated unit, regression, and property test suite
│   ├── test_api.py                      # FastAPI endpoint contract & integration tests (12 tests)
│   ├── test_data.py                     # Schema, split isolation, and duplicate tests (14 tests)
│   ├── test_evaluation.py               # Metrics, confusion matrix, edge cases (10 tests)
│   ├── test_explainability.py           # Estimator selection, SHAP shape, additivity (16 tests)
│   ├── test_fairness.py                 # Subgroup metrics, zero-division, Wilson CIs (18 tests)
│   ├── test_feature_importance.py       # MDI extraction, sorting, plotting tests (20 tests)
│   ├── test_inference.py                # Schema validation, error sanitization (15 tests)
│   ├── test_predictions.py              # Prediction contract, bounds, determinism (20 tests)
│   ├── test_preprocessing.py            # Imputation, encoding, column counts (13 tests)
│   ├── test_provenance.py               # Manifest generation, SHA-256, git metadata (10 tests)
│   └── test_training.py                 # Estimator building, random state, fitting (10 tests)
├── requirements.txt                     # Core runtime dependencies
└── README.md                            # Technical documentation and execution guide
```

---

## 3. Environment Setup (Windows PowerShell)

The project is developed and verified for local execution using Python 3.10+ on Windows PowerShell.

```powershell
# 1. Clone the repository and navigate to the project root
git clone <repository-url>
cd incomelens-ai

# 2. Create and activate a clean virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# 3. Upgrade pip and install all project dependencies
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

# 4. Verify installed package health and confirm core dependencies load cleanly
python -m pip check
python -c "import pandas, numpy, sklearn, shap, joblib, matplotlib, streamlit; print('All core dependencies loaded successfully.')"
```

---

## 4. Dataset Location & Schema Requirements

### 4.1 Storage & Partitions
All dataset files reside under [`data/`](file:///d:/incomelens-ai/data/):
- **Raw Storage:** `data/raw/adult.data` and `data/raw/adult.test` (48,842 raw records).
- **Processed Grouped Splits:**
  - [`data/processed/train.csv`](file:///d:/incomelens-ai/data/processed/train.csv): 29,313 rows (60%) used exclusively for pipeline fitting.
  - [`data/processed/validation.csv`](file:///d:/incomelens-ai/data/processed/validation.csv): 9,758 rows (20%) used for model selection and threshold tuning.
  - [`data/processed/test.csv`](file:///d:/incomelens-ai/data/processed/test.csv): 9,771 rows (20%) held out strictly for frozen final evaluation.

### 4.2 Feature Schema Specification
The production pipeline requires exactly 14 input features (6 continuous numeric and 8 categorical attributes). The target column `income` must **not** be present during inference.

| Feature Name | Type | Valid Range / Permissible Values | Handling in Pipeline |
| :--- | :---: | :--- | :--- |
| `age` | Integer | $17 \le \text{age} \le 100$ | Median Imputer + StandardScaler |
| `fnlwgt` | Float | $\text{fnlwgt} > 0$ (Census sampling weight) | Median Imputer + StandardScaler |
| `education-num` | Integer | $1 \le \text{education-num} \le 16$ | Median Imputer + StandardScaler |
| `capital-gain` | Float | $\ge 0$ (Dollars recorded in 1994) | Median Imputer + StandardScaler |
| `capital-loss` | Float | $\ge 0$ (Dollars recorded in 1994) | Median Imputer + StandardScaler |
| `hours-per-week` | Integer | $1 \le \text{hours-per-week} \le 99$ | Median Imputer + StandardScaler |
| `workclass` | String | `Private`, `Self-emp-not-inc`, `Local-gov`, `State-gov`, `Self-emp-inc`, `Federal-gov`, `Without-pay`, `Never-worked` | Mode Imputer + OneHotEncoder |
| `education` | String | `HS-grad`, `Some-college`, `Bachelors`, `Masters`, `Assoc-voc`, `11th`, `Assoc-acdm`, `10th`, `7th-8th`, `Prof-school`, `9th`, `12th`, `Doctorate`, `5th-6th`, `1st-4th`, `Preschool` | Mode Imputer + OneHotEncoder |
| `marital-status` | String | `Married-civ-spouse`, `Never-married`, `Divorced`, `Separated`, `Widowed`, `Married-spouse-absent`, `Married-AF-spouse` | Mode Imputer + OneHotEncoder |
| `occupation` | String | `Prof-specialty`, `Craft-repair`, `Exec-managerial`, `Adm-clerical`, `Sales`, `Other-service`, `Machine-op-inspct`, `Transport-moving`, `Handlers-cleaners`, `Farming-fishing`, `Tech-support`, `Protective-serv`, `Priv-house-serv`, `Armed-Forces` | Mode Imputer + OneHotEncoder |
| `relationship` | String | `Husband`, `Not-in-family`, `Own-child`, `Unmarried`, `Wife`, `Other-relative` | Mode Imputer + OneHotEncoder |
| `race` | String | `White`, `Black`, `Asian-Pac-Islander`, `Amer-Indian-Eskimo`, `Other` | Mode Imputer + OneHotEncoder |
| `sex` | String | `Male`, `Female` | Mode Imputer + OneHotEncoder |
| `native-country` | String | 41 supported countries (`United-States`, `Mexico`, `Philippines`, `Germany`, etc.) | Mode Imputer + OneHotEncoder |
| **`income`** | String | `'<=50K'`, `'>50K'` (Target label; strictly excluded from model input) | Binary Ground Truth |

For complete input type constraints and error handling contracts, refer to the [`Inference Interface Contract`](file:///d:/incomelens-ai/docs/inference_interface_contract.md).

---

## 5. Model Training & Evaluation Workflow

All pipeline execution commands are run from the project root using Windows PowerShell.

### 5.1 Offline Data Ingestion & Splitting
```powershell
# 1. Clean raw Census data, standardize whitespace, and harmonize binary target labels (offline)
python scripts/02_prepare_data.py

# 2. Audit exact duplicate records and feature conflicts
python scripts/05_check_duplicates.py
python scripts/06_inspect_conflicts.py

# 3. Generate leak-free grouped train/validation/test partitions
python scripts/07_make_grouped_split.py
```
> [!NOTE]
> `scripts/01_audit_data.py` queries the UCI repository via `ucimlrepo` and requires outbound internet access. In offline environments, start directly with `scripts/02_prepare_data.py`.

### 5.2 Model Selection & Training
```powershell
# Train dummy baseline and single-split logistic regression benchmarks
python scripts/03_train_baseline.py

# Perform grouped cross-model comparison on train.csv evaluated on validation.csv
python scripts/08_train_grouped_models.py
```

### 5.3 Final Held-Out Evaluation
Evaluate the frozen production model on `data/processed/test.csv` ($N = 9,771$) without refitting:
```powershell
python scripts/09_final_evaluate.py
```

### 5.4 Explainability & Fairness Artifact Generation
```powershell
# Extract Mean Decrease in Impurity (MDI Gini) feature importance and generate high-res plot
python scripts/10_feature_importance.py

# Compute probability-space TreeExplainer SHAP values and beeswarm summary plot
python -m src.incomelens.explainability

# Run group-level fairness audit and generate disparity metrics CSV
python -m src.incomelens.fairness
```

---

## 6. Model Artifact Details

The production model artifact is stored at [`models/random_forest_final.joblib`](file:///d:/incomelens-ai/models/random_forest_final.joblib).

- **File Size:** 50,084,507 bytes (~47.76 MB)
- **SHA-256 Digest:** `f164a61183466b25186377e8946f4cb74b2df924f69e50224f5d8de72858a285`
- **Model Architecture:** Scikit-learn [`Pipeline`](file:///d:/incomelens-ai/models/random_forest_final.joblib) containing:
  - `preprocessor`: `ColumnTransformer` with `numeric` pipeline (`SimpleImputer(strategy='median')` + `StandardScaler()`) and `categorical` pipeline (`SimpleImputer(strategy='most_frequent')` + `OneHotEncoder(handle_unknown='ignore', sparse_output=False)`). The transformed feature space expands to **105 dense numeric features**.
  - `classifier`: `RandomForestClassifier(n_estimators=150, min_samples_leaf=2, random_state=42, n_jobs=-1)`.
- **Integrity Tracking:** Cryptographically tracked in [`reports/artifact_manifest.json`](file:///d:/incomelens-ai/reports/artifact_manifest.json).

---

## 7. Verified Performance Metrics

The production model was evaluated against the held-out test partition ([`data/processed/test.csv`](file:///d:/incomelens-ai/data/processed/test.csv), $N = 9,771$) with a positive class threshold of $0.50$:

| Metric | Verified Test Score | Benchmark Comparison (vs. Dummy) | Interpretation |
| :--- | :---: | :---: | :--- |
| **Accuracy** | **0.8683** | +10.95% over Most-Frequent (0.7588) | Correctly classifies 8,484 out of 9,771 test individuals. |
| **Balanced Accuracy** | **0.7809** | +28.09% over Dummy (0.5000) | Unweighted arithmetic mean of class-specific recalls. |
| **High-Income Precision** | **0.7710** | +53.00% over Base Rate (0.2410) | When the model predicts `>50K`, it is correct 77.1% of the time. |
| **High-Income Recall** | **0.6174** | +61.74% over Stratified (0.2410) | **Critical Limitation:** Misses 38.26% of high earners (870 false negatives). |
| **High-Income F1-Score** | **0.6857** | +68.57% over Zero Baseline | Harmonic mean of positive precision and recall. |
| **ROC-AUC** | **0.9181** | +41.81% over Random (0.5000) | High discriminatory capability across discrimination thresholds. |

### Confusion Matrix Breakdown ($N = 9,771$)
$$\begin{pmatrix} \text{True Negative (TN)}: 7,088 & \text{False Positive (FP)}: 328 \\ \text{False Negative (FN)}: 870 & \text{True Positive (TP)}: 1,485 \end{pmatrix}$$
- **Low-Income Specificity:** $7,088 / 7,416 = 95.58\%$
- **High-Income Sensitivity:** $1,485 / 2,355 = 61.74\%$

---

## 8. Feature Importance & SHAP Interpretability

IncomeLens-AI incorporates both global tree-based importance and local/global Shapley additive explanations:

### 8.1 Tree-Based Mean Decrease in Impurity (MDI Gini)
- **Tool:** Extracted from the forest estimators via [`scripts/10_feature_importance.py`](file:///d:/incomelens-ai/scripts/10_feature_importance.py).
- **Artifacts:** Saved to [`reports/figures/feature_importance.png`](file:///d:/incomelens-ai/reports/figures/feature_importance.png) and [`reports/feature_importance.csv`](file:///d:/incomelens-ai/reports/feature_importance.csv).
- **Methodological Limitation Identified:** MDI demonstrates severe selection bias toward high-cardinality continuous variables. For example, `fnlwgt` (a survey sampling weight with zero true predictive relationship to income) ranks as the **#7 most important feature** under MDI. Full details in [`reports/feature_importance_methodology.md`](file:///d:/incomelens-ai/reports/feature_importance_methodology.md).

### 8.2 Exact Probability-Space TreeExplainer SHAP
- **Tool:** Computed using `shap.TreeExplainer` on a stratified 1,000-sample training slice via `python -m src.incomelens.explainability`.
- **Artifacts:** Saved to [`reports/figures/shap_summary.png`](file:///d:/incomelens-ai/reports/figures/shap_summary.png) and [`reports/shap_summary.csv`](file:///d:/incomelens-ai/reports/shap_summary.csv).
- **Exact Additivity Guarantee:** For Random Forest classifiers, TreeExplainer operates in exact class probability space. The base value ($E[f(x)] = 0.2372$) plus the sum of SHAP contributions equals the model's predicted probability of `>50K` within numerical machine precision ($\max |\Delta| < 6 \times 10^{-8}$). See [`reports/shap_validation.md`](file:///d:/incomelens-ai/reports/shap_validation.md).
- **Top Predictive Drivers of High Income:**
  1. `capital-gain`: Strongest positive driver when non-zero.
  2. `education-num` / `education`: Directly increases high-income probability with higher degree attainment.
  3. `age`: Parabolic effect peaking in prime earning years (45–55).
  4. `relationship_Husband` & `marital-status_Married-civ-spouse`: Substantial positive attribution reflecting 1994 household filing dynamics.
  5. `hours-per-week`: Significant positive attribution above 40 hours/week.

---

## 9. Algorithmic Fairness & Demographic Disparities

A rigorous fairness audit was performed across demographic subgroups (`sex` and `race`) on the held-out test set ($N = 9,771$) using [`src/incomelens/fairness.py`](file:///d:/incomelens-ai/src/incomelens/fairness.py). Full audit results are documented in [`reports/fairness_report.md`](file:///d:/incomelens-ai/reports/fairness_report.md) and [`reports/fairness_metrics.csv`](file:///d:/incomelens-ai/reports/fairness_metrics.csv).

### Summary Fairness Disparity Table
| Sensitive Group | Group Size ($N$) | Selection Rate ($P(\hat{Y}=1)$) | Wilson 95% CI | Disparate Impact Ratio | Group Recall (Sensitivity) | 80% Rule Compliance |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Male (Reference)** | 6,561 | **24.25%** | [23.23%, 25.30%] | **1.000** | **63.36%** | Reference |
| **Female** | 3,210 | **7.29%** | [6.44%, 8.24%] | **0.301** | **52.86%** | **VIOLATION (< 0.80)** |
| **White (Reference)** | 8,367 | **19.82%** | [18.98%, 20.68%] | **1.000** | **62.33%** | Reference |
| **Black** | 934 | **9.42%** | [7.70%, 11.48%] | **0.475** | **47.66%** | **VIOLATION (< 0.80)** |
| **Asian-Pac-Islander** | 309 | **22.01%** | [17.75%, 26.96%] | **1.110** | **64.71%** | Compliant |
| **Amer-Indian-Eskimo** | 92 | **9.78%** | [5.21%, 17.61%] | **0.494** | **50.00%** | **VIOLATION (< 0.80)** |

### Core Ethical & Fairness Findings
1. **Severe Gender Selection Disparity:** The disparate impact ratio for females is **0.301**, severely below the EEOC four-fifths (80%) rule guideline. Females are selected at less than one-third the rate of males.
2. **Disparate Opportunity (Recall Gap):** Female high earners face an omission rate of $47.14\%$ (recall = $52.86\%$) compared to $36.64\%$ for males (recall = $63.36\%$).
3. **The "Fairness Through Blindness" Fallacy:** Naively dropping the `sex` or `race` columns during training does not eliminate group disparities. Proxy features—particularly `relationship_Husband`, `marital-status`, and `occupation`—reconstruct demographic membership and maintain disparate selection rates.

---

## 10. Interactive Application Launch

IncomeLens-AI features a modern full-stack production architecture with an authoritative Next.js 16 frontend communicating with a FastAPI ML backend, alongside a legacy developer Streamlit application.

### 10.1 Primary Production Application (Next.js 16 + FastAPI)

The production architecture separates the high-performance ML inference runtime from the responsive web frontend:

#### Step 1: Start the FastAPI Backend
From the project root (`D:\incomelens-ai`):
```powershell
python -m uvicorn src.incomelens.api:app --host 127.0.0.1 --port 8000 --reload
```
- REST API documentation & Swagger UI: `http://127.0.0.1:8000/docs`
- Health check: `http://127.0.0.1:8000/api/health`

#### Step 2: Start the Next.js Frontend
From the `frontend/` directory (`D:\incomelens-ai\frontend`):
```powershell
cd frontend
npm run dev
```
Open `http://localhost:3000` in your browser. All requests to `/api/*` are automatically reverse-proxied to `http://127.0.0.1:8000/api/*`.

#### Production Workspace Routes
- **`/overview`**: Executive dashboard, held-out test metrics, 2×2 confusion matrix, and system health.
- **`/predict`**: 14-feature interactive prediction form with presets and local SHAP waterfall attribution.
- **`/performance`**: Classification report, ROC-AUC, FPR, TNR, and diagnostic error rates.
- **`/explainability`**: TreeExplainer SHAP beeswarm plot and top global driver breakdown.
- **`/feature-importance`**: 105-feature MDI impurity ranking table with cardinality bias callouts.
- **`/fairness`**: 13-subgroup demographic audit, Wilson 95% CIs, and EEOC 80% rule compliance.
- **`/dataset`**: 1994 CPS Census database schema, split sizes, and duplicate record audit.
- **`/model-comparison`**: Baseline vs Logistic Regression vs Random Forest validation benchmarks.
- **`/experiments`**: Cryptographic model provenance registry (SHA-256 `f164a611...`) and artifact checklists.

### 10.2 Legacy Developer Interface (Streamlit)

For rapid local exploratory scripting, the original Streamlit application remains accessible:
```powershell
streamlit run app/Home.py
```
The application will launch locally at `http://localhost:8501`.

---

## 11. Automated Test Suite & Quality Verification

The repository contains **158 unit, integration, property-based, API contract, and regression tests** across 11 dedicated test modules.

### Executing the Tests
```powershell
# Run the entire test suite
python -m pytest tests -v

# Run with concise summary
python -m pytest tests -q
```

### Verified Test Suite Results
```text
============================== test session starts ==============================
platform win32 -- Python 3.14.7, pytest-9.1.1, pluggy-1.6.0
rootdir: D:\incomelens-ai
collected 158 items

tests/test_api.py ............                                             [  7%]
tests/test_data.py ...............                                         [ 17%]
tests/test_evaluation.py ..........                                       [ 23%]
tests/test_explainability.py ................                              [ 33%]
tests/test_fairness.py ..................                                  [ 44%]
tests/test_feature_importance.py ....................                      [ 56%]
tests/test_inference.py ...............                                    [ 65%]
tests/test_predictions.py ....................                             [ 77%]
tests/test_preprocessing.py .............                                  [ 85%]
tests/test_provenance.py ..........                                        [ 91%]
tests/test_training.py ..........                                          [100%]

====================== 158 passed, 207 warnings in 17.50s =======================
```

### Test Module Distribution
| Module | Test Count | Scope of Coverage |
| :--- | :---: | :--- |
| [`tests/test_api.py`](file:///d:/incomelens-ai/tests/test_api.py) | 12 | FastAPI endpoint health, model metadata, prediction contracts, metrics, fairness, and static mounts. |
| [`tests/test_data.py`](file:///d:/incomelens-ai/tests/test_data.py) | 14 | Raw schema validation, split isolation, missing value checks, duplicate handling. |
| [`tests/test_evaluation.py`](file:///d:/incomelens-ai/tests/test_evaluation.py) | 10 | Metric calculation, confusion matrix orientation, single-class safe handling, report schema. |
| [`tests/test_explainability.py`](file:///d:/incomelens-ai/tests/test_explainability.py) | 16 | TreeExplainer selection, SHAP output dimensions, feature alignment, additivity proof. |
| [`tests/test_fairness.py`](file:///d:/incomelens-ai/tests/test_fairness.py) | 18 | Subgroup counts, Wilson 95% CIs, zero-division safety, disparate impact ratios. |
| [`tests/test_feature_importance.py`](file:///d:/incomelens-ai/tests/test_feature_importance.py) | 20 | MDI Gini extraction, descending sort order, normalization to sum 1.0, figure saving. |
| [`tests/test_inference.py`](file:///d:/incomelens-ai/tests/test_inference.py) | 15 | Input schema validation, range checks, category whitelist, target leakage rejection, sanitized errors. |
| [`tests/test_predictions.py`](file:///d:/incomelens-ai/tests/test_predictions.py) | 20 | Artifact loading, class label mapping, probability range conservation, batch vs single consistency, determinism. |
| [`tests/test_preprocessing.py`](file:///d:/incomelens-ai/tests/test_preprocessing.py) | 13 | Preprocessor builders, median/mode imputation, one-hot encoding feature counts (105 columns). |
| [`tests/test_provenance.py`](file:///d:/incomelens-ai/tests/test_provenance.py) | 10 | Provenance manifest creation, SHA-256 verification, parseable JSON, secret absence assertion. |
| [`tests/test_training.py`](file:///d:/incomelens-ai/tests/test_training.py) | 10 | Train/test data isolation, seed determinism, pipeline estimator integrity, non-refitting. |

---

## 12. Limitations, Intended Use & Reproducibility

### 12.1 Appropriate & Intended Uses
- Academic benchmarking of tabular classification and hyperparameter tuning methods.
- Comparative research into tree interpretability (MDI selection bias vs. exact SHAP values).
- Diagnostic auditing of algorithmic fairness, demographic disparities, and demographic proxy effects in machine learning pipelines.
- Educational demonstrations of robust ML engineering (provenance tracking, interface contracts, and automated testing).

### 12.2 Prohibited & Out-of-Scope Uses
> [!CAUTION]
> This model and codebase must **NOT** be used for:
> - Automated credit underwriting, credit scoring, or loan origination decisions.
> - Employment hiring, applicant screening, compensation benchmarking, or termination decisions.
> - Housing, rental, or tenant screening applications.
> - Criminal justice risk assessments or bail evaluations.
> - Solely automated decision-making affecting individuals' economic or legal standing.

### 12.3 Known Dataset & Model Limitations
- **Dated Economic Realities:** The dataset reflects 1994 US labor market and economic conditions. Nominal dollar thresholds (\$50,000 in 1994 is equivalent to over \$105,000 today) and 1994 household labor force participation rates do not represent contemporary economies.
- **High-Income Recall Ceiling:** The model achieves 61.74% recall on `>50K`, failing to detect 38.26% of high-income earners (870 false negatives in 9,771 test cases).
- **Sampling Weight Flaw:** The `fnlwgt` attribute represents demographic sampling controls and contains no predictive relationship with individual earning potential, yet it distorts tree-based split importances.
- **Top-Coded Variables:** Capital gains and losses are top-coded at \$99,999 in the raw Census data, clipping extreme economic outliers.

### 12.4 Reproducibility Guarantees
- **Deterministic Pipeline:** Random seeds are fixed (`random_state=42`) across split generation, estimator initialization, and test suites.
- **Frozen Partitions:** Data partitions (`train.csv`, `validation.csv`, `test.csv`) are permanently saved in [`data/processed/`](file:///d:/incomelens-ai/data/processed/).
- **Cryptographic Provenance:** Artifact metadata, hardware environments, library versions, and SHA-256 digests are recorded in [`reports/artifact_manifest.json`](file:///d:/incomelens-ai/reports/artifact_manifest.json).

---

## 13. Comprehensive Documentation & Reports Index

| Category | Document / Report | Link | Description |
| :--- | :--- | :---: | :--- |
| **Specifications** | Model Card | [`reports/model_card.md`](file:///d:/incomelens-ai/reports/model_card.md) | Official Model Card covering architecture, quantitative metrics, and ethical limits. |
| | Dataset Card | [`docs/dataset_card.md`](file:///d:/incomelens-ai/docs/dataset_card.md) | In-depth description of the 1994 Census Adult data, collection, and features. |
| | Problem Statement | [`docs/problem_statement.md`](file:///d:/incomelens-ai/docs/problem_statement.md) | Formal mathematical problem formulation, scope, and non-goals. |
| | Inference Contract | [`docs/inference_interface_contract.md`](file:///d:/incomelens-ai/docs/inference_interface_contract.md) | Schema definitions, supported categorical levels, and validation rules. |
| **Audits & Security**| Security Review | [`reports/security_audit.md`](file:///d:/incomelens-ai/reports/security_audit.md) | Comprehensive credential scan, deserialization risk, and dependency review. |
| | Deployment Readiness | [`reports/deployment_readiness.md`](file:///d:/incomelens-ai/reports/deployment_readiness.md) | 10/10 PASS verification audit for local demonstration and execution. |
| | Final Regression Validation | [`reports/final_validation.md`](file:///d:/incomelens-ai/reports/final_validation.md) | 10/10 PASS exhaustive regression validation across all components. |
| | Release Checklist | [`reports/release_checklist.md`](file:///d:/incomelens-ai/reports/release_checklist.md) | Formal 13-point PASS release assessment with zero blocking defects. |
| | Final Project Review | [`reports/final_project_review.md`](file:///d:/incomelens-ai/reports/final_project_review.md) | Comprehensive code, repository, security, and artifact review. |
| | Engineering Handover | [`reports/engineering_handover.md`](file:///d:/incomelens-ai/reports/engineering_handover.md) | Production engineering handover with full metrics and inventory. |
| | Execution Audit | [`reports/execution_audit.md`](file:///d:/incomelens-ai/reports/execution_audit.md) | Standardized CLI command verification and PowerShell compatibility audit. |
| | Test Quality Audit | [`reports/test_quality_audit.md`](file:///d:/incomelens-ai/reports/test_quality_audit.md) | Detailed assessment of test suite independence, assertions, and mock hygiene. |
| | Data Integrity Audit | [`reports/data_integrity_audit.md`](file:///d:/incomelens-ai/reports/data_integrity_audit.md) | Missing values, exact duplicate rows, and schema anomaly inspection. |
| | Split Validation | [`reports/split_validation.md`](file:///d:/incomelens-ai/reports/split_validation.md) | Verification of leak-free grouped train/validation/test partitions. |
| | Metric Consistency | [`reports/metric_consistency_audit.md`](file:///d:/incomelens-ai/reports/metric_consistency_audit.md) | Verification of evaluation metrics against scikit-learn ground truth. |
| | Model Integrity | [`reports/model_integrity.md`](file:///d:/incomelens-ai/reports/model_integrity.md) | Architecture checks, pipeline step verification, and serialization safety. |
| **Analysis & Research**| Explainability Report | [`reports/explainability_report.md`](file:///d:/incomelens-ai/reports/explainability_report.md) | Detailed analysis of global TreeExplainer SHAP and Gini importance outputs. |
| | Fairness Audit | [`reports/fairness_report.md`](file:///d:/incomelens-ai/reports/fairness_report.md) | Full demographic parity and disparate impact analysis across sex and race. |
| | Fairness Metrics | [`reports/fairness_metrics.csv`](file:///d:/incomelens-ai/reports/fairness_metrics.csv) | Machine-readable table of group selection rates, error rates, and Wilson CIs. |
| | MDI Methodology | [`reports/feature_importance_methodology.md`](file:///d:/incomelens-ai/reports/feature_importance_methodology.md) | Methodological study of MDI selection bias on high-cardinality noise. |
| | SHAP Validation | [`reports/shap_validation.md`](file:///d:/incomelens-ai/reports/shap_validation.md) | Mathematical proof of TreeExplainer probability-space exact additivity. |
| | Baseline Benchmarks | [`reports/baseline_validation.md`](file:///d:/incomelens-ai/reports/baseline_validation.md) | Dummy classifier and logistic regression benchmark comparisons. |
| **Provenance** | Artifact Manifest | [`reports/artifact_manifest.json`](file:///d:/incomelens-ai/reports/artifact_manifest.json) | Cryptographic SHA-256 hashes, library versions, git commit, and timestamps. |
| | Report Workflow | [`reports/report_generation.md`](file:///d:/incomelens-ai/reports/report_generation.md) | Standard operating procedure for reproducible artifact generation. |
