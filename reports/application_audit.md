# IncomeLens-AI Application Architecture & Readiness Audit

**Author / Auditor:** Antigravity  
**Date:** October 26, 2026  
**Audited Directory:** [`app/`](file:///d:/incomelens-ai/app/)  
**Target Production Model Artifact:** [`models/random_forest_final.joblib`](file:///d:/incomelens-ai/models/random_forest_final.joblib)  
**Report Destination:** [`reports/application_audit.md`](file:///d:/incomelens-ai/reports/application_audit.md)  

---

## 1. Executive Summary

This audit assesses the readiness, architecture, user-input flow, model-serving mechanics, and dependencies of the interactive application in `app/`.

### Key Conclusions:
1. **Framework & Architecture:** The application is architected as a **Streamlit multi-page web application**, using the standard directory convention: entry point [`app/Home.py`](file:///d:/incomelens-ai/app/Home.py) with four dedicated subpages under [`app/pages/`](file:///d:/incomelens-ai/app/pages/).
2. **Current Execution State (Non-Operational):** All five application files under `app/` are currently **0-byte empty placeholders**. Running `streamlit run app/Home.py` fails immediately because `streamlit` is neither listed in `requirements.txt` nor installed in the active virtual environment (`.venv`), and the files contain no executable code.
3. **Pipeline Serving Compatibility:** The serialized model artifact [`models/random_forest_final.joblib`](file:///d:/incomelens-ai/models/random_forest_final.joblib) is an end-to-end scikit-learn `Pipeline` (`ColumnTransformer` + `RandomForestClassifier`). User inputs must be provided as a raw pandas DataFrame matching the 14 raw survey columns. The pipeline handles missing-value imputation, scaling, and one-hot encoding internally.
4. **Label & Probability Mapping Alert:** The classifier's internal `classes_` attribute is `array([0, 1])` (integers), **not** strings. Calling `model.predict(X)` yields `0` or `1`. The UI must explicitly map `0 -> '<=50K'` and `1 -> '>50K'`, and extract high-income probability from `predict_proba(X)[:, 1]`.
5. **Preservation of Existing Design:** The multi-page structure (`Home.py`, `1_Predict_Income.py`, `2_Model_Comparison.py`, `3_Explainability.py`, `4_Fairness_Audit.py`) is preserved. No second framework (e.g., Flask, FastAPI, Dash, Gradio) will be introduced.

---

## 2. Directory Layout & Multi-Page Framework Identification

### 2.1 File Inventory
```
app/
├── Home.py                       (0 bytes - Entry Point)
└── pages/
    ├── 1_Predict_Income.py       (0 bytes - Interactive Inference)
    ├── 2_Model_Comparison.py     (0 bytes - Validation Benchmarks)
    ├── 3_Explainability.py       (0 bytes - MDI & SHAP Visualizations)
    └── 4_Fairness_Audit.py       (0 bytes - Group Parity & Disparity Metrics)
```

### 2.2 Framework Identification & Entry Point
- **Web Framework:** **Streamlit** (Python multi-page web application framework).
- **Canonical Entry Point:** `app/Home.py`.
- **Execution Command:** `streamlit run app/Home.py` from repository root.
- **Subpage Auto-Discovery:** Streamlit automatically discovers and orders pages in `app/pages/` alphabetically by numerical prefix, generating the sidebar navigation.

---

## 3. Input Tracing, Preprocessing, and Inference Flow

### 3.1 Required Input Schema (14 Raw Survey Attributes)
To make a prediction, the application must construct a single-row `pd.DataFrame` containing the 14 canonical features:

```
[User Form / UI Widgets]
        │
        ▼ (Constructs 1-row DataFrame)
Raw DataFrame (14 columns):
  - age (numeric)
  - fnlwgt (numeric)
  - education-num (numeric)
  - capital-gain (numeric)
  - capital-loss (numeric)
  - hours-per-week (numeric)
  - workclass (categorical)
  - education (categorical)
  - marital-status (categorical)
  - occupation (categorical)
  - relationship (categorical)
  - race (categorical)
  - sex (categorical)
  - native-country (categorical)
        │
        ▼ (Direct call: model.predict(df) / model.predict_proba(df))
Pipeline Object: models/random_forest_final.joblib
  ├── Step 1: ColumnTransformer ('preprocessor')
  │     ├── SimpleImputer (median for numeric, most_frequent for categorical)
  │     ├── StandardScaler (numeric features)
  │     └── OneHotEncoder (105 sparse/dense categorical dummy features)
  └── Step 2: RandomForestClassifier ('classifier', 150 trees)
        │
        ▼
Output:
  - Class: 0 (<=50K) or 1 (>50K)
  - Probabilities: [P(<=50K), P(>50K)]
```

### 3.2 Preprocessing Anti-Pattern Warning
- **DO NOT** pre-transform user inputs in `app/` before passing them to `model`.
- Because `models/random_forest_final.joblib` already bundles the fitted `ColumnTransformer`, passing pre-transformed features will crash with a shape mismatch error (expected 14 columns, received 105).

---

## 4. Model-Loading Paths, Caching, and Error Handling

### 4.1 Path Resolution
To ensure portability across environments and working directories:
```python
from pathlib import Path

# When executed via 'streamlit run app/Home.py'
ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "random_forest_final.joblib"
```
The application must use dynamic relative path resolution based on `__file__`, rather than hardcoded absolute Windows paths (`D:\incomelens-ai\...`).

### 4.2 Streamlit Resource Caching (`@st.cache_resource`)
- `random_forest_final.joblib` is an 85 MB serialized ensemble of 150 deep decision trees.
- Loading this artifact takes approximately $1.5$–$2.5$ seconds.
- Without Streamlit's `@st.cache_resource` decorator, the application would re-deserialize the 85 MB model on every widget interaction or slider change, causing severe UI lag.
- **Recommended loader:**
```python
import joblib
import streamlit as st


@st.cache_resource(show_spinner="Loading production Random Forest model...")
def load_production_model():
  if not MODEL_PATH.is_file():
    raise FileNotFoundError(f"Model artifact not found at: {MODEL_PATH}")
  return joblib.load(MODEL_PATH)
```

### 4.3 Error Handling
If the model file is missing or corrupted:
- Catch `FileNotFoundError` or deserialization exceptions.
- Display a clean `st.error` alert explaining that `models/random_forest_final.joblib` is missing and prompt the user to train or restore the model.

---

## 5. Input Field Specifications, Categorical Options, and Numerical Bounds

From our audit of the fitted `preprocessor` and training distribution, the following input choices and ranges must be enforced:

### 5.1 Categorical Options (from fitted `OneHotEncoder`)
1. **`workclass` (8 options):** `'Private'`, `'Self-emp-not-inc'`, `'Self-emp-inc'`, `'Federal-gov'`, `'Local-gov'`, `'State-gov'`, `'Without-pay'`, `'Never-worked'`. (Default: `'Private'`).
2. **`education` (16 options):** `'Bachelors'`, `'Some-college'`, `'11th'`, `'HS-grad'`, `'Prof-school'`, `'Assoc-acdm'`, `'Assoc-voc'`, `'9th'`, `'7th-8th'`, `'12th'`, `'Masters'`, `'1st-4th'`, `'10th'`, `'Doctorate'`, `'5th-6th'`, `'Preschool'`. (Default: `'Bachelors'`).
3. **`marital-status` (7 options):** `'Married-civ-spouse'`, `'Divorced'`, `'Never-married'`, `'Separated'`, `'Widowed'`, `'Married-spouse-absent'`, `'Married-AF-spouse'`. (Default: `'Married-civ-spouse'`).
4. **`occupation` (14 options):** `'Tech-support'`, `'Craft-repair'`, `'Other-service'`, `'Sales'`, `'Exec-managerial'`, `'Prof-specialty'`, `'Handlers-cleaners'`, `'Machine-op-inspct'`, `'Adm-clerical'`, `'Farming-fishing'`, `'Transport-moving'`, `'Priv-house-serv'`, `'Protective-serv'`, `'Armed-Forces'`. (Default: `'Exec-managerial'`).
5. **`relationship` (6 options):** `'Wife'`, `'Own-child'`, `'Husband'`, `'Not-in-family'`, `'Other-relative'`, `'Unmarried'`. (Default: `'Husband'`).
6. **`race` (5 options):** `'White'`, `'Asian-Pac-Islander'`, `'Amer-Indian-Eskimo'`, `'Other'`, `'Black'`. (Default: `'White'`).
7. **`sex` (2 options):** `'Male'`, `'Female'`. (Default: `'Male'`).
8. **`native-country` (41 options):** `'United-States'`, `'Mexico'`, `'Philippines'`, `'Germany'`, `'Canada'`, `'Puerto-Rico'`, `'El-Salvador'`, `'India'`, `'Cuba'`, `'England'`, `'Jamaica'`, `'South'`, `'China'`, `'Italy'`, `'Dominican-Republic'`, `'Vietnam'`, `'Guatemala'`, `'Japan'`, `'Poland'`, `'Columbia'`, `'Taiwan'`, `'Haiti'`, `'Iran'`, `'Portugal'`, `'Nicaragua'`, `'Peru'`, `'Greece'`, `'Ecuador'`, `'France'`, `'Ireland'`, `'Hong'`, `'Trinadad&Tobago'`, `'Cambodia'`, `'Thailand'`, `'Laos'`, `'Yugoslavia'`, `'Outlying-US(Guam-USVI-etc)'`, `'Hungary'`, `'Honduras'`, `'Scotland'`, `'Holand-Netherlands'`. (Default: `'United-States'`).

### 5.2 Numeric Features, Ranges, and Defaults
1. **`age`:** Integer range $[17, 90]$, default $38$, step $1$.
2. **`education-num`:** Integer range $[1, 16]$, default $10$ (or synchronized automatically with the selected `education` credential).
   - *UI Enhancement:* Auto-bind `education-num` to `education` (e.g. `'Bachelors' -> 13`, `'Masters' -> 14`, `'Doctorate' -> 16`, `'HS-grad' -> 9`) to eliminate contradictory user inputs.
3. **`hours-per-week`:** Integer range $[1, 99]$, default $40$, step $1$.
4. **`capital-gain`:** Float range $[0.0, 99999.0]$, default $0.0$, step $500.0$.
5. **`capital-loss`:** Float range $[0.0, 4356.0]$, default $0.0$, step $100.0$.
6. **`fnlwgt` (Final Sampling Weight):**
   - *Special Handling:* `fnlwgt` is a census demographic expansion factor (range $13,769$ to $1,490,400$, mean $\approx 189,729$).
   - It is meaningless to an end-user.
   - The UI should default to median ($178,356$) with an expander/advanced setting explaining its census weighting role.

---

## 6. Prediction Label & Probability Mapping

- **Internal Model Output:**
  ```python
  raw_pred = model.predict(input_df)[0]  # Integer 0 or 1
  probs = model.predict_proba(input_df)[0]  # Array of [P(<=50K), P(>50K)]
```
- **Label Mapping:**
  - `0` $\rightarrow$ `<=50K` ("Income $\le$ \$50,000 / year")
  - `1` $\rightarrow$ `>50K` ("Income $>$ \$50,000 / year")
- **Probability Interpretation:**
  - `P_high = float(probs[1])`
  - `P_low = float(probs[0])`
  - At standard decision threshold $\tau = 0.5$, prediction is high income if $P_{\text{high}} \ge 0.50$.
- **UI Display Recommendation:**
  - Display probability progress bars or metric cards (e.g., `82.4% Probability of >$50K`).
  - Add boundary case warnings if $0.40 \le P_{\text{high}} \le 0.60$ (indicating low model confidence requiring human review).

---

## 7. Dependency & Import Audit

### 7.1 Missing Direct Dependencies
| Package | Required By | Status in `.venv` | Status in `requirements.txt` | Action Needed |
| :--- | :--- | :---: | :---: | :--- |
| **`streamlit`** | `app/*.py` | **Not Installed** | **Missing** | Install `streamlit>=1.40.0` and append to `requirements.txt` |
| **`incomelens`** | `app/*.py` | Not Installed (no editable install) | Missing packaging | Use `sys.path.insert(0, str(ROOT / 'src'))` in app files |

### 7.2 Broken / Outdated Imports to Avoid
- Do not import `scripts.*` inside `app/`.
- Import clean library routines from `src/incomelens/` (`data.py`, `evaluation.py`, `fairness.py`, `preprocessing.py`).

---

## 8. Minimal Implementation Plan

To make the existing multi-page Streamlit application fully functional without altering its visual architecture or creating competing frameworks:

### Step 1: Dependency Installation & Environment Alignment
1. Install `streamlit` in the project virtual environment:
   ```powershell
   .\.venv\Scripts\python.exe -m pip install streamlit
   ```
2. Add `streamlit` to [`requirements.txt`](file:///d:/incomelens-ai/requirements.txt).

### Step 2: Implement Entry Point (`app/Home.py`)
- Project hero banner and overview.
- Dataset card summary and 1994 Census context.
- Overview of the 4 subpages with direct navigation links.
- Key model metrics summary card (Accuracy: $86.83\%$, High-Income Precision: $77.10\%$, Recall: $61.74\%$).
- Prominent operational warnings regarding non-causal inference and human review requirements.

### Step 3: Implement Interactive Inference (`app/pages/1_Predict_Income.py`)
- Two-column input form:
  - Left column: Demographics (`age`, `sex`, `race`, `native-country`, `marital-status`, `relationship`).
  - Right column: Human capital & financials (`education`, `workclass`, `occupation`, `hours-per-week`, `capital-gain`, `capital-loss`).
  - Advanced expander: `fnlwgt` (defaulted to median $178,356$).
- Auto-sync `education-num` to `education`.
- Input validation: ensure numerical inputs are positive and within valid ranges.
- Inference execution using `@st.cache_resource` model.
- Clean output card showing predicted income class, probability gauges, decision threshold indicator ($\tau = 0.5$), and uncertainty callout if within $[0.40, 0.60]$.

### Step 4: Implement Model Comparison Dashboard (`app/pages/2_Model_Comparison.py`)
- Load and display [`reports/validation_model_comparison.csv`](file:///d:/incomelens-ai/reports/validation_model_comparison.csv).
- Render interactive bar charts comparing validation accuracy, balanced accuracy, precision, recall, F1, and ROC-AUC across candidate models (Logistic Regression, Decision Tree, Random Forest, Gradient Boosting).
- Document why Random Forest was selected as the final production model.

### Step 5: Implement Explainability Dashboard (`app/pages/3_Explainability.py`)
- Display the global feature importance bar chart ([`reports/figures/feature_importance.png`](file:///d:/incomelens-ai/reports/figures/feature_importance.png)) with clear MDI/Gini explanations and cardinality bias notes.
- Display the SHAP beeswarm summary plot ([`reports/figures/shap_summary.png`](file:///d:/incomelens-ai/reports/figures/shap_summary.png)) explaining exact probability-space attributions.
- Display the top 15 features table with directionality notes.
- Explicit disclaimers regarding non-causal interpretation.

### Step 6: Implement Fairness Audit Dashboard (`app/pages/4_Fairness_Audit.py`)
- Load and display the quantitative metrics from [`reports/fairness_metrics.csv`](file:///d:/incomelens-ai/reports/fairness_metrics.csv).
- Interactive filter to inspect disparities across Sex, Race, Age Group, and Geographic Origin.
- Disparate Impact Ratio gauge relative to the EEOC $0.80$ four-fifths rule.
- Display 95% Wilson score confidence intervals and visual alert banners for flagged small-sample subgroups (`Amer-Indian-Eskimo`, `Other` race, `<25` age group).
- Include disclaimers on proxy discrimination and the limits of metric compliance.

### Step 7: Automated Integration Verification
- Add automated tests in `tests/test_app.py` verifying that all 5 Streamlit page scripts compile without syntax errors and that model loading and dummy prediction execute cleanly without missing dependencies.
