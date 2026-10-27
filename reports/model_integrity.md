# IncomeLens AI — Model Integrity & Pipeline Inspection Report

**Audit Date:** October 20, 2026  
**Auditor:** Senior Machine Learning Engineer  
**Project Path:** `D:\incomelens-ai`  
**Report Destination:** `reports/model_integrity.md`

---

## 1. Executive Summary

An audit of the primary production model artifact [`models/random_forest_final.joblib`](file:///D:/incomelens-ai/models/random_forest_final.joblib) and the code that generated it ([`scripts/09_final_evaluate.py`](file:///D:/incomelens-ai/scripts/09_final_evaluate.py)) was conducted.

### Key Conclusions:
1. **Load Success & Object Completeness:** The artifact loads cleanly with zero warnings or deserialization errors. It contains a complete, self-contained **`sklearn.pipeline.Pipeline`** combining feature preprocessing (`ColumnTransformer`) and the final estimator (`RandomForestClassifier`).
2. **Serving Contract Confirmed:** Because the pipeline embeds its own `ColumnTransformer`, `model.predict()` and `model.predict_proba()` **strictly expect raw 14-column tabular inputs**. Passing pre-transformed 105-column matrices directly to `model.predict()` raises a `ValueError`.
3. **Inference Functionality Verified:** Prediction and probability estimation were verified on sample test instances, generating valid class assignments (`0` and `1`) with probability distributions summing to $1.0$.
4. **Provenance & Zero Leakage:** The model was trained via `scripts/09_final_evaluate.py` on the development partition (`train.csv` + `validation.csv` = 39,071 rows). The held-out test split (`test.csv` = 9,771 rows) was never used for training or fitting.
5. **No Retraining or Replacement Required:** The model artifact is mathematically sound, uncorrupted, and matches all reported test benchmarks (86.83% accuracy, 0.9181 ROC-AUC).

---

## 2. Artifact Identity & Cryptographic Hashes

- **File Path:** `D:\incomelens-ai\models\random_forest_final.joblib`
- **File Size:** 50,084,507 bytes (~50.08 MB)
- **Last Modified Timestamp:** `10/20/2026 1:26:57 PM`
- **Cryptographic Hashes:**
  - **MD5:** `f903788b033d8e2c57cfac26579c5224`
  - **SHA-256:** `f164a61183466b25186377e8946f4cb74b2df924f69e50224f5d8de72858a285`

---

## 3. Pipeline Architecture & Component Inspection

The deserialized object is an instance of `sklearn.pipeline.Pipeline` consisting of two distinct stages:

```
Raw Input DataFrame (14 Columns: 6 numeric, 8 categorical)
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│ Step 1: "preprocessor" (sklearn.compose.ColumnTransformer)   │
│                                                             │
│ ├─ "numeric" (6 columns):                                   │
│ │   ['age', 'fnlwgt', 'education-num',                      │
│ │    'capital-gain', 'capital-loss', 'hours-per-week']      │
│ │   Steps: SimpleImputer(median) -> StandardScaler()        │
│ │                                                           │
│ └─ "categorical" (8 columns):                               │
│     ['workclass', 'education', 'marital-status',            │
│      'occupation', 'relationship', 'race', 'sex',           │
│      'native-country']                                      │
│     Steps: SimpleImputer(mode) -> OneHotEncoder(sparse=True)│
│                                                             │
│ Output: 105 Transformed Numeric Features                    │
└─────────────────────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│ Step 2: "classifier" (sklearn.ensemble.RandomForest)        │
│                                                             │
│   n_estimators = 150                                        │
│   min_samples_leaf = 2                                      │
│   random_state = 42                                         │
│   n_jobs = -1                                               │
│   classes_ = [0, 1]                                         │
│   n_features_in_ = 105                                      │
└─────────────────────────────────────────────────────────────┘
```

### 3.1 Preprocessor Step Inspection (`ColumnTransformer`)
- **Input Features Expected (`feature_names_in_`):**
  `['age', 'fnlwgt', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week', 'workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'native-country']`
- **Learned Imputation Medians:**
  - `age`: 37.0
  - `fnlwgt`: 178,272.0
  - `education-num`: 10.0
  - `capital-gain`: 0.0
  - `capital-loss`: 0.0
  - `hours-per-week`: 40.0
- **Learned Imputation Modes:**
  - `workclass`: `'Private'`
  - `education`: `'HS-grad'`
  - `marital-status`: `'Married-civ-spouse'`
  - `occupation`: `'Prof-specialty'`
  - `relationship`: `'Husband'`
  - `race`: `'White'`
  - `sex`: `'Male'`
  - `native-country`: `'United-States'`
- **One-Hot Encoding Dimensionality:**
  - 8 categorical features produce 99 binary dummy features.
  - Combined with 6 numeric features $\rightarrow$ **105 total output features**.

### 3.2 Classifier Step Inspection (`RandomForestClassifier`)
- **Estimator Class:** `sklearn.ensemble.RandomForestClassifier`
- **Fitted State:** Fully fitted (`classes_ = [0, 1]`, `n_classes_ = 2`, `n_features_in_ = 105`).
- **Ensemble Dimensions:**
  - Trees fitted: **150 estimators**
  - Tree depths: Minimum = 33, Maximum = 61, Mean = 45.98
  - Leaf counts per tree: Minimum = 1,699, Maximum = 2,570, Mean = 2,084.13
- **Hyperparameter Configuration:**
  - `criterion`: `'gini'`
  - `n_estimators`: `150`
  - `min_samples_leaf`: `2`
  - `max_depth`: `None` (unbounded, constrained by `min_samples_leaf=2`)
  - `class_weight`: `None`
  - `random_state`: `42`
  - `n_jobs`: `-1`

---

## 4. Input Expectations & Serving Contract

### 4.1 Expected Input Type: Raw Features
Because the top-level pipeline includes `preprocessor`, user code, REST APIs, or Streamlit forms must pass **raw demographic features** directly into `model.predict()`:

```python
# Correct Usage (Raw DataFrame)
raw_input = pd.DataFrame([{
    "age": 39,
    "workclass": "Private",
    "fnlwgt": 77516,
    "education": "Bachelors",
    "education-num": 13,
    "marital-status": "Never-married",
    "occupation": "Adm-clerical",
    "relationship": "Not-in-family",
    "race": "White",
    "sex": "Male",
    "capital-gain": 2174,
    "capital-loss": 0,
    "hours-per-week": 40,
    "native-country": "United-States",
}])

prediction = model.predict(raw_input)           # Returns: array([0])
probabilities = model.predict_proba(raw_input)   # Returns: array([[0.85, 0.15]])
```

### 4.2 Behavior on Transformed Input
Passing a pre-transformed 105-column matrix directly into `model.predict()` fails intentionally:
```text
ValueError: X has 105 features, but ColumnTransformer is expecting 14 features as input.
```

*Note for Explainability:* To explain the model with TreeSHAP, explainers must explain the inner tree estimator (`model.named_steps['classifier']`) using preprocessed data (`model.named_steps['preprocessor'].transform(X)`), not the outer `Pipeline`.

---

## 5. Live Inference Verification

Inference was tested against 5 test instances from `data/processed/test.csv`:

| Sample Index | Ground Truth | Model Prediction | Probability `<=50K` ($P_0$) | Probability `>50K` ($P_1$) | Correct? |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **0** | `<=50K` (0) | `0` | `0.8981` | `0.1019` | Yes |
| **1** | `>50K` (1) | `1` | `0.4153` | `0.5847` | Yes |
| **2** | `<=50K` (0) | `0` | `0.7072` | `0.2928` | Yes |
| **3** | `<=50K` (0) | `0` | `0.8081` | `0.1919` | Yes |
| **4** | `<=50K` (0) | `0` | `0.7410` | `0.2590` | Yes |

- **Probability Validation:** $P_0 + P_1 = 1.0000$ across all samples.
- **Latency:** Sub-millisecond execution for single-sample inference.

---

## 6. Training Provenance & Protocol

The training script [`scripts/09_final_evaluate.py`](file:///D:/incomelens-ai/scripts/09_final_evaluate.py) established the exact training protocol:
1. **Dataset:** Concatenated `data/processed/train.csv` (29,301 rows) and `data/processed/validation.csv` (9,770 rows) into a unified development dataset of **39,071 rows**.
2. **Fitting:** Executed `model.fit(X_dev, y_dev)` with `random_state=42`.
3. **Evaluation:** Evaluated strictly on the held-out `data/processed/test.csv` (9,771 rows).
4. **Metrics Recorded:**
   - Accuracy: `86.83%`
   - Balanced Accuracy: `78.09%`
   - Precision (`>50K`): `77.10%`
   - Recall (`>50K`): `61.74%`
   - F1-Score (`>50K`): `68.57%`
   - ROC-AUC: `0.9181`
5. **Serialization:** Dumped via `joblib.dump(model, "models/random_forest_final.joblib")`.

---

## 7. Library-Version Compatibility & Security Risks

### 7.1 Serialization Compatibility Risks
The model was compiled using:
- **Python:** `3.14.7`
- **scikit-learn:** `1.9.1`
- **joblib:** `1.6.0`
- **NumPy:** `2.5.4`

> [!WARNING]
> Serialized scikit-learn tree ensembles rely on internal C/Cython data structures (`Tree` structs). Upgrading or downgrading `scikit-learn` across minor versions (e.g. to 1.8 or 1.10) can produce `InconsistentVersionWarning` or cause hard crashes during deserialization.
> 
> **Recommendation:** Lock `scikit-learn==1.9.1` and `joblib==1.6.0` in `requirements.txt`.

### 7.2 Security Warning on Untrusted Artifacts
Because `joblib` relies on Python's `pickle` engine, loading untrusted `.joblib` files from the internet or unauthenticated users can execute arbitrary shellcode.
- **Audit finding:** `models/random_forest_final.joblib` was generated locally within this repository and matches recorded local development outputs.
- **Policy:** Never replace or load `.joblib` files from untrusted external repositories.

---

## 8. Final Verdict

`models/random_forest_final.joblib` is **valid, self-contained, fully functional, and ready for deployment**. No retraining, re-serialization, or replacement is required.
