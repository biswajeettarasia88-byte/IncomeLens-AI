# IncomeLens-AI Inference Interface Contract

**Module:** [`src/incomelens/inference.py`](file:///d:/incomelens-ai/src/incomelens/inference.py)  
**UI Interface:** [`app/pages/1_Predict_Income.py`](file:///d:/incomelens-ai/app/pages/1_Predict_Income.py)  
**Validated Production Model:** [`models/random_forest_final.joblib`](file:///d:/incomelens-ai/models/random_forest_final.joblib)  
**Test Suite:** [`tests/test_inference.py`](file:///d:/incomelens-ai/tests/test_inference.py)

---

## 1. Overview & Service Architecture

The `IncomePredictor` class provides a production-grade inference contract for single-instance and batch predictions. It guarantees:
- **No Retraining:** Loads the validated production model directly from disk.
- **Preprocessing Pipeline Reuse:** Automatically reuses the fitted `ColumnTransformer` inside the model pipeline (or extracted from the training artifact for standalone estimators).
- **Strict Input Validation:** Enforces non-null checks, type conversions, range checks, and supported category matching.
- **Data Leakage Prevention:** Rejects inputs containing the target column `income`.
- **Sanitized Errors:** Raises actionable errors (`ValidationError`, `InferenceError`) without exposing internal server paths or stack traces.
- **Safe Output Mapping:** Uses model class labels (`classes_`) and only returns probabilities when supported by the underlying model.

---

## 2. Input Specification & Feature Contract

Predictions require a single dictionary or `pd.DataFrame` containing the exact **14 raw survey attributes**.

### 2.1 Numeric Features (6)
| Field Name | Type | Allowed Range | Default (UI) | Validation Rule |
| :--- | :--- | :---: | :---: | :--- |
| `age` | Integer | $[17, 100]$ | 38 | Positive integer; within census survey bounds. |
| `education-num` | Integer | $[1, 16]$ | Auto | Synchronized with selected `education` credential. |
| `hours-per-week` | Integer | $[1, 99]$ | 40 | Positive integer; standard workweek limits. |
| `capital-gain` | Float | $[0.0, 99999.0]$ | 0.0 | Non-negative numeric value. |
| `capital-loss` | Float | $[0.0, 4356.0]$ | 0.0 | Non-negative numeric value. |
| `fnlwgt` | Float | $> 0.0$ | 178,356.0 | Census sampling weight; defaults to survey median. |

### 2.2 Categorical Features (8)
All string inputs are stripped of leading/trailing whitespace. Inputs must match the categories recognized by the fitted `OneHotEncoder`:

1. **`workclass` (8):** `'Federal-gov'`, `'Local-gov'`, `'Never-worked'`, `'Private'`, `'Self-emp-inc'`, `'Self-emp-not-inc'`, `'State-gov'`, `'Without-pay'`.
2. **`education` (16):** `'10th'`, `'11th'`, `'12th'`, `'1st-4th'`, `'5th-6th'`, `'7th-8th'`, `'9th'`, `'Assoc-acdm'`, `'Assoc-voc'`, `'Bachelors'`, `'Doctorate'`, `'HS-grad'`, `'Masters'`, `'Preschool'`, `'Prof-school'`, `'Some-college'`.
3. **`marital-status` (7):** `'Divorced'`, `'Married-AF-spouse'`, `'Married-civ-spouse'`, `'Married-spouse-absent'`, `'Never-married'`, `'Separated'`, `'Widowed'`.
4. **`occupation` (14):** `'Adm-clerical'`, `'Armed-Forces'`, `'Craft-repair'`, `'Exec-managerial'`, `'Farming-fishing'`, `'Handlers-cleaners'`, `'Machine-op-inspct'`, `'Other-service'`, `'Priv-house-serv'`, `'Prof-specialty'`, `'Protective-serv'`, `'Sales'`, `'Tech-support'`, `'Transport-moving'`.
5. **`relationship` (6):** `'Husband'`, `'Not-in-family'`, `'Other-relative'`, `'Own-child'`, `'Unmarried'`, `'Wife'`.
6. **`race` (5):** `'Amer-Indian-Eskimo'`, `'Asian-Pac-Islander'`, `'Black'`, `'Other'`, `'White'`.
7. **`sex` (2):** `'Female'`, `'Male'`.
8. **`native-country` (41):** 41 countries (e.g. `'United-States'`, `'Mexico'`, `'Canada'`, etc.).

### 2.3 Prohibited Input Fields
- **`income`:** The target column is prohibited. Supplying `income` in the input payload immediately raises `ValidationError`.

---

## 3. Output Contract

Calling `IncomePredictor.predict(data)` returns a structured dictionary:

```python
{
    "prediction": str,  # '<=50K' or '>50K'
    "raw_prediction": int,  # 0 or 1
    "high_income_probability": (
        float | None
    ),  # Float [0.0, 1.0] or None if unsupported
    "low_income_probability": (
        float | None
    ),  # Float [0.0, 1.0] or None if unsupported
    "probabilities": {  # Dictionary mapping class -> probability
        "<=50K": float,
        ">50K": float,
    }
    | None,
    "confidence_level": (
        str  # 'High', 'Moderate', or 'Low (Boundary Case)'
    ),
}
```

### 3.1 Decision Rules & Probability Handling
- **Decision Threshold:** $\tau = 0.50$. If $P(\text{income} > 50\text{K}) \ge 0.50$, `prediction = '>50K'`.
- **Confidence Rating:**
  - `High`: $|P - 0.50| \ge 0.25$ ($P \ge 0.75$ or $P \le 0.25$).
  - `Moderate`: $0.10 \le |P - 0.50| < 0.25$.
  - `Low (Boundary Case)`: $|P - 0.50| < 0.10$ ($0.40 \le P \le 0.60$). Triggers a user-facing review alert.
- **Models Without Probabilities:** If an estimator lacks `predict_proba`, `high_income_probability` returns `None` and `confidence_level` returns an explanatory flag.

---

## 4. Error Handling Contract

All exceptions raised by the inference interface inherit from standard exceptions with sanitized, user-friendly messages:

1. **`ValidationError` (subclass of `ValueError`):**
   - *Missing fields:* `"Missing required input fields: age, workclass. All 14 features are required."`
   - *Target leakage:* `"Field 'income' is a prediction target and cannot be provided as an input feature."`
   - *Out-of-range numeric:* `"Field 'age' must be between 17 and 100 years. Received: -5"`
   - *Unsupported category:* `"Unsupported category 'Astronaut' for field 'workclass'. Supported options include: Federal-gov, Local-gov, Never-worked, Private..."`
2. **`InferenceError` (subclass of `RuntimeError`):**
   - Raised for backend pipeline or model failures without exposing local Windows file paths, environment variables, or raw Python stack traces.
