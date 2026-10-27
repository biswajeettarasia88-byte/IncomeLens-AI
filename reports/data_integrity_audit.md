# IncomeLens AI — Data Integrity & Dataset Audit Report

**Audit Date:** October 20, 2026  
**Auditor:** Senior Machine Learning Engineer  
**Project Path:** `D:\incomelens-ai`  
**Report Destination:** `reports/data_integrity_audit.md`

---

## 1. Executive Summary & Audit Scope

A comprehensive data integrity audit was conducted across all datasets in `data/`, covering the raw source data (`data/raw/adult.csv`), the intermediate cleaned dataset (`data/processed/adult_cleaned.csv`), and the partitioned production splits (`train.csv`, `validation.csv`, `test.csv`).

### High-Level Findings:
1. **Target Column Verified:** `income` is consistently present across all 5 datasets. Raw target labels containing trailing periods from legacy partitions (`<=50K.`, `>50K.`) are cleanly normalized to binary classes (`<=50K`, `>50K`) in all processed files.
2. **Strict Schema Consistency:** All 5 datasets exhibit 100% column name alignment and column ordering across all 15 columns (6 numeric, 8 categorical, 1 target). There are zero duplicate column names.
3. **Partition Isolation & Zero Leakage:** The grouped splitting strategy (`GroupShuffleSplit` on hashed feature vectors) completely isolated duplicate and near-duplicate instances. The intersection of feature groups across train, validation, and test is **0 (zero leakage)**.
4. **Vocabulary Coverage:** There are **zero out-of-vocabulary categories** in validation or test across all 8 categorical features.
5. **Preprocessing State Confirmed:** The CSV files contain **raw, unscaled, un-imputed, and un-encoded feature values**.
6. **Model Expectation Confirmed:** The production pipeline (`models/random_forest_final.joblib`) embeds `ColumnTransformer` and **expects raw 14-column tabular inputs**, not pre-transformed numeric matrices.
7. **Architectural Redundancy Identified:** `education` (16 categorical levels) and `education-num` (integers 1–16) share an exact 1:1 bijection, meaning education information is redundantly modeled twice.
8. **Label Ambiguity:** 5 distinct feature combinations (demographic profiles) carry contradictory income labels (both `<=50K` and `>50K`). All conflicting records are safely contained within individual splits (4 in train, 1 in val, 0 in test).

---

## 2. Dataset Inventory & Structural Dimensions

| Dataset Path | Partition Role | Total Rows | Total Columns | Feature Columns | Target Column | File Size (Bytes) | Exact Duplicates | Missing Cells |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `data/raw/adult.csv` | Raw UCI Adult | 48,842 | 15 | 14 | `income` | 5,336,181 | 29 | 6,465 |
| `data/processed/adult_cleaned.csv` | Standardized Full | 48,842 | 15 | 14 | `income` | 5,319,900 | 52 | 6,465 |
| `data/processed/train.csv` | Grouped Train (60%) | 29,301 | 15 | 14 | `income` | 3,190,538 | 26 | 3,961 |
| `data/processed/validation.csv` | Grouped Val (20%) | 9,770 | 15 | 14 | `income` | 1,064,781 | 12 | 1,212 |
| `data/processed/test.csv` | Grouped Test (20%) | 9,771 | 15 | 14 | `income` | 1,064,895 | 14 | 1,292 |
| **Sum of Splits** | *Train + Val + Test* | **48,842** | **15** | **14** | `income` | — | **52** | **6,465** |

*Verification:* The sum of split rows ($29,301 + 9,770 + 9,771 = 48,842$) exactly matches the parent cleaned dataset.

---

## 3. Target Column Audit (`income`)

### 3.1 Raw vs. Processed Target Distribution
In `data/raw/adult.csv`, four distinct target string variations exist due to historical census test-set formatting (trailing period):
- `<=50K`: 24,720 (50.61%)
- `<=50K.`: 12,435 (25.46%)
- `>50K`: 7,841 (16.05%)
- `>50K.`: 3,846 (7.87%)

In `data/processed/adult_cleaned.csv` and all splits, `scripts/02_prepare_data.py` standardized these strings:
- `<=50K`: 37,155 (76.07%)
- `>50K`: 11,687 (23.93%)

### 3.2 Target Distribution by Partition
| Partition | Total Rows | `<=50K` (Class 0) | `>50K` (Class 1) | Positive Class % | Target Nulls |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Train** (`train.csv`) | 29,301 | 22,288 | 7,013 | **23.93%** | 0 (0.0%) |
| **Validation** (`validation.csv`) | 9,770 | 7,370 | 2,400 | **24.56%** | 0 (0.0%) |
| **Test** (`test.csv`) | 9,771 | 7,497 | 2,274 | **23.27%** | 0 (0.0%) |
| **Combined** | 48,842 | 37,155 | 11,687 | **23.93%** | 0 (0.0%) |

*Conclusion:* The class balance is preserved with high fidelity across all partitions (~23.3% to ~24.6% positive class). There are zero missing or malformed target labels.

---

## 4. Column Schema & Data Type Verification

### 4.1 Schema Consistency
All 5 files share an identical column ordering and naming convention:
```python
COLUMNS = [
    'age',              # Numeric (int64)
    'workclass',        # Categorical (string)
    'fnlwgt',           # Numeric (int64)
    'education',        # Categorical (string)
    'education-num',    # Numeric (int64)
    'marital-status',   # Categorical (string)
    'occupation',       # Categorical (string)
    'relationship',     # Categorical (string)
    'race',             # Categorical (string)
    'sex',              # Categorical (string)
    'capital-gain',     # Numeric (int64)
    'capital-loss',     # Numeric (int64)
    'hours-per-week',   # Numeric (int64)
    'native-country',   # Categorical (string)
    'income'            # Target (string)
]
```

- **Duplicate Column Names:** None in any file.
- **Whitespace Issues:** All column names and string cells are trimmed of leading/trailing whitespace.
- **Literal Placeholder Tokens:** In `data/raw/adult.csv`, literal `'?'` tokens were mapped to `pd.NA`. There are zero remaining literal `'?'` strings in any dataset.

### 4.2 Numeric Feature Bounds (`train.csv`)
| Feature | Min | Max | Mean | Median | Std Dev | Physical / Domain Validity |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `age` | 17 | 90 | 38.64 | 37.0 | 13.71 | Valid census working age range |
| `fnlwgt` | 13,769 | 1,490,400 | 189,664.13 | 178,489.0 | 105,604.02 | Census sampling weight range |
| `education-num` | 1 | 16 | 10.08 | 10.0 | 2.57 | Discrete integer grade scale |
| `capital-gain` | 0 | 99,999 | 1,079.19 | 0.0 | 7,458.21 | 99,999 is top-coded census cap |
| `capital-loss` | 0 | 4,356 | 87.50 | 0.0 | 403.00 | Valid loss amount |
| `hours-per-week` | 1 | 99 | 40.42 | 40.0 | 12.39 | Valid weekly work hours range |

---

## 5. Missing Value Analysis

Missing values occur exclusively in three categorical variables:
- `workclass`: 2,799 total missing across dataset (5.73%)
- `occupation`: 2,809 total missing across dataset (5.75%)
- `native-country`: 857 total missing across dataset (1.75%)
- All 6 numeric features have **0 missing values**.

### Split Distribution of Missing Values:
| Feature | Raw (`adult.csv`) | Cleaned (`adult_cleaned.csv`) | Train (`train.csv`) | Validation (`validation.csv`) | Test (`test.csv`) | Split Sum |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `workclass` | 2,799 | 2,799 | 1,709 (5.83%) | 530 (5.42%) | 560 (5.73%) | **2,799** |
| `occupation` | 2,809 | 2,809 | 1,714 (5.85%) | 532 (5.45%) | 563 (5.76%) | **2,809** |
| `native-country`| 857 | 857 | 538 (1.84%) | 150 (1.54%) | 169 (1.73%) | **857** |
| **Total Nulls** | **6,465** | **6,465** | **3,961** | **1,212** | **1,292** | **6,465** |

*Imputation Note:* Missing values are retained as standard `NaN` in all CSVs. They are imputed during model training/inference via `SimpleImputer(strategy="most_frequent")` inside the pipeline's `ColumnTransformer`.

---

## 6. Duplicate Rows, Group Leakage, & Conflicting Records

### 6.1 Exact Duplicate Rows
- `data/raw/adult.csv`: **29 exact duplicates** (features + target match).
- `data/processed/adult_cleaned.csv`: **52 exact duplicates**.
  *Reason for Increase:* Normalizing `<=50K.` $\rightarrow$ `<=50K` united 23 pairs of rows that were identical except for the legacy trailing dot in the target label.
- Partition distribution:
  - `train.csv`: 26 exact duplicate rows
  - `validation.csv`: 12 exact duplicate rows
  - `test.csv`: 14 exact duplicate rows
  - Sum: $26 + 12 + 14 = 52$ exact duplicates.

### 6.2 Repeated Feature Vectors & Group Leakage Test
- Total rows with identical 14-feature combinations in full dataset: **111 rows** (57 duplicate occurrences).
- **Leakage Test:** We computed 64-bit feature vector hashes for every row in `train.csv`, `validation.csv`, and `test.csv`:
  - Overlap between `train.csv` and `validation.csv`: **0 rows**
  - Overlap between `train.csv` and `test.csv`: **0 rows**
  - Overlap between `validation.csv` and `test.csv`: **0 rows**
  *Finding:* The `GroupShuffleSplit` implemented in `scripts/07_make_grouped_split.py` successfully isolated duplicate feature profiles. No individual profile appears in more than one partition.

### 6.3 Conflicting Labels (Inherent Noise / Non-Determinism)
In `data/processed/adult_cleaned.csv`, there are **5 distinct feature combinations** where identical demographic profiles have opposing labels (one individual earns `<=50K` and another earns `>50K`).
- In `train.csv`: 4 conflicting combinations (8 rows).
- In `validation.csv`: 1 conflicting combination (2 rows).
- In `test.csv`: 0 conflicting combinations.
Because of grouped splitting, all rows belonging to a conflicting combination were assigned to the same partition, preventing conflicting labels from cross-pollinating train and test sets.

---

## 7. Categorical Vocabulary & Out-of-Vocabulary (OOV) Audit

To confirm model stability during inference, category sets across all 8 categorical features were audited for novel or unseen levels:

| Categorical Feature | Unique Categories in Train | Novel Categories in Val | Novel Categories in Test | Status |
| :--- | :--- | :--- | :--- | :--- |
| `workclass` | 8 | 0 | 0 | **100% Coverage** |
| `education` | 16 | 0 | 0 | **100% Coverage** |
| `marital-status` | 7 | 0 | 0 | **100% Coverage** |
| `occupation` | 14 | 0 | 0 | **100% Coverage** |
| `relationship` | 6 | 0 | 0 | **100% Coverage** |
| `race` | 5 | 0 | 0 | **100% Coverage** |
| `sex` | 2 | 0 | 0 | **100% Coverage** |
| `native-country` | 41 | 0 | 0 | **100% Coverage** |

*Verification:* All categorical levels present in validation and test sets are fully represented in `train.csv`. Furthermore, the pipeline's `OneHotEncoder` is configured with `handle_unknown="ignore"`, providing robust fallback for any novel category encountered in production.

---

## 8. Feature Redundancy, Leakage & Identifier Analysis

### 8.1 1:1 Bijection: `education` vs. `education-num`
An exact mapping analysis reveals that `education` and `education-num` are perfectly collinear:
```
Preschool     <==>  1        HS-grad       <==>  9
1st-4th       <==>  2        Some-college  <==> 10
5th-6th       <==>  3        Assoc-voc     <==> 11
7th-8th       <==>  4        Assoc-acdm    <==> 12
9th           <==>  5        Bachelors     <==> 13
10th          <==>  6        Masters       <==> 14
11th          <==>  7        Prof-school   <==> 15
12th          <==>  8        Doctorate     <==> 16
```
- **Current Pipeline Behavior:** In `src/incomelens/preprocessing.py`, `education-num` is passed through `StandardScaler` (1 numeric feature), while `education` is passed through `OneHotEncoder` (16 binary dummy features).
- **Impact:** Education is represented twice in the model (17 total features out of 105). While tree ensembles can tolerate redundant splits, for linear baselines this introduces multicollinearity.

### 8.2 Analysis of `fnlwgt` (Final Weight)
- `fnlwgt` is an administrative sampling weight computed by the US Census Bureau to indicate how many population units a given sample represents.
- It is not a direct personal identifier (e.g. SSN or person ID), but it has high cardinality (28,523 unique values among 48,842 rows).
- In tree models, unregularized splits on `fnlwgt` can lead to overfitting on survey artifacts rather than true income drivers. In our audit, `fnlwgt` ranked 7th in feature importance (6.23%).

### 8.3 Identifier & Direct Target Leakage Audit
- There are **no ID columns, transaction IDs, or row index columns** in any CSV.
- No feature directly correlates 1:1 with the target or exhibits post-event leakage.
- Features represent demographic and employment conditions measured concurrently with census income reporting.

---

## 9. Preprocessing State & Model Input Contract

### 9.1 Preprocessing State of CSVs
- **Missing Values:** Un-imputed (present as `NaN`).
- **Categoricals:** Un-encoded strings (`"Private"`, `"Bachelors"`, `"Married-civ-spouse"`).
- **Numerics:** Unscaled raw integers (`age = 39`, `hours-per-week = 40`).
- **Target:** Cleaned strings (`"<=50K"`, `">50K"`).

### 9.2 Production Model Input Expectation
An experimental inspection of `models/random_forest_final.joblib` confirmed:
- The model expects a **Pandas DataFrame with the 14 raw feature columns**.
- `model.predict(raw_test_df)` runs inference seamlessly.
- Passing a pre-transformed matrix (105 features) directly to `model.predict()` raises:
  ```text
  ValueError: X has 105 features, but ColumnTransformer is expecting 14 features as input.
  ```
- **Conclusion:** Any serving API or Streamlit interface must feed **raw user inputs** directly into `model.predict()`, without applying manual one-hot encoding or scaling beforehand.

---

## 10. Prioritized Data Findings & Recommendations

### Severity 1: Critical (Direct impact on model validity or serving)
1. **Raw Model Serving Contract:**
   - *Finding:* Application developers might assume the model expects numeric encodings.
   - *Action:* Document clearly in the model card and API that `models/random_forest_final.joblib` must receive raw demographic strings.

### Severity 2: High (Feature engineering & modeling efficiency)
2. **Deduplicate Education Features:**
   - *Finding:* `education` (16 one-hot columns) and `education-num` (1 continuous column) are 100% redundant.
   - *Action:* In future model iterations, experiment with keeping `education-num` as an ordinal integer and dropping the 16 one-hot dummy columns from `education` to reduce model size from 105 to 89 features.
3. **Audit `fnlwgt` Predictive Value:**
   - *Finding:* `fnlwgt` represents demographic sampling weight, not an intrinsic individual attribute.
   - *Action:* Evaluate model performance with and without `fnlwgt` to verify if it improves generalization or simply overfits census survey sampling blocks.

### Severity 3: Medium (Data cleaning & documentation)
4. **Formalize Label Conflict Protocol:**
   - *Finding:* 5 feature vectors have contradictory labels.
   - *Action:* While grouped splitting ensures no conflict crosses partitions, document this inherent data noise in `reports/model_card.md`.
5. **Data Ingestion Validation (Pydantic / Schema):**
   - *Finding:* Datasets rely on manual pandas parsing without a formal schema validator.
   - *Action:* Implement a lightweight validation schema in `src/incomelens/data.py` to enforce column presence, expected dtypes, and valid numerical bounds on ingest.
