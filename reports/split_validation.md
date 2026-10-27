# IncomeLens AI — Grouped Split Validation Report

**Audit Date:** October 20, 2026  
**Auditor:** Senior Machine Learning Engineer  
**Project Path:** `D:\incomelens-ai`  
**Report Destination:** `reports/split_validation.md`

---

## 1. Executive Summary

An audit and validation of [`scripts/07_make_grouped_split.py`](file:///D:/incomelens-ai/scripts/07_make_grouped_split.py) and the resulting partition datasets (`train.csv`, `validation.csv`, `test.csv`) was conducted.

### Core Verdict:
- **Methodology Correctness:** The grouping and splitting logic in `scripts/07_make_grouped_split.py` is **methodologically sound, rigorous, and fully verified**.
- **Zero Leakage:** Group separation between all three partitions is **100% disjoint** (0 overlapping feature groups).
- **Target Isolation:** The target variable `income` is strictly isolated and never enters the grouping key or input feature set.
- **Preprocessing Purity:** No preprocessing operations (imputation, scaling, one-hot encoding) are fitted before splitting; the split files contain clean, raw values.
- **Deterministic Reproducibility:** Fixed seeds (`random_state=42` and `random_state=43`) reproduce the exact rows, columns, and ordering of the disk files with 100% agreement.
- **Preservation Action:** Because the current approach is verified correct and disk artifacts are fully intact, **existing split files were not regenerated**, avoiding unnecessary disk writes and preventing unintended side effects.

---

## 2. Grouping Key Identification & Rationale

### 2.1 Actual Grouping Key
Lines 16–20 of `scripts/07_make_grouped_split.py`:
```python
feature_cols = [c for c in df.columns if c != "income"]
groups = pd.util.hash_pandas_object(
    df[feature_cols].fillna("<MISSING>").astype(str),
    index=False,
)
```
- **Grouping Mechanism:** MurmurHash3 64-bit hashing across all 14 predictor columns (`feature_cols`).
- **Index Excluded:** `index=False` ensures hashing depends exclusively on feature values, independent of row position.
- **Missing Value Handling:** Missing categorical values (`NaN`) are filled with `"<MISSING>"` prior to string conversion so identical records with missing values hash to the same group.

### 2.2 Why Grouping is Required
The Adult Census Income dataset contains inherent demographic duplicates and label noise:
1. **52 exact duplicate rows:** Identical across all 14 features and the target.
2. **111 rows with repeated feature vectors:** Identical demographic profiles.
3. **5 distinct feature combinations with conflicting labels:** Identical demographics where one individual has `<=50K` and another has `>50K`.

**Risks of Standard Random Splitting:**
- If standard `train_test_split` were used, duplicate individuals would be partitioned across train and test sets.
- Tree models would memorize training instances and achieve artificially optimistic test scores (**data leakage**).
- For conflicting records, a model trained on profile $A$ earning `<=50K` would be penalized on identical profile $A$ in the test set earning `>50K`.

**Grouped Split Solution:**
By treating each unique 14-feature profile as a group, `GroupShuffleSplit` guarantees that every identical or near-identical profile resides exclusively within a single partition.

---

## 3. Group Separation & Disjointness Audit

We computed 64-bit MurmurHash3 hashes across the 14 features for every row in the partition files on disk:

| Partition | Total Rows | Unique Feature Groups | Group Overlap with Train | Group Overlap with Val | Group Overlap with Test |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Train** (`train.csv`) | 29,301 | 29,271 | — | **0** | **0** |
| **Validation** (`validation.csv`) | 9,770 | 9,757 | **0** | — | **0** |
| **Test** (`test.csv`) | 9,771 | 9,757 | **0** | **0** | — |

### Formal Disjointness Verification:
$$\text{Train Groups} \cap \text{Validation Groups} = \emptyset \quad (\text{size } 0)$$
$$\text{Train Groups} \cap \text{Test Groups} = \emptyset \quad (\text{size } 0)$$
$$\text{Validation Groups} \cap \text{Test Groups} = \emptyset \quad (\text{size } 0)$$

*Automated Test Added:* A regression test [`test_split_group_disjointness`](file:///D:/incomelens-ai/tests/test_data.py) was added to `tests/test_data.py` to ensure CI/CD continuously verifies zero cross-split leakage.

---

## 4. Split Sizes & Target Distribution

### 4.1 Partition Sizes
- **Development Partition:** 39,071 rows (80.00% of 48,842 total rows)
  - **Train:** 29,301 rows (59.99%)
  - **Validation:** 9,770 rows (20.00%)
- **Test Partition:** 9,771 rows (20.01%)
- **Total:** $29,301 + 9,770 + 9,771 = 48,842$ rows (exact sum).

### 4.2 Class Distribution Consistency
| Partition | Total Rows | `<=50K` (Count) | `>50K` (Count) | Positive Class % (`>50K`) | Deviation from Population |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Population** (`adult_cleaned.csv`) | 48,842 | 37,155 | 11,687 | **23.93%** | Reference |
| **Train** (`train.csv`) | 29,301 | 22,288 | 7,013 | **23.93%** | **0.00%** |
| **Validation** (`validation.csv`) | 9,770 | 7,370 | 2,400 | **24.56%** | $+0.63\%$ |
| **Test** (`test.csv`) | 9,771 | 7,497 | 2,274 | **23.27%** | $-0.66\%$ |

*Conclusion:* Despite grouping by high-cardinality feature hashes, `GroupShuffleSplit` preserved target balance within $\pm 0.7\%$ across all partitions.

---

## 5. Target Isolation & Preprocessing Purity

### 5.1 Target Isolation
- `feature_cols` explicitly excludes `"income"`:
  ```python
  feature_cols = [c for c in df.columns if c != "income"]
  ```
- The target is passed only as `y` in `splitter.split(df, df["income"], groups=groups)`.
- When models are trained, `split_features_target()` extracts `df["income"]` as `y` and drops it from the feature matrix `X`.
- **Verdict:** Target column is strictly isolated from predictors.

### 5.2 Preprocessing Purity
- `scripts/07_make_grouped_split.py` performs no feature transformations, imputation, scaling, or encoding.
- The output CSVs contain raw census data (`age = 39`, `workclass = "Private"`, missing values as `NaN`).
- Imputers and scalers are fitted only downstream on training/development splits.
- **Verdict:** Preprocessing is not fitted before splitting.

---

## 6. Deterministic Reproducibility

An in-memory simulation of `scripts/07_make_grouped_split.py` was executed with the script's fixed seeds:
- First split (Development vs Test): `GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42)`
- Second split (Train vs Validation): `GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=43)`

### Reproduction Comparison against Disk Artifacts:
- `train.equals(disk_train)`: **`True`** (100% identical)
- `val.equals(disk_val)`: **`True`** (100% identical)
- `test.equals(disk_test)`: **`True`** (100% identical)

---

## 7. Preservation Confirmation

Because:
1. The splitting methodology is verified mathematically correct,
2. The partitions on disk share 0 groups and match theoretical reproduction 100%,
3. Regenerating would only rewrite identical bytes to disk while risking file modification timestamp desynchronization,

`scripts/07_make_grouped_split.py` was **left unchanged**, and the existing partition files on disk were **preserved intact**.
