# IncomeLens-AI Problem Statement & Technical Specification

## 1. Objective
IncomeLens-AI addresses binary classification of personal income status based on 1994 US Census Bureau survey records. The technical goal is to evaluate model performance, interpretability, and group-level fairness under a rigorous, reproducible ML engineering workflow.

## 2. Prediction Target
- **Target Variable:** `income`
- **Class 0:** `<=50K` (prevalence: $75.91\%$ in development data)
- **Class 1:** `>50K` (prevalence: $24.09\%$ in development data)
- **Evaluation Priority:** In addition to overall accuracy and ROC-AUC, evaluation tracks balanced accuracy, precision, and recall on the positive minority class (`>50K`).

## 3. Input Features & Schema
14 raw survey attributes:
- **Numeric (6):** `age`, `fnlwgt`, `education-num`, `capital-gain`, `capital-loss`, `hours-per-week`.
- **Categorical (8):** `workclass`, `education`, `marital-status`, `occupation`, `relationship`, `race`, `sex`, `native-country`.

## 4. Methodological Constraints & Guardrails
1. **Grouped Split Rationale:** Because `fnlwgt` (final sampling weight) groups identical demographic sampling strata, records sharing an `fnlwgt` value are partitioned together to prevent data leakage between training, validation, and test splits.
2. **Train-Only Preprocessing:** All imputation and scaling operations are fitted strictly on the training set and applied out-of-sample to validation and test data.
3. **No Test-Set Tuning:** Hyperparameters and model architecture selections are conducted using the validation split; the final test split is reserved exclusively for unadjusted baseline reporting.
4. **Non-Causal Interpretability:** Model attributions (MDI and SHAP) are evaluated as observational statistical associations conditional on model structure, not causal interventions.
5. **Fairness Diagnostics:** Group-level disparities are evaluated across sensitive attributes (`sex`, `race`, `age`, `native-country`) with explicit reference groups and uncertainty reporting.
