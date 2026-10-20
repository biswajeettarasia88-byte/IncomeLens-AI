
# IncomeLens AI — Dataset Card

## Dataset
Adult Census Income dataset.

## Project objective
Predict whether an individual's recorded annual income
falls into the <=50K or >50K category using census features.

## Dataset size
- Total rows: 48,842
- Input features: 14, before feature selection
- Target: income
- Classes: <=50K and >50K

## Data-quality audit
- Exact duplicate rows: 52
- Rows with repeated feature values: 111
- Feature combinations with conflicting income labels: 5
- Missing values exist in workclass, occupation,
  and native-country.

## Current handling decisions
- Original raw data is preserved.
- Income labels are normalized to two classes.
- Missing values are imputed inside the model pipeline.
- Duplicate records have not been deleted.
- Conflicting labels require further investigation.

## Evaluation limitations
Random train/validation/test splitting may place
duplicate records in different partitions.
This can produce optimistic performance estimates.

## Intended use
Educational machine-learning experimentation and
demonstration of classification, explainability,
and fairness evaluation.

## Limitations
This dataset is based on US census records and does
not represent every population or current economic
conditions. Predictions are estimates, not verified
income measurements.
