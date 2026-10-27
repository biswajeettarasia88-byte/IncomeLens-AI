"""Inference service and validation contract for IncomeLens-AI.

This module provides a production-grade inference interface that:
- Loads the validated saved model without retraining.
- Validates input schemas, ranges, missing fields, and categories.
- Reuses the fitted preprocessing pipeline.
- Handles both complete Pipelines and standalone estimators paired with preprocessors.
- Maps predicted classes and probabilities safely using model.classes_.
- Provides actionable, sanitized error messages without exposing file paths or stack traces.
- Rejects the target column if supplied as an input feature.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_PATH = ROOT / "models" / "random_forest_final.joblib"

# Canonical feature names expected by the model
NUMERIC_FEATURES = [
    "age",
    "fnlwgt",
    "education-num",
    "capital-gain",
    "capital-loss",
    "hours-per-week",
]

CATEGORICAL_FEATURES = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]

EXPECTED_FEATURES = [
    "age",
    "workclass",
    "fnlwgt",
    "education",
    "education-num",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "capital-gain",
    "capital-loss",
    "hours-per-week",
    "native-country",
]

# Supported categorical options derived from fitted preprocessor
SUPPORTED_CATEGORIES: Dict[str, List[str]] = {
    "workclass": [
        "Federal-gov", "Local-gov", "Never-worked", "Private",
        "Self-emp-inc", "Self-emp-not-inc", "State-gov", "Without-pay",
    ],
    "education": [
        "10th", "11th", "12th", "1st-4th", "5th-6th", "7th-8th", "9th",
        "Assoc-acdm", "Assoc-voc", "Bachelors", "Doctorate", "HS-grad",
        "Masters", "Preschool", "Prof-school", "Some-college",
    ],
    "marital-status": [
        "Divorced", "Married-AF-spouse", "Married-civ-spouse",
        "Married-spouse-absent", "Never-married", "Separated", "Widowed",
    ],
    "occupation": [
        "Adm-clerical", "Armed-Forces", "Craft-repair", "Exec-managerial",
        "Farming-fishing", "Handlers-cleaners", "Machine-op-inspct",
        "Other-service", "Priv-house-serv", "Prof-specialty",
        "Protective-serv", "Sales", "Tech-support", "Transport-moving",
    ],
    "relationship": [
        "Husband", "Not-in-family", "Other-relative", "Own-child",
        "Unmarried", "Wife",
    ],
    "race": [
        "Amer-Indian-Eskimo", "Asian-Pac-Islander", "Black", "Other", "White",
    ],
    "sex": [
        "Female", "Male",
    ],
    "native-country": [
        "Cambodia", "Canada", "China", "Columbia", "Cuba",
        "Dominican-Republic", "Ecuador", "El-Salvador", "England",
        "France", "Germany", "Greece", "Guatemala", "Haiti",
        "Holand-Netherlands", "Honduras", "Hong", "Hungary", "India",
        "Iran", "Ireland", "Italy", "Jamaica", "Japan", "Laos",
        "Mexico", "Nicaragua", "Outlying-US(Guam-USVI-etc)", "Peru",
        "Philippines", "Poland", "Portugal", "Puerto-Rico", "Scotland",
        "South", "Taiwan", "Thailand", "Trinadad&Tobago", "United-States",
        "Vietnam", "Yugoslavia",
    ],
}

# Mapping of education credentials to education-num
EDUCATION_TO_NUM: Dict[str, int] = {
    "Preschool": 1,
    "1st-4th": 2,
    "5th-6th": 3,
    "7th-8th": 4,
    "9th": 5,
    "10th": 6,
    "11th": 7,
    "12th": 8,
    "HS-grad": 9,
    "Some-college": 10,
    "Assoc-voc": 11,
    "Assoc-acdm": 12,
    "Bachelors": 13,
    "Masters": 14,
    "Prof-school": 15,
    "Doctorate": 16,
}


class ValidationError(ValueError):
    """Raised when user inputs fail schema, type, range, or categorical validation."""
    pass


class InferenceError(RuntimeError):
    """Raised when model inference fails during processing."""
    pass


def load_model(
    model_path: Optional[Union[str, Path]] = None,
) -> Any:
    """Load the validated production model artifact without retraining.

    Parameters
    ----------
    model_path : str or Path, optional
        Path to the saved joblib model artifact. If None, uses DEFAULT_MODEL_PATH.

    Returns
    -------
    fitted scikit-learn Pipeline or Estimator.

    Raises
    ------
    InferenceError
        If the model file does not exist or fails to load.
    """
    path = Path(model_path) if model_path is not None else DEFAULT_MODEL_PATH
    if not path.is_file():
        raise InferenceError(
            f"Production model artifact not found. Please ensure the model file is available."
        )

    try:
        model = joblib.load(path)
        return model
    except Exception as e:
        raise InferenceError(
            "Failed to load the model artifact. The file may be corrupted or incompatible."
        ) from None


class IncomePredictor:
    """Production inference interface for IncomeLens-AI.

    Parameters
    ----------
    model : Pipeline or Estimator, optional
        Fitted model pipeline or estimator. If None, loads DEFAULT_MODEL_PATH.
    preprocessor : Transformer, optional
        Fitted preprocessor if `model` is a standalone estimator. If `model` is a
        Pipeline, this argument is ignored and the pipeline's internal preprocessor is reused.
    """

    def __init__(
        self,
        model: Optional[Any] = None,
        preprocessor: Optional[Any] = None,
    ):
        if model is None:
            self.model = load_model()
        else:
            self.model = model

        # Check whether model contains its own preprocessing pipeline
        if isinstance(self.model, Pipeline) and hasattr(self.model, "named_steps") and "preprocessor" in self.model.named_steps:
            self.is_pipeline = True
            self.preprocessor = self.model.named_steps["preprocessor"]
            self.classifier = self.model.named_steps["classifier"]
        elif isinstance(self.model, Pipeline) and len(self.model.steps) >= 2:
            self.is_pipeline = True
            self.preprocessor = self.model[:-1]
            self.classifier = self.model.steps[-1][1]
        else:
            # Standalone estimator: requires fitted preprocessor
            self.is_pipeline = False
            self.classifier = self.model
            if preprocessor is not None:
                self.preprocessor = preprocessor
            else:
                # Attempt to extract preprocessor from default production pipeline
                try:
                    default_pipe = load_model(DEFAULT_MODEL_PATH)
                    self.preprocessor = default_pipe.named_steps["preprocessor"]
                except Exception:
                    raise InferenceError(
                        "Model artifact is a standalone estimator but no fitted preprocessing "
                        "artifact was provided."
                    )

        # Inspect classes
        if hasattr(self.classifier, "classes_"):
            self.classes_ = self.classifier.classes_
        elif hasattr(self.model, "classes_"):
            self.classes_ = self.model.classes_
        else:
            self.classes_ = np.array([0, 1])

        # Check probability support
        self.supports_proba = hasattr(self.model, "predict_proba") or hasattr(self.classifier, "predict_proba")
        self._explainer = None

    def get_explainer(self) -> Any:
        """Lazily initialize and cache the TreeExplainer for the tree classifier."""
        if self._explainer is None:
            import shap
            # Initialize explainer on the underlying tree classifier
            self._explainer = shap.TreeExplainer(self.classifier)
        return self._explainer

    @staticmethod
    def _format_display_name(feature_name: str) -> str:
        """Format raw transformed feature name into human-readable label."""
        if feature_name.startswith("categorical__"):
            clean = feature_name[len("categorical__"):]
            if "_" in clean:
                col, val = clean.split("_", 1)
                return f"{col} = {val}"
            return clean
        elif feature_name.startswith("numeric__"):
            return feature_name[len("numeric__"):]
        return feature_name

    def explain_instance(
        self,
        X_clean: pd.DataFrame,
        top_k: int = 8,
    ) -> Optional[Dict[str, Any]]:
        """Compute local per-instance SHAP explanation for the high-income class (>50K).

        Parameters
        ----------
        X_clean : pd.DataFrame
            Single-row sanitized DataFrame matching the 14 raw survey features.
        top_k : int, default=8
            Maximum number of strongest feature contributions to return.

        Returns
        -------
        dict or None
            Dictionary containing base value, sum of contributions, reconstructed probability,
            and strongest positive/negative contributors. Returns None if explainer is unsupported.
        """
        # Transform raw features using fitted preprocessor to obtain 105 dense numeric features
        X_trans = self.preprocessor.transform(X_clean)
        if hasattr(X_trans, "toarray"):
            X_trans = X_trans.toarray()
        X_dense = np.asarray(X_trans, dtype=np.float64)

        if X_dense.shape[0] == 0:
            return None

        # Take first row
        row_dense = X_dense[:1]

        explainer = self.get_explainer()
        shap_raw = explainer.shap_values(row_dense)

        arr = np.asarray(shap_raw)
        if arr.ndim == 3:
            # SHAP format: (samples, features, classes). Index 1 is Class 1 (>50K)
            shap_vals = arr[0, :, 1]
        elif isinstance(shap_raw, list) and len(shap_raw) > 1:
            shap_vals = np.asarray(shap_raw[1])[0]
        else:
            shap_vals = arr[0]

        # Extract base value for Class 1 (>50K)
        expected_val = explainer.expected_value
        if isinstance(expected_val, (list, np.ndarray)) and len(expected_val) > 1:
            base_val = float(expected_val[1])
        else:
            base_val = float(expected_val)

        feature_names = list(self.preprocessor.get_feature_names_out())
        row_values = row_dense[0]

        items = []
        for name, s_val, f_val in zip(feature_names, shap_vals, row_values):
            if abs(s_val) < 1e-6:
                continue

            disp_name = self._format_display_name(name)
            is_pos = s_val > 0
            direction = "Positive (+)" if is_pos else "Negative (-)"
            impact_desc = (
                f"Increases probability of >$50K by +{abs(s_val)*100:.2f}%"
                if is_pos
                else f"Decreases probability of >$50K by -{abs(s_val)*100:.2f}%"
            )

            items.append({
                "feature": name,
                "display_name": disp_name,
                "feature_value": float(f_val),
                "shap_value": round(float(s_val), 4),
                "direction": direction,
                "impact_description": impact_desc,
            })

        # Sort by absolute impact
        items_by_abs = sorted(items, key=lambda x: abs(x["shap_value"]), reverse=True)
        top_contributions = items_by_abs[:top_k]

        # Separate positive and negative contributors
        top_positive = sorted(
            [x for x in items if x["shap_value"] > 0],
            key=lambda x: x["shap_value"],
            reverse=True,
        )[:top_k]

        top_negative = sorted(
            [x for x in items if x["shap_value"] < 0],
            key=lambda x: x["shap_value"],
        )[:top_k]

        sum_shap = float(np.sum(shap_vals))
        reconstructed = float(base_val + sum_shap)

        return {
            "target_class": ">50K",
            "base_value": round(base_val, 4),
            "sum_contributions": round(sum_shap, 4),
            "reconstructed_probability": round(reconstructed, 4),
            "top_contributions": top_contributions,
            "top_positive": top_positive,
            "top_negative": top_negative,
        }

    def validate_input(self, data: Union[Dict[str, Any], pd.DataFrame]) -> pd.DataFrame:
        """Validate input dictionary or DataFrame against feature schema, ranges, and categories.

        Parameters
        ----------
        data : dict or pd.DataFrame
            User inputs containing the 14 raw survey features.

        Returns
        -------
        pd.DataFrame
            Sanitized, properly typed DataFrame with canonical feature column ordering.

        Raises
        ------
        ValidationError
            With actionable message if missing fields, invalid types, out-of-range values,
            or unsupported categories are present, or if the target column is supplied.
        """
        # Convert dictionary to DataFrame
        if isinstance(data, dict):
            # Guard against target column leakage
            if "income" in data:
                raise ValidationError(
                    "Field 'income' is a prediction target and cannot be provided as an input feature."
                )
            df = pd.DataFrame([data])
        elif isinstance(data, pd.DataFrame):
            df = data.copy()
            if "income" in df.columns:
                raise ValidationError(
                    "Column 'income' is a prediction target and cannot be provided as an input feature."
                )
        else:
            raise ValidationError("Input data must be a dictionary or pandas DataFrame.")

        if len(df) == 0:
            raise ValidationError("Input contains zero rows to predict.")

        # Check for missing required features
        missing_cols = [c for c in EXPECTED_FEATURES if c not in df.columns]
        if missing_cols:
            raise ValidationError(
                f"Missing required input fields: {', '.join(missing_cols)}. "
                f"All 14 features are required."
            )

        # Validate numeric features
        for col in NUMERIC_FEATURES:
            # Check for NaN / null
            if df[col].isna().any():
                raise ValidationError(f"Field '{col}' cannot be null or empty.")

            # Attempt numeric conversion
            try:
                numeric_vals = pd.to_numeric(df[col], errors="raise")
                df[col] = numeric_vals
            except Exception:
                raise ValidationError(
                    f"Field '{col}' must be a valid numeric value, received non-numeric input."
                )

        # Range checks
        # age: [17, 100]
        if (df["age"] < 17).any() or (df["age"] > 100).any():
            raise ValidationError(
                f"Field 'age' must be between 17 and 100 years. Received: {df['age'].iloc[0]}"
            )

        # hours-per-week: [1, 99]
        if (df["hours-per-week"] < 1).any() or (df["hours-per-week"] > 99).any():
            raise ValidationError(
                f"Field 'hours-per-week' must be between 1 and 99 hours. Received: {df['hours-per-week'].iloc[0]}"
            )

        # capital-gain: >= 0
        if (df["capital-gain"] < 0).any():
            raise ValidationError("Field 'capital-gain' cannot be negative.")

        # capital-loss: >= 0
        if (df["capital-loss"] < 0).any():
            raise ValidationError("Field 'capital-loss' cannot be negative.")

        # fnlwgt: > 0
        if (df["fnlwgt"] <= 0).any():
            raise ValidationError("Field 'fnlwgt' (sampling weight) must be greater than zero.")

        # education-num: [1, 16]
        if (df["education-num"] < 1).any() or (df["education-num"] > 16).any():
            raise ValidationError("Field 'education-num' must be an integer between 1 and 16.")

        # Validate categorical features
        for col in CATEGORICAL_FEATURES:
            if df[col].isna().any():
                raise ValidationError(f"Field '{col}' cannot be null or empty.")

            # String conversion & strip whitespace
            df[col] = df[col].astype(str).str.strip()

            valid_options = set(SUPPORTED_CATEGORIES[col])
            unsupported = [val for val in df[col] if val not in valid_options]
            if unsupported:
                invalid_val = unsupported[0]
                sample_allowed = ", ".join(SUPPORTED_CATEGORIES[col][:5])
                raise ValidationError(
                    f"Unsupported category '{invalid_val}' for field '{col}'. "
                    f"Supported options include: {sample_allowed}..."
                )

        # Return DataFrame in exact canonical column order
        return df[EXPECTED_FEATURES]

    def predict(
        self,
        data: Union[Dict[str, Any], pd.DataFrame],
        explain: bool = False,
        top_k: int = 8,
    ) -> Dict[str, Any]:
        """Perform validated prediction on single input or batch.

        Parameters
        ----------
        data : dict or pd.DataFrame
            User inputs containing the 14 raw survey features.
        explain : bool, default=False
            Whether to compute local per-instance SHAP explanation.
        top_k : int, default=8
            Maximum number of strongest feature contributions if explain=True.

        Returns
        -------
        dict
            Structured prediction result containing:
            - 'prediction': '<=50K' or '>50K'
            - 'raw_prediction': 0 or 1
            - 'high_income_probability': float (or None if unsupported)
            - 'low_income_probability': float (or None if unsupported)
            - 'probabilities': dict of class probabilities
            - 'confidence_level': 'High', 'Moderate', or 'Low (Boundary Case)'
            - 'explanation': dict or None (local SHAP attribution details)

        Raises
        ------
        ValidationError
            If input fails schema validation.
        InferenceError
            If execution fails.
        """
        X_clean = self.validate_input(data)

        try:
            if self.is_pipeline:
                raw_pred = self.model.predict(X_clean)
                if self.supports_proba:
                    raw_probs = self.model.predict_proba(X_clean)
                else:
                    raw_probs = None
            else:
                X_trans = self.preprocessor.transform(X_clean)
                raw_pred = self.classifier.predict(X_trans)
                if self.supports_proba:
                    raw_probs = self.classifier.predict_proba(X_trans)
                else:
                    raw_probs = None
        except ValidationError:
            raise
        except Exception as e:
            # Sanitize error to prevent leaking file paths or internal stacks
            raise InferenceError(
                "An unexpected error occurred during prediction. Please verify input data formatting."
            ) from None

        # Process single prediction
        pred_val = raw_pred[0]
        # Map class label safely:
        # If classes are [0, 1], map 0 -> '<=50K', 1 -> '>50K'
        if pred_val in [1, ">50K", "1"]:
            predicted_label = ">50K"
            mapped_raw = 1
        else:
            predicted_label = "<=50K"
            mapped_raw = 0

        # Handle probabilities
        if raw_probs is not None and self.supports_proba:
            prob_row = raw_probs[0]
            # Identify which index corresponds to positive class
            classes_list = list(self.classes_)
            if ">50K" in classes_list:
                pos_idx = classes_list.index(">50K")
                neg_idx = 1 - pos_idx
            elif 1 in classes_list:
                pos_idx = classes_list.index(1)
                neg_idx = 1 - pos_idx
            else:
                pos_idx = 1
                neg_idx = 0

            p_high = float(prob_row[pos_idx])
            p_low = float(prob_row[neg_idx])
            probabilities = {
                "<=50K": round(p_low, 4),
                ">50K": round(p_high, 4),
            }

            # Confidence assessment
            # Close to 0.50 decision boundary indicates lower confidence
            margin = abs(p_high - 0.50)
            if margin >= 0.25:
                confidence_level = "High"
            elif margin >= 0.10:
                confidence_level = "Moderate"
            else:
                confidence_level = "Low (Boundary Case)"
        else:
            p_high = None
            p_low = None
            probabilities = None
            confidence_level = "Unknown (Model does not output probabilities)"

        # Local per-instance explanation handling (failsafe)
        explanation = None
        explanation_error = None
        if explain:
            try:
                explanation = self.explain_instance(X_clean, top_k=top_k)
            except Exception:
                # Crucial Requirement: Explanation failures must never break ordinary prediction
                explanation = None
                explanation_error = "Local feature explanation could not be computed for this instance."

        output = {
            "prediction": predicted_label,
            "raw_prediction": mapped_raw,
            "high_income_probability": round(p_high, 4) if p_high is not None else None,
            "low_income_probability": round(p_low, 4) if p_low is not None else None,
            "probabilities": probabilities,
            "confidence_level": confidence_level,
            "explanation": explanation,
        }
        if explanation_error is not None:
            output["explanation_error"] = explanation_error

        return output
