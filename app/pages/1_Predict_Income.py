"""IncomeLens-AI - Interactive Prediction Interface.

Allows users to input individual demographic, educational, and labor attributes
to generate validated income predictions, calibrated probabilities, and local SHAP explanations.
"""

from pathlib import Path
import sys
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from incomelens.inference import (
    IncomePredictor,
    SUPPORTED_CATEGORIES,
    EDUCATION_TO_NUM,
    ValidationError,
    InferenceError,
)

st.set_page_config(
    page_title="IncomeLens-AI | Predict Income",
    page_icon="🎯",
    layout="wide",
)

st.title("🎯 Individual Income Prediction & Local Explanation")
st.markdown(
    """
    Input individual demographic, educational, and labor market characteristics to evaluate
    the predicted income category (`<=50K` vs `>50K`) and inspect per-instance feature attributions (SHAP).
    """
)

# Brief Model Performance Context
with st.expander("ℹ️ Model Performance & Baseline Context (Click to expand)"):
    st.markdown(
        """
        - **Model:** Random Forest Classifier (150 trees, `min_samples_leaf=2`) with `ColumnTransformer`.
        - **Test Accuracy:** **86.83%** (8,484 / 9,771 correct).
        - **High-Income Precision:** **77.10%** (1,404 / 1,821 correct).
        - **High-Income Recall:** **61.74%** (*Model misses 38.26% of high-income earners*).
        - **Decision Threshold:** Standard uniform classification threshold $\\tau = 0.50$.
        """
    )


@st.cache_resource(show_spinner="Loading production model pipeline and explainability engine...")
def get_predictor():
    p = IncomePredictor()
    # Pre-initialize explainer in background to warm up cache
    try:
        p.get_explainer()
    except Exception:
        pass
    return p


try:
    predictor = get_predictor()
except Exception:
    st.error(
        "Model Loading Error: The production model artifact could not be loaded. "
        "Please ensure models/random_forest_final.joblib is available."
    )
    st.stop()

# Responsive Form Controls
with st.form("prediction_form"):
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Demographic & Household Information")
        age = st.slider(
            "Age (years)",
            min_value=17,
            max_value=90,
            value=38,
            step=1,
            help="Age of the individual in years (census survey range: 17 to 90).",
        )
        sex = st.selectbox(
            "Sex",
            options=SUPPORTED_CATEGORIES["sex"],
            index=1,
            help="Biological sex as recorded in the 1994 Census.",
        )
        race = st.selectbox(
            "Race",
            options=SUPPORTED_CATEGORIES["race"],
            index=4,  # White
            help="Racial background category as recorded in census survey.",
        )
        marital_status = st.selectbox(
            "Marital Status",
            options=SUPPORTED_CATEGORIES["marital-status"],
            index=2,  # Married-civ-spouse
            help="Current legal marital status.",
        )
        relationship = st.selectbox(
            "Household Relationship",
            options=SUPPORTED_CATEGORIES["relationship"],
            index=0,  # Husband
            help="Relationship role within the household unit.",
        )
        native_country = st.selectbox(
            "Native Country / Geographic Origin",
            options=SUPPORTED_CATEGORIES["native-country"],
            index=SUPPORTED_CATEGORIES["native-country"].index("United-States"),
            help="Country of birth or national origin.",
        )

    with col2:
        st.subheader("Education, Occupation & Financials")
        education = st.selectbox(
            "Highest Education Level",
            options=SUPPORTED_CATEGORIES["education"],
            index=SUPPORTED_CATEGORIES["education"].index("Bachelors"),
            help="Highest completed degree or educational attainment level.",
        )
        # Auto-synchronize education-num to the selected education credential
        default_edu_num = EDUCATION_TO_NUM.get(education, 10)
        education_num = st.number_input(
            "Education Years / Level Code",
            min_value=1,
            max_value=16,
            value=default_edu_num,
            step=1,
            help="Standard numeric code corresponding to highest educational credential (1: Preschool to 16: Doctorate).",
        )
        workclass = st.selectbox(
            "Workclass / Employment Sector",
            options=SUPPORTED_CATEGORIES["workclass"],
            index=3,  # Private
            help="Type of employment organization or self-employment status.",
        )
        occupation = st.selectbox(
            "Occupation Category",
            options=SUPPORTED_CATEGORIES["occupation"],
            index=3,  # Exec-managerial
            help="Primary professional or technical role category.",
        )
        hours_per_week = st.slider(
            "Hours Worked Per Week",
            min_value=1,
            max_value=99,
            value=40,
            step=1,
            help="Typical weekly working hours.",
        )
        capital_gain = st.number_input(
            "Annual Capital Gain ($ / year)",
            min_value=0.0,
            max_value=99999.0,
            value=0.0,
            step=500.0,
            help="Net capital gains from investments recorded in the survey year.",
        )
        capital_loss = st.number_input(
            "Annual Capital Loss ($ / year)",
            min_value=0.0,
            max_value=4356.0,
            value=0.0,
            step=100.0,
            help="Net capital losses from investments recorded in the survey year.",
        )

    with st.expander("Advanced Settings (Sampling Weight & Explanation Options)"):
        fnlwgt = st.number_input(
            "Final Sampling Weight (fnlwgt)",
            min_value=1.0,
            max_value=1500000.0,
            value=178356.0,
            step=1000.0,
            help="Census Bureau sampling demographic expansion weight. Defaults to the survey median.",
        )
        compute_explanation = st.checkbox(
            "Compute Local Per-Instance SHAP Explanation",
            value=True,
            help="Calculates exact mathematical contributions of this individual profile's features to the high-income probability.",
        )

    submit_button = st.form_submit_button(
        "Evaluate Prediction & Explanation", use_container_width=True
    )

# Prediction Pipeline Execution with Loading and Error States
if submit_button:
    input_payload = {
        "age": age,
        "workclass": workclass,
        "fnlwgt": fnlwgt,
        "education": education,
        "education-num": education_num,
        "marital-status": marital_status,
        "occupation": occupation,
        "relationship": relationship,
        "race": race,
        "sex": sex,
        "capital-gain": capital_gain,
        "capital-loss": capital_loss,
        "hours-per-week": hours_per_week,
        "native-country": native_country,
    }

    with st.spinner("Executing model inference and probability estimation..."):
        try:
            result = predictor.predict(input_payload, explain=compute_explanation)

            st.divider()
            st.subheader("Model Evaluation Result")

            pred_label = result["prediction"]
            p_high = result["high_income_probability"]
            p_low = result["low_income_probability"]

            res_col1, res_col2 = st.columns([1, 2])

            with res_col1:
                if pred_label == ">50K":
                    st.success("### Predicted Class: **> $50,000 / year**")
                    st.caption("Model evaluates annual income category as exceeding $50K threshold.")
                else:
                    st.info("### Predicted Class: **≤ $50,000 / year**")
                    st.caption("Model evaluates annual income category as within or below $50K threshold.")

                if p_high is not None:
                    st.metric("P(Income > $50K)", f"{p_high * 100:.1f}%")
                    st.caption("Standard Decision Threshold: $\\tau = 0.50$")

            with res_col2:
                if p_high is not None:
                    st.markdown("**Estimated Class Probability Distribution:**")
                    st.progress(p_high)
                    st.write(
                        f"- **P(Income > $50K):** `{p_high:.4f}` ({p_high * 100:.2f}%)\n"
                        f"- **P(Income ≤ $50K):** `{p_low:.4f}` ({p_low * 100:.2f}%)"
                    )

                    # Boundary Case Alert (no fake confidence scores, just threshold margin)
                    if 0.40 <= p_high <= 0.60:
                        st.warning(
                            "⚠️ **Decision Threshold Proximity Notice:** The estimated probability is close to "
                            "the 50% classification threshold ($0.40 \\le P \\le 0.60$). In high-stakes applications, "
                            "boundary cases must undergo manual review with verified financial documents."
                        )

            # Local SHAP Explanation Section (Per-Instance)
            explanation = result.get("explanation")
            if explanation is not None:
                st.divider()
                st.subheader("🔍 Local Per-Instance Explanation (SHAP Attributions)")
                st.markdown(
                    f"""
                    This breakdown explains the **specific prediction for this individual profile**, not a global average.
                    The model begins at the baseline population probability of **{explanation['base_value']*100:.2f}%** 
                    (training prevalence) and adds the individual feature contributions below to yield 
                    **{explanation['reconstructed_probability']*100:.2f}%**.
                    """
                )

                # Show Positive and Negative Contributors side by side
                pos_col, neg_col = st.columns(2)

                with pos_col:
                    st.markdown("##### 🟢 Top Factors Increasing High-Income Odds")
                    top_pos = explanation.get("top_positive", [])
                    if top_pos:
                        pos_df = pd.DataFrame([
                            {
                                "Feature": item["display_name"],
                                "Probability Shift": f"+{item['shap_value']*100:.2f}%",
                                "Raw Value": item["feature_value"],
                            }
                            for item in top_pos[:5]
                        ])
                        st.dataframe(pos_df, use_container_width=True, hide_index=True)
                    else:
                        st.caption("No significant positive contributors identified for this instance.")

                with neg_col:
                    st.markdown("##### 🔴 Top Factors Decreasing High-Income Odds")
                    top_neg = explanation.get("top_negative", [])
                    if top_neg:
                        neg_df = pd.DataFrame([
                            {
                                "Feature": item["display_name"],
                                "Probability Shift": f"{item['shap_value']*100:.2f}%",
                                "Raw Value": item["feature_value"],
                            }
                            for item in top_neg[:5]
                        ])
                        st.dataframe(neg_df, use_container_width=True, hide_index=True)
                    else:
                        st.caption("No significant negative contributors identified for this instance.")

                # Explanatory Causal Disclaimer
                st.info(
                    "💡 **Contribution vs. Causation Disclaimer:**\n"
                    "SHAP values reflect mathematical credit allocation under the decision trees for this individual profile. "
                    "They describe observational associations in 1994 Census data and **must not** be interpreted as causal proof "
                    "that intervening on a feature (e.g. changing marital status or hours) will causally change an individual's income."
                )

            elif result.get("explanation_error"):
                st.warning(f"ℹ️ {result['explanation_error']}")

            st.divider()
            st.caption(
                "**Operational & Ethical Notice:** Predictions represent statistical associations from 1994 Census survey records, "
                "not verified contemporary financial standing. For consequential decisions, human oversight is mandatory."
            )

        except ValidationError as e:
            st.error(f"Input Validation Error: {e}")
        except InferenceError as e:
            st.error(f"Inference Processing Error: {e}")
        except Exception:
            st.error(
                "An unexpected system error occurred while generating the prediction. "
                "Please verify that all fields contain valid inputs."
            )
