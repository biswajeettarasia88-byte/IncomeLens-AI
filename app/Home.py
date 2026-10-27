"""IncomeLens-AI Streamlit Web Application - Home Page.

Comprehensive overview of IncomeLens-AI: problem formulation, verified benchmark metrics,
intended use guidelines, and multi-page application navigation.
"""

from pathlib import Path
import sys
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

st.set_page_config(
    page_title="IncomeLens-AI | Transparent Income Classification & Auditing",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🔍 IncomeLens-AI: Transparent Income Classification & Auditing")
st.markdown(
    """
    **IncomeLens-AI** is a machine learning benchmark and auditing application evaluated on the 
    1994 US Census Bureau Adult database ($N = 48,842$). It demonstrates tabular classification, 
    feature interpretability, and algorithmic fairness under a rigorous, reproducible engineering workflow.
    """
)

# Verified Model Performance Section
st.subheader("📈 Verified Production Model Performance")
st.caption(
    "Evaluated on the held-out test split (N = 9,771) using the validated production "
    "Random Forest Pipeline (150 trees, min_samples_leaf=2) without test-set tuning."
)

col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    st.metric("Test Accuracy", "86.83%", help="Overall correct predictions: 8,484 / 9,771")
with col2:
    st.metric("Balanced Accuracy", "78.09%", help="Average of class 0 recall (94.4%) and class 1 recall (61.7%)")
with col3:
    st.metric("High-Income Precision", "77.10%", help="When predicting >50K, correct 77.1% of the time (1,404 / 1,821)")
with col4:
    st.metric("High-Income Recall", "61.74%", help="CRITICAL LIMITATION: Misses 38.26% of actual high earners (870 false negatives)")
with col5:
    st.metric("ROC-AUC", "0.9181", help="Area under the Receiver Operating Characteristic curve")

# Key Limitation Callout
st.warning(
    "⚠️ **Key Performance Limitation — Recall Deficit:** While overall accuracy is high (86.8%), "
    "the model's high-income recall is **61.74%**, meaning it fails to detect nearly **4 out of 10 actual high earners** "
    "(870 false negatives out of 2,274 high earners). The model is conservative and biased toward the majority "
    "negative class (<=50K, 75.9% prevalence). It must **not** be used for automated eligibility gating."
)

st.divider()

# Navigation Cards
st.subheader("Explore the Application Modules")
c1, c2 = st.columns(2)

with c1:
    with st.container(border=True):
        st.markdown("### 🎯 [1. Predict Income](Predict_Income)")
        st.markdown(
            "Input demographic, education, and labor characteristics to generate verified income predictions, "
            "calibrated probability distributions, and decision threshold boundary assessments."
        )

    with st.container(border=True):
        st.markdown("### 📊 [2. Model Comparison](Model_Comparison)")
        st.markdown(
            "Review validation benchmarks across candidate models (Logistic Regression, Decision Trees, "
            "Random Forest, Gradient Boosting) evaluated on the held-out validation partition ($N = 4,882$)."
        )

with c2:
    with st.container(border=True):
        st.markdown("### 🧠 [3. Explainability](Explainability)")
        st.markdown(
            "Inspect global Gini feature importance (Mean Decrease in Impurity) alongside game-theoretic "
            "SHAP (Shapley Additive exPlanations) operating in exact probability space."
        )

    with st.container(border=True):
        st.markdown("### ⚖️ [4. Fairness Audit](Fairness_Audit)")
        st.markdown(
            "Audit group-level selection rates, disparate impact ratios, true-positive rates, and 95% Wilson "
            "confidence intervals across gender, racial, age, and geographic cohorts."
        )

st.divider()

# Intended Use & Governance Framework
st.subheader("🛡️ Intended Use, Scope & Governance Framework")

use_col1, use_col2 = st.columns(2)

with use_col1:
    st.markdown("#### ✅ Supported & Intended Uses")
    st.markdown(
        """
        - **Educational & Research Benchmarking:** Evaluating tabular ML algorithms, grouped feature splitting, and hyperparameter validation.
        - **Fairness & Interpretability Research:** Auditing disparities, error rate distributions, and proxy discrimination mechanisms.
        - **Decision Support:** Secondary exploratory tool used by trained human analysts with access to verified primary financial records.
        """
    )

with use_col2:
    st.markdown("#### 🚫 Prohibited & Out-of-Scope Uses")
    st.markdown(
        """
        - **Automated Financial Underwriting:** Credit scoring, loan approvals, or insurance pricing.
        - **Employment Gating:** Candidate screening, hiring, promotion, or termination decisions.
        - **Housing Eligibility:** Tenancy, mortgage, or rental qualification.
        - **Autonomous Decision-Making:** Any automated process affecting legal, financial, or employment rights without human appeal.
        """
    )

st.info(
    "📖 **Documentation & Transparency References:**\n"
    "- Complete Model Card: `reports/model_card.md`\n"
    "- Dataset Audit Card: `docs/dataset_card.md`\n"
    "- Fairness Audit Report: `reports/fairness_report.md`\n"
    "- Explainability Report: `reports/explainability_report.md`"
)
