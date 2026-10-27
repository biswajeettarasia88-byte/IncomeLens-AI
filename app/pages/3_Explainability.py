"""IncomeLens-AI - Explainability & Interpretability Page.

Visualizes Gini feature importance (MDI) and game-theoretic SHAP attributions.
"""

from pathlib import Path
import sys
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
MDI_CSV = ROOT / "reports" / "feature_importance.csv"
MDI_PNG = ROOT / "reports" / "figures" / "feature_importance.png"
SHAP_PNG = ROOT / "reports" / "figures" / "shap_summary.png"

st.set_page_config(
    page_title="IncomeLens-AI | Explainability",
    page_icon="🧠",
    layout="wide",
)

st.title("🧠 Explainability & Feature Attribution Dashboard")
st.markdown(
    """
    Examine the internal drivers of the Random Forest model using both global split metrics 
    (Mean Decrease in Impurity) and local game-theoretic attributions (SHAP).
    """
)

tab1, tab2 = st.tabs(["Tree Split Importance (MDI)", "SHAP Summary (Probability Space)"])

with tab1:
    st.subheader("Global Feature Importance (Mean Decrease in Impurity)")
    col1, col2 = st.columns([3, 2])

    with col1:
        if MDI_PNG.is_file():
            st.image(str(MDI_PNG), caption="Top 20 Features by Gini Impurity Reduction")
        else:
            st.warning("Feature importance plot not found.")

    with col2:
        st.markdown(
            """
            **Methodological Disclaimers:**
            - **In-Sample Metric:** MDI measures total Gini impurity reduction across training tree splits.
            - **Cardinality Bias:** High-cardinality continuous features (e.g. `fnlwgt`) receive inflated importance.
            - **Collinear Dilution:** Redundant features (`marital-status_Married-civ-spouse` vs `relationship_Husband`) 
              compete and dilute individual importance.
            - **Non-Causal:** MDI reflects model split frequency, not real-world causal levers.
            """
        )
        if MDI_CSV.is_file():
            mdi_df = pd.read_csv(MDI_CSV)
            st.dataframe(mdi_df.head(15), use_container_width=True)

with tab2:
    st.subheader("SHAP Beeswarm Summary Plot (Class Probability Space)")
    st.markdown(
        """
        Because the Random Forest classifier computes arithmetic means over leaf class probabilities,
        TreeExplainer outputs additive attributions directly in **class probability space** ($[0, 1]$).
        """
    )
    if SHAP_PNG.is_file():
        st.image(str(SHAP_PNG), caption="SHAP Summary Plot for Class 1 (>50K) - 1,000 Sample Batch")
    else:
        st.warning("SHAP summary plot not found.")

    st.info(
        "**Reading the Plot:** Each dot represents an individual. Horizontal position indicates probability impact "
        "(positive or negative shift in odds of earning >$50K); color indicates feature magnitude (red = high, blue = low)."
    )
