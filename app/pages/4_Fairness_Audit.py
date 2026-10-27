"""IncomeLens-AI - Fairness & Demographic Disparity Page.

Displays quantitative group fairness metrics and disparate impact ratios across cohorts.
"""

from pathlib import Path
import sys
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
FAIRNESS_CSV = ROOT / "reports" / "fairness_metrics.csv"

st.set_page_config(
    page_title="IncomeLens-AI | Fairness Audit",
    page_icon="⚖️",
    layout="wide",
)

st.title("⚖️ Algorithmic Fairness & Demographic Disparity Audit")
st.markdown(
    """
    Auditing selection rates, error rates, and disparate impact ratios of the production model 
    on the held-out test split ($N = 9,771$) across protected demographic attributes.
    """
)

if FAIRNESS_CSV.is_file():
    df = pd.read_csv(FAIRNESS_CSV)

    attributes = df["sensitive_attribute"].unique()
    selected_attr = st.selectbox("Select Sensitive Demographic Attribute", options=attributes)

    attr_df = df[df["sensitive_attribute"] == selected_attr].copy()

    st.subheader(f"Group Performance Metrics: {selected_attr.upper()}")
    display_cols = [
        "subgroup", "sample_count", "base_rate", "selection_rate",
        "true_positive_rate", "false_positive_rate", "precision",
        "disparate_impact_ratio", "equal_opportunity_diff", "unstable_estimate_flag"
    ]
    st.dataframe(attr_df[display_cols].set_index("subgroup"), use_container_width=True)

    # Disparate Impact Ratio Chart
    st.subheader("Disparate Impact Ratio vs. 80% Four-Fifths Guideline")
    dir_data = attr_df.set_index("subgroup")["disparate_impact_ratio"]
    st.bar_chart(dir_data)

    # Small sample uncertainty flags
    unstable_rows = attr_df[attr_df["unstable_estimate_flag"] == True]
    if not unstable_rows.empty:
        for _, urow in unstable_rows.iterrows():
            st.warning(
                f"⚠️ **Small Sample Alert for '{urow['subgroup']}':** {urow['unstable_estimate_note']}. "
                f"Wide confidence intervals preclude confident conclusions for this group."
            )

    st.divider()
    st.markdown(
        """
        **Ethical & Governance Disclaimers:**
        - **Diagnostic Only:** Parity metrics are quantitative diagnostics; meeting thresholds does **not** establish legal compliance under Title VII or ECOA.
        - **Proxy Discrimination:** Omitting sensitive columns does not prevent proxy discrimination due to correlated features (e.g. `relationship_Husband`, `occupation`).
        - **Model Preservation:** The production model was not altered based on test set audit results.
        """
    )
else:
    st.warning("Fairness metrics data not found at reports/fairness_metrics.csv.")
