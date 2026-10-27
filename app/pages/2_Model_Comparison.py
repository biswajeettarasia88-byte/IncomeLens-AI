"""IncomeLens-AI - Model Comparison Page.

Visualizes validation split benchmarks across candidate models.
"""

from pathlib import Path
import sys
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
CSV_PATH = ROOT / "reports" / "validation_model_comparison.csv"

st.set_page_config(
    page_title="IncomeLens-AI | Model Comparison",
    page_icon="📊",
    layout="wide",
)

st.title("📊 Model Selection & Validation Benchmarks")
st.markdown(
    """
    Comparison of candidate model architectures evaluated on the held-out validation partition 
    ($N = 4,882$) using standardized grouped feature splits.
    """
)

if CSV_PATH.is_file():
    df = pd.read_csv(CSV_PATH)

    st.subheader("Candidate Model Comparison Table")
    st.dataframe(df, use_container_width=True)

    # Plot metric comparisons
    st.subheader("Validation Performance Comparison")
    metric_cols = [c for c in df.columns if c not in ["model", "Model"]]
    if metric_cols:
        selected_metric = st.selectbox("Select Metric to Compare", options=metric_cols, index=0)
        st.bar_chart(data=df.set_index(df.columns[0])[selected_metric])

    st.info(
        "**Selection Rationale:** Random Forest was selected as the final production model due to "
        "superior ROC-AUC and balanced accuracy on the validation split while mitigating overfitting."
    )
else:
    st.warning("Model comparison data not found at reports/validation_model_comparison.csv.")
