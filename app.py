from __future__ import annotations

import pandas as pd
import streamlit as st

from src.AutoInsight.charts import chart_candidates
from src.AutoInsight.ai_narrative import build_ai_prompt, deterministic_ai_summary
from src.AutoInsight.orchestrator import run_analysis
from src.AutoInsight.profiling import profile_frame
from src.AutoInsight.reporting import build_markdown_report
from src.AutoInsight.validation import DatasetValidationError


st.set_page_config(
    page_title="AutoInsight",
    page_icon="AI",
    layout="wide",
)

st.title("AutoInsight")
st.caption("Upload a CSV, ask a business question, and get profiling, modeling, charts, and a report.")

uploaded_file = st.file_uploader("Dataset", type=["csv"])
business_question = st.text_area(
    "Business question",
    placeholder="Example: Which customers are most likely to churn, and what patterns explain it?",
    height=90,
)

if uploaded_file is None:
    st.info("Upload a CSV file to begin.")
    st.stop()

df = pd.read_csv(uploaded_file)
target_options = ["Auto infer"] + [str(column) for column in df.columns]
target_choice = st.selectbox("Target column", target_options)
target_override = None if target_choice == "Auto infer" else target_choice

with st.spinner("Running autonomous analysis..."):
    try:
        cleaned_df, result = run_analysis(df, business_question, target_override)
        report = build_markdown_report(result, business_question)
    except DatasetValidationError as exc:
        st.error(str(exc))
        st.stop()

overview_cols = st.columns(4)
overview_cols[0].metric("Rows", f"{result.profile.rows:,}")
overview_cols[1].metric("Columns", f"{result.profile.columns:,}")
overview_cols[2].metric("Duplicates", f"{result.profile.duplicate_rows:,}")
overview_cols[3].metric("Memory", f"{result.profile.memory_mb:.2f} MB")

st.subheader("Inferred Setup")
setup_cols = st.columns(2)
setup_cols[0].write(f"**Target:** {result.target or 'None'}")
setup_cols[1].write(f"**Task:** {result.task_type or 'None'}")

tab_profile, tab_models, tab_charts, tab_report, tab_data = st.tabs(
    ["Profile", "Models", "Charts", "Report", "Clean Data"]
)

with tab_profile:
    st.dataframe(profile_frame(result.profile), use_container_width=True)

with tab_models:
    if result.model_results:
        model_rows = [
            {
                "model": model.name,
                "task": model.task_type,
                model.score_name: round(model.score, 4),
                **{name: round(value, 4) for name, value in model.secondary_metrics.items()},
            }
            for model in result.model_results
        ]
        st.dataframe(pd.DataFrame(model_rows), use_container_width=True)
    else:
        st.warning("No model was trained. Try selecting a target column with enough non-empty labels.")

with tab_charts:
    for title, figure in chart_candidates(cleaned_df):
        st.plotly_chart(figure, use_container_width=True)

with tab_report:
    st.markdown("### AI Summary")
    st.write(deterministic_ai_summary(result, business_question))
    st.markdown(report)
    with st.expander("LLM prompt for connected AI provider"):
        st.code(build_ai_prompt(result, business_question), language="markdown")
    st.download_button(
        "Download report",
        report,
        file_name="AutoInsight_report.md",
        mime="text/markdown",
    )

with tab_data:
    st.dataframe(cleaned_df.head(500), use_container_width=True)
    st.download_button(
        "Download cleaned CSV",
        cleaned_df.to_csv(index=False),
        file_name="cleaned_dataset.csv",
        mime="text/csv",
    )
