from __future__ import annotations
from pathlib import Path
import pandas as pd
import streamlit as st

from src.AutoInsight.charts import chart_candidates
from src.AutoInsight.ai_narrative import build_ai_prompt, deterministic_ai_summary
from src.AutoInsight.orchestrator import run_analysis
from src.AutoInsight.profiling import profile_frame
from src.AutoInsight.reporting import build_markdown_report
from src.AutoInsight.validation import DatasetValidationError


st.set_page_config(
    page_title="AutoInsight | AI-Powered Data Analysis",
    page_icon="",
    layout="wide",
)

st.markdown("""
<style>

/* ---------- Header ---------- */

.main-title {
    font-size: 3.8rem;
    font-weight: 800;
    margin-bottom: 0.2rem;
}

.subtitle {
    font-size: 1.5rem;
    font-weight: 600;
    color: #BDBDBD;
    margin-bottom: 0.35rem;
}

.description {
    font-size: 1.2rem;
    color: #A8A8A8;
    margin-bottom: 2rem;
}

/* ---------- Labels ---------- */

label {
    font-size: 1.1rem !important;
    font-weight: 600 !important;
}

/* ---------- Markdown Headings ---------- */

h1 {font-size: 3rem !important;}
h2 {font-size: 2.2rem !important;}
h3 {font-size: 1.7rem !important;}

/* ---------- Text Area ---------- */

textarea {
    font-size: 1.05rem !important;
}

/* ---------- Select Box ---------- */

.stSelectbox div[data-baseweb="select"] {
    font-size: 1.05rem !important;
}

/* ---------- Buttons ---------- */

.stButton button,
.stDownloadButton button {
    font-size: 1rem !important;
    font-weight: 600 !important;
}

/* ---------- Tabs ---------- */

button[data-baseweb="tab"] {
    font-size: 1.05rem !important;
    font-weight: 600 !important;
}

/* ---------- Metrics ---------- */

[data-testid="stMetricValue"] {
    font-size: 2rem !important;
}

[data-testid="stMetricLabel"] {
    font-size: 1rem !important;
}
/* Widget labels (Dataset, Business Question, Target column, etc.) */
.stFileUploader label,
.stTextArea label,
.stSelectbox label {
    font-size: 1.25rem !important;
    font-weight: 700 !important;
}

</style>

<div class="main-title">AutoInsight : AI-Powered Data Analysis Platform</div>

<div class="subtitle">
Developed by Hamza Ikram
</div>

<div class="description">
Upload any CSV and automatically generate data profiling, visualizations,
machine learning models, and an AI-powered report.
</div>

""", unsafe_allow_html=True)


uploaded_file = st.file_uploader("Dataset", type=["csv"])

sample_path = Path("example dataset") / "Kaggle-Telco-Customer-Churn.csv"

with open(sample_path, "rb") as f:
    st.download_button(
        label="Download Sample Dataset",
        data=f,
        file_name="Kaggle-Telco-Customer-Churn.csv",
        mime="text/csv",
    )

business_question = st.text_area(
    "Business Question",
     placeholder="Describe the business problem you want AutoInsight to solve...",
    height=100,
)

st.markdown("##### Need inspiration? Choose an example business question")

example_questions = [
    "Select an example business question...",
    "Which factors have the strongest impact on the target variable?",
    "What patterns or trends can be identified in this dataset?",
    "Which records are most likely to have the predicted outcome?",
    "Are there any unusual values or anomalies in the data?",
    "What business insights and recommendations can be derived from this dataset?"
]

selected_question = st.selectbox(
    "Example Business Questions",
    example_questions,
)
if selected_question != example_questions[0]:
    business_question = selected_question


if uploaded_file is None:
    st.info("Upload a CSV file to begin.")
    st.stop()

df = pd.read_csv(uploaded_file)

target_options = ["Auto infer"] + [str(column) for column in df.columns]
target_choice = st.selectbox("Target column", target_options)
target_override = None if target_choice == "Auto infer" else target_choice.strip().lower()

run_analysis_btn = st.button("Run Analysis")

if not run_analysis_btn:
    st.info("Configure your options and click **Run Analysis**.")
    st.stop()

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
