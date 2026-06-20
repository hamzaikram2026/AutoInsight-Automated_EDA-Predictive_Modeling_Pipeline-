from __future__ import annotations

from .schema import AnalysisResult


def build_ai_prompt(result: AnalysisResult, business_question: str) -> str:
    """Create a provider-agnostic prompt for an optional LLM narrative layer."""
    best_model = result.model_results[0].name if result.model_results else "No trained model"
    return f"""
You are an AI data scientist writing for business stakeholders.

Business question:
{business_question or "No question provided"}

Dataset:
- Rows: {result.profile.rows}
- Columns: {result.profile.columns}
- Target: {result.target}
- Task: {result.task_type}
- Best model: {best_model}

Write a concise report with:
1. Executive answer
2. Evidence from the dataset
3. Model quality and limitations
4. Recommended next actions
""".strip()


def deterministic_ai_summary(result: AnalysisResult, business_question: str) -> str:
    if result.model_results:
        best = result.model_results[0]
        model_sentence = (
            f"The strongest baseline is {best.name} with {best.score_name} of {best.score:.3f}."
        )
    else:
        model_sentence = "The app could not train a reliable baseline model from the available data."

    return (
        f"For the question '{business_question or 'not provided'}', the dataset contains "
        f"{result.profile.rows:,} rows and {result.profile.columns:,} columns. "
        f"The inferred target is {result.target or 'not selected'} and the task is "
        f"{result.task_type or 'not inferred'}. {model_sentence}"
    )
