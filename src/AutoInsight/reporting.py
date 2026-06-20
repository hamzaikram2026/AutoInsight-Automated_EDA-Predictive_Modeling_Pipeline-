from __future__ import annotations

from .schema import AnalysisResult


def build_markdown_report(result: AnalysisResult, business_question: str) -> str:
    profile = result.profile
    lines = [
        "# AutoInsight Report",
        "",
        "## Business Question",
        business_question.strip() or "No business question provided.",
        "",
        "## Dataset Overview",
        f"- Rows: {profile.rows:,}",
        f"- Columns: {profile.columns:,}",
        f"- Duplicate rows: {profile.duplicate_rows:,}",
        f"- Memory footprint: {profile.memory_mb:.2f} MB",
        "",
        "## Modeling Setup",
        f"- Target column: {result.target or 'Not selected'}",
        f"- Task type: {result.task_type or 'Not inferred'}",
        "",
        "## Model Comparison",
    ]

    if result.model_results:
        for model in result.model_results:
            metrics = ", ".join(
                f"{name}: {value:.3f}" for name, value in model.secondary_metrics.items()
            )
            lines.append(f"- {model.name}: {model.score_name}={model.score:.3f}; {metrics}")
    else:
        lines.append("- No model was trained. The dataset may be too small or missing a valid target.")

    lines.extend(["", "## Recommendations"])
    lines.extend(f"- {recommendation}" for recommendation in result.recommendations)
    return "\n".join(lines)
