from __future__ import annotations

from .schema import AnalysisResult, DatasetProfile


def data_quality_recommendations(profile: DatasetProfile) -> list[str]:
    recommendations: list[str] = []

    if profile.duplicate_rows:
        recommendations.append(
            f"Review {profile.duplicate_rows:,} duplicate rows before using this dataset for production decisions."
        )

    high_missing = [
        column.name
        for column in profile.column_profiles
        if column.missing_rate >= 0.3
    ]
    if high_missing:
        recommendations.append(
            "Investigate high-missing columns: " + ", ".join(high_missing[:5]) + "."
        )

    identifiers = [
        column.name
        for column in profile.column_profiles
        if column.role_hint == "identifier"
    ]
    if identifiers:
        recommendations.append(
            "Exclude identifier-like columns from predictive modeling unless they encode meaningful business groups."
        )

    if not recommendations:
        recommendations.append("The dataset has no major structural quality warnings from the automated profile.")

    return recommendations


def modeling_recommendations(result: AnalysisResult) -> list[str]:
    if not result.model_results:
        return [
            "Add more labeled rows or choose a clearer target column before relying on predictive modeling."
        ]

    best = result.model_results[0]
    if best.task_type == "classification":
        return [
            f"Use {best.name} as the current baseline with weighted F1 of {best.score:.3f}.",
            "Compare future models against this score using the same validation split or cross-validation.",
        ]
    return [
        f"Use {best.name} as the current baseline with R2 of {best.score:.3f}.",
        "Review residuals and business cost of errors before deploying this regression model.",
    ]
