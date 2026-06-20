from __future__ import annotations

import pandas as pd

from .cleaning import clean_dataset
from .modeling import train_baselines
from .profiling import profile_dataset
from .recommendations import data_quality_recommendations, modeling_recommendations
from .schema import AnalysisResult
from .targeting import infer_target, infer_task_type
from .validation import validate_dataset


def run_analysis(
    df: pd.DataFrame,
    business_question: str,
    target_override: str | None = None,
) -> tuple[pd.DataFrame, AnalysisResult]:
    validate_dataset(df)
    cleaned = clean_dataset(df)
    profile = profile_dataset(cleaned)
    target = target_override or infer_target(cleaned, business_question)

    task_type = None
    model_results = []
    if target and target in cleaned.columns:
        task_type = infer_task_type(cleaned[target])
        model_results = train_baselines(cleaned, target, task_type)

    result = AnalysisResult(
        target=target,
        task_type=task_type,
        profile=profile,
        model_results=model_results,
        recommendations=[],
    )
    recommendations = data_quality_recommendations(profile) + modeling_recommendations(result)
    result = AnalysisResult(
        target=result.target,
        task_type=result.task_type,
        profile=result.profile,
        model_results=result.model_results,
        recommendations=recommendations,
    )
    return cleaned, result
