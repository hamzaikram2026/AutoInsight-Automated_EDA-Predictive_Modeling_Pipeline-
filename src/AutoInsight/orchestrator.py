from __future__ import annotations
import time
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
    

    start = time.perf_counter()
    validate_dataset(df)
    print("Validation:", time.perf_counter() - start)

    start = time.perf_counter()
    cleaned = clean_dataset(df)
    print("Cleaning:", time.perf_counter() - start)

    start = time.perf_counter()
    profile = profile_dataset(cleaned)
    print("Profiling:", time.perf_counter() - start)

    start = time.perf_counter()
    target = target_override or infer_target(cleaned, business_question)
    print("Target inference:", time.perf_counter() - start)

    print("Cleaned Columns:", cleaned.columns.tolist())
    print("Target Override:", target_override)
    print("Final Target:", target)

    task_type = None
    model_results = []

    print("Target:", target)
    print("Target in columns:", target in cleaned.columns)


    if target and target in cleaned.columns:
        print("Inside IF block")

        print("Target dtype:", cleaned[target].dtype)
        print("Unique values:", cleaned[target].unique()[:10])

        start = time.perf_counter()
        task_type = infer_task_type(cleaned[target])
        print("Task inference:", time.perf_counter() - start)
        print("Task Type:", task_type)

        start = time.perf_counter()
        model_results = train_baselines(cleaned, target, task_type)
        print("Training:", time.perf_counter() - start)
        print("Models Trained:", len(model_results))

    result = AnalysisResult(
        target=target,
        task_type=task_type,
        profile=profile,
        model_results=model_results,
        recommendations=[],
    )
    start = time.perf_counter()
    recommendations = data_quality_recommendations(profile) + modeling_recommendations(result)
    print("Recommendations:", time.perf_counter() - start)

    result = AnalysisResult(
        target=result.target,
        task_type=result.task_type,
        profile=result.profile,
        model_results=result.model_results,
        recommendations=recommendations,
    )
    return cleaned, result
