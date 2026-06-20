from __future__ import annotations

import re

import pandas as pd

from .schema import TaskType


TARGET_KEYWORDS = (
    "target",
    "label",
    "outcome",
    "churn",
    "revenue",
    "sales",
    "profit",
    "price",
    "score",
    "risk",
    "conversion",
)


def infer_target(df: pd.DataFrame, business_question: str) -> str | None:
    if df.empty:
        return None

    question = business_question.lower()
    normalized_question = set(re.findall(r"[a-z0-9_]+", question))
    columns = [str(column) for column in df.columns]

    for column in columns:
        normalized_column = column.lower().replace(" ", "_")
        if normalized_column in normalized_question or column.lower() in question:
            return column

    keyword_matches = [
        column
        for column in columns
        if any(keyword in column.lower() for keyword in TARGET_KEYWORDS)
    ]
    if keyword_matches:
        return keyword_matches[-1]

    usable_columns = [
        column
        for column in columns
        if not column.lower().endswith("_id") and column.lower() not in {"id", "uuid"}
    ]
    return usable_columns[-1] if usable_columns else columns[-1]


def infer_task_type(series: pd.Series) -> TaskType:
    unique = series.nunique(dropna=True)
    if pd.api.types.is_numeric_dtype(series) and unique > 12:
        return "regression"
    return "classification"
