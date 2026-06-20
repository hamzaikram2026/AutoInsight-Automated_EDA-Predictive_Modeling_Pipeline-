from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


TaskType = Literal["classification", "regression"]


@dataclass(frozen=True)
class ColumnProfile:
    name: str
    dtype: str
    missing: int
    missing_rate: float
    unique: int
    role_hint: str


@dataclass(frozen=True)
class DatasetProfile:
    rows: int
    columns: int
    duplicate_rows: int
    memory_mb: float
    column_profiles: list[ColumnProfile]


@dataclass(frozen=True)
class ModelResult:
    name: str
    task_type: TaskType
    score_name: str
    score: float
    secondary_metrics: dict[str, float]


@dataclass(frozen=True)
class AnalysisResult:
    target: str | None
    task_type: TaskType | None
    profile: DatasetProfile
    model_results: list[ModelResult]
    recommendations: list[str]
