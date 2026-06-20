from __future__ import annotations

import pandas as pd

from .schema import ColumnProfile, DatasetProfile


def _role_hint(series: pd.Series) -> str:
    name = series.name.lower()
    unique = series.nunique(dropna=True)
    unique_rate = unique / max(len(series), 1)

    if name in {"id", "uuid"} or name.endswith("_id"):
        return "identifier"
    if pd.api.types.is_bool_dtype(series) or unique <= 2:
        return "binary"
    if pd.api.types.is_numeric_dtype(series):
        return "numeric"
    if pd.api.types.is_datetime64_any_dtype(series):
        return "datetime"
    if unique_rate < 0.2:
        return "categorical"
    return "text"


def profile_dataset(df: pd.DataFrame) -> DatasetProfile:
    profiles: list[ColumnProfile] = []
    rows = len(df)

    for column in df.columns:
        series = df[column]
        missing = int(series.isna().sum())
        profiles.append(
            ColumnProfile(
                name=str(column),
                dtype=str(series.dtype),
                missing=missing,
                missing_rate=missing / max(rows, 1),
                unique=int(series.nunique(dropna=True)),
                role_hint=_role_hint(series),
            )
        )

    memory_mb = float(df.memory_usage(deep=True).sum() / 1_000_000)
    return DatasetProfile(
        rows=rows,
        columns=len(df.columns),
        duplicate_rows=int(df.duplicated().sum()),
        memory_mb=memory_mb,
        column_profiles=profiles,
    )


def profile_frame(profile: DatasetProfile) -> pd.DataFrame:
    return pd.DataFrame([profile.__dict__ for profile in profile.column_profiles])
