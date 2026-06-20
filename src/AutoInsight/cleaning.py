from __future__ import annotations

import pandas as pd


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = df.copy()
    cleaned.columns = [
        str(column).strip().lower().replace(" ", "_").replace("-", "_")
        for column in cleaned.columns
    ]
    return cleaned


def coerce_dates(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = df.copy()
    for column in cleaned.select_dtypes(include=["object", "string"]).columns:
        sample = cleaned[column].dropna().astype(str).head(25)
        if sample.empty:
            continue
        date_like = sample.str.match(r"^\d{4}[-/]\d{1,2}[-/]\d{1,2}").mean()
        if date_like < 0.8:
            continue
        parsed = pd.to_datetime(sample, errors="coerce", format="mixed")
        if parsed.notna().mean() >= 0.8:
            cleaned[column] = pd.to_datetime(cleaned[column], errors="coerce", format="mixed")
    return cleaned


def remove_high_missing_columns(df: pd.DataFrame, threshold: float = 0.6) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    keep = df.isna().mean() <= threshold
    return df.loc[:, keep].copy()


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = normalize_columns(df)
    cleaned = cleaned.drop_duplicates()
    cleaned = remove_high_missing_columns(cleaned)
    cleaned = coerce_dates(cleaned)
    return cleaned
