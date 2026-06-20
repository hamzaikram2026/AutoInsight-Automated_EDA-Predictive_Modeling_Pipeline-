from __future__ import annotations

import pandas as pd
import plotly.express as px
from plotly.graph_objects import Figure


def missingness_chart(df: pd.DataFrame) -> Figure:
    missing = (
        df.isna()
        .mean()
        .sort_values(ascending=False)
        .reset_index()
        .rename(columns={"index": "column", 0: "missing_rate"})
    )
    return px.bar(
        missing.head(20),
        x="column",
        y="missing_rate",
        title="Missing Values by Column",
        labels={"missing_rate": "Missing Rate", "column": "Column"},
    )


def numeric_distribution_chart(df: pd.DataFrame, column: str) -> Figure:
    return px.histogram(
        df,
        x=column,
        nbins=40,
        title=f"Distribution of {column}",
        marginal="box",
    )


def categorical_chart(df: pd.DataFrame, column: str) -> Figure:
    values = df[column].astype(str).value_counts().head(15).reset_index()
    values.columns = [column, "count"]
    return px.bar(values, x=column, y="count", title=f"Top Values for {column}")


def correlation_chart(df: pd.DataFrame) -> Figure | None:
    numeric = df.select_dtypes(include=["number"])
    if numeric.shape[1] < 2:
        return None
    corr = numeric.corr(numeric_only=True)
    return px.imshow(
        corr,
        text_auto=".2f",
        title="Numeric Correlation Heatmap",
        color_continuous_scale="RdBu_r",
        zmin=-1,
        zmax=1,
    )


def chart_candidates(df: pd.DataFrame) -> list[tuple[str, Figure]]:
    charts: list[tuple[str, Figure]] = [("Missingness", missingness_chart(df))]
    numeric_columns = df.select_dtypes(include=["number"]).columns.tolist()
    categorical_columns = df.select_dtypes(exclude=["number"]).columns.tolist()

    if numeric_columns:
        charts.append((f"{numeric_columns[0]} Distribution", numeric_distribution_chart(df, numeric_columns[0])))
    if categorical_columns:
        charts.append((f"{categorical_columns[0]} Categories", categorical_chart(df, categorical_columns[0])))

    corr = correlation_chart(df)
    if corr is not None:
        charts.append(("Correlation", corr))
    return charts
