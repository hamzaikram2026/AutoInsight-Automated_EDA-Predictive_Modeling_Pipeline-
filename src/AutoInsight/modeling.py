from __future__ import annotations

import math

import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from .features import build_preprocessor, split_features_target
from .schema import ModelResult, TaskType


def _classification_models() -> dict[str, object]:
    return {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Random Forest Classifier": RandomForestClassifier(
            n_estimators=150,
            random_state=42,
            class_weight="balanced",
        ),
    }


def _regression_models() -> dict[str, object]:
    return {
        "Ridge Regression": Ridge(alpha=1.0),
        "Random Forest Regressor": RandomForestRegressor(
            n_estimators=150,
            random_state=42,
        ),
    }


def train_baselines(df: pd.DataFrame, target: str, task_type: TaskType) -> list[ModelResult]:
    modeling_df = df.dropna(subset=[target]).copy()
    if len(modeling_df) < 10 or modeling_df[target].nunique(dropna=True) < 2:
        return []

    x, y = split_features_target(modeling_df, target)
    if x.shape[1] == 0:
        return []

    stratify = y if task_type == "classification" and y.value_counts().min() >= 2 else None
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.25,
        random_state=42,
        stratify=stratify,
    )

    models = _classification_models() if task_type == "classification" else _regression_models()
    results: list[ModelResult] = []

    for name, estimator in models.items():
        pipeline = Pipeline(
            steps=[
                ("preprocessor", build_preprocessor(x_train)),
                ("model", estimator),
            ]
        )
        pipeline.fit(x_train, y_train)
        predictions = pipeline.predict(x_test)

        if task_type == "classification":
            accuracy = float(accuracy_score(y_test, predictions))
            f1 = float(f1_score(y_test, predictions, average="weighted", zero_division=0))
            results.append(
                ModelResult(
                    name=name,
                    task_type=task_type,
                    score_name="weighted_f1",
                    score=f1,
                    secondary_metrics={"accuracy": accuracy},
                )
            )
        else:
            mae = float(mean_absolute_error(y_test, predictions))
            rmse = float(math.sqrt(mean_squared_error(y_test, predictions)))
            r2 = float(r2_score(y_test, predictions))
            results.append(
                ModelResult(
                    name=name,
                    task_type=task_type,
                    score_name="r2",
                    score=r2,
                    secondary_metrics={"mae": mae, "rmse": rmse},
                )
            )

    return sorted(results, key=lambda result: result.score, reverse=True)
