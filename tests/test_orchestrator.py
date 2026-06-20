import pandas as pd

from src.ai_data_scientist.orchestrator import run_analysis


def test_run_analysis_returns_cleaned_data_and_result():
    df = pd.DataFrame(
        {
            "age": [22, 30, 41, 35, 28, 50, 46, 33, 39, 24, 55, 60],
            "plan": ["basic", "pro"] * 6,
            "churn": [0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1],
        }
    )

    cleaned, result = run_analysis(df, "What predicts churn?", "churn")

    assert cleaned.shape[0] == 12
    assert result.target == "churn"
    assert result.task_type == "classification"
    assert result.recommendations
