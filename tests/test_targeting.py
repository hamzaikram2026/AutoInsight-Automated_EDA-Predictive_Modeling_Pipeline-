import pandas as pd

from src.ai_data_scientist.targeting import infer_target, infer_task_type


def test_infer_target_from_business_question():
    df = pd.DataFrame(
        {
            "customer_id": [1, 2, 3],
            "monthly_revenue": [10, 20, 30],
            "churn": [0, 1, 0],
        }
    )

    assert infer_target(df, "What predicts churn?") == "churn"


def test_infer_regression_for_continuous_numeric_target():
    series = pd.Series(range(20))

    assert infer_task_type(series) == "regression"
