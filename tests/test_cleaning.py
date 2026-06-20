import pandas as pd

from src.ai_data_scientist.cleaning import clean_dataset


def test_clean_dataset_normalizes_columns_and_removes_duplicates():
    df = pd.DataFrame(
        {
            "Customer ID": [1, 1, 2],
            "Plan-Type": ["Pro", "Pro", "Basic"],
            "Mostly Missing": [None, None, None],
        }
    )

    cleaned = clean_dataset(df)

    assert cleaned.columns.tolist() == ["customer_id", "plan_type"]
    assert len(cleaned) == 2
