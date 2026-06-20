from __future__ import annotations

import pandas as pd


class DatasetValidationError(ValueError):
    """Raised when an uploaded dataset cannot be analyzed."""


def validate_dataset(df: pd.DataFrame) -> None:
    if df.empty:
        raise DatasetValidationError("The uploaded dataset is empty.")
    if df.shape[1] < 2:
        raise DatasetValidationError("The dataset needs at least two columns for analysis.")
    if df.dropna(how="all").empty:
        raise DatasetValidationError("The dataset only contains blank rows.")
