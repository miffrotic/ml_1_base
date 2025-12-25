import re

from typing import Callable, Self

import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin


template = re.compile(r"\d+\.?\d*")


def get_float(value: str | float) -> float | None:
    if pd.isna(value):
        return

    if not isinstance(value, str):
        value = str(value)

    match = template.search(value)

    if match:
        return float(match[0])

    return


def get_object(value: np.float64) -> str | None:
    if pd.isna(value):
        return np.float64("nan")

    return str(int(value))


class ColumnConverter(BaseEstimator, TransformerMixin):
    def __init__(self, func: Callable) -> None:
        self.func = func

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> Self:
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        for column in X.columns:
            X[column] = X[column].apply(self.func)
        return X

    def get_feature_names_out(self, input_features: pd.Series | None = None) -> pd.Index:
        return input_features
