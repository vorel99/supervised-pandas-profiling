from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from ydata_profiling.config import Settings
from ydata_profiling.model.pandas.correlations_pandas import (
    pandas_auto_compute,
    pandas_cramers_compute,
    pandas_spearman_compute,
)


@pytest.fixture
def test_dataframe():
    size = 50
    dataframe = pd.DataFrame(
        {
            "float_1": np.random.rand(size),
            "float_2": np.random.rand(size),
            "integer_1": np.random.randint(low=2, high=1000, size=size),
            "integer_2": np.random.randint(low=2, high=1000, size=size),
            "string_1": np.random.randint(
                low=1,
                high=10,
                size=size,
            ).astype(str),
            "string_2": np.random.randint(
                low=1,
                high=10,
                size=size,
            ).astype(str),
        }
    )
    return dataframe


@pytest.fixture
def test_config():
    return Settings()


@pytest.fixture
def test_summary():
    # mock numeric description
    num_desc = MagicMock()
    num_desc.n_distinct = 10
    num_desc.__getitem__.return_value = "Numeric"

    # mock categorical description
    cat_desc = MagicMock()
    cat_desc.n_distinct = 10
    cat_desc.__getitem__.return_value = "Categorical"

    return {
        "float_1": num_desc,
        "float_2": num_desc,
        "integer_1": num_desc,
        "integer_2": num_desc,
        "string_1": cat_desc,
        "string_2": cat_desc,
    }


def test_auto_compute_all(test_config, test_dataframe, test_summary):
    pandas_auto_compute(test_config, test_dataframe, test_summary)


def test_numeric_auto_equals_spearman(test_config, test_dataframe, test_summary):
    df = test_dataframe[["float_1", "float_2"]]
    summary = {
        column: value
        for column, value in test_summary.items()
        if column in ["float_1", "float_2"]
    }
    auto_result = pandas_auto_compute(test_config, df, summary)
    spearman_result = pandas_spearman_compute(test_config, df, summary)
    assert auto_result.iloc[0][1] == pytest.approx(spearman_result.iloc[0][1], 0.01)


def test_categorical_auto_equals_equals_cramers(
    test_config, test_dataframe, test_summary
):
    df = test_dataframe[["string_1", "string_2"]]
    summary = {
        column: value
        for column, value in test_summary.items()
        if column in ["string_1", "string_2"]
    }
    auto_result = pandas_auto_compute(test_config, df, summary)
    cramers_result = pandas_cramers_compute(test_config, df, summary)
    assert auto_result.iloc[0][1] == pytest.approx(cramers_result.iloc[0][1], 0.01)
