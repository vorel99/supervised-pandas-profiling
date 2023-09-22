from __future__ import annotations

import pandas as pd

from ydata_profiling.config import Settings
from ydata_profiling.model.pandas.var_description.counts_pandas import get_counts_pandas
from ydata_profiling.model.var_description.default import (
    VarDescription,
    VarDescriptionHashable,
)


def get_default_pandas_description(
    config: Settings, series: pd.Series, init_dict: dict
) -> VarDescription | VarDescriptionHashable:
    var_counts = get_counts_pandas(config, series)

    if var_counts.hashable:
        count = var_counts.count
        value_counts = var_counts.value_counts_without_nan
        distinct_count = len(value_counts)
        unique_count = value_counts.where(value_counts == 1).count()

        return VarDescriptionHashable(
            n=var_counts.n,
            count=var_counts.count,
            n_missing=var_counts.n_missing,
            p_missing=var_counts.p_missing,
            hashable=var_counts.hashable,
            memory_size=var_counts.memory_size,
            ordering=var_counts.ordering,
            value_counts_index_sorted=var_counts.value_counts_index_sorted,
            value_counts_without_nan=var_counts.value_counts_without_nan,
            n_distinct=distinct_count,
            p_distinct=distinct_count / count if count > 0 else 0,
            is_unique=unique_count == count and count > 0,
            n_unique=unique_count,
            p_unique=unique_count / count if count > 0 else 0,
            value_counts=None,
            var_specific=init_dict,
        )
    return VarDescription.from_var_counts(var_counts, init_dict)
