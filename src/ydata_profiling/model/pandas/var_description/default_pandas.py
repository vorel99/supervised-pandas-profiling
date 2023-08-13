from dataclasses import dataclass

import pandas as pd

from ydata_profiling.config import Settings
from ydata_profiling.model.pandas.var_description.counts_pandas import VarCountsPandas
from ydata_profiling.model.var_description.default import VarDescription


@dataclass
class VarDescriptionPandas(VarDescription):
    """Default description for pandas columns"""

    def __init__(self, config: Settings, series: pd.Series, init_dict: dict):
        _var_counts = VarCountsPandas(config, series)

        _count = _var_counts.count
        value_counts = _var_counts.value_counts_without_nan
        distinct_count = len(value_counts)
        unique_count = value_counts.where(value_counts == 1).count()

        super().__init__(
            n=_var_counts.n,
            count=_var_counts.count,
            n_missing=_var_counts.n_missing,
            p_missing=_var_counts.p_missing,
            hashable=_var_counts.hashable,
            memory_size=_var_counts.memory_size,
            ordering=_var_counts.ordering,
            value_counts_index_sorted=_var_counts.value_counts_index_sorted,
            value_counts_without_nan=_var_counts.value_counts_without_nan,
            n_distinct=distinct_count,
            p_distinct=distinct_count / _count if _count > 0 else 0,
            is_unique=unique_count == _count and _count > 0,
            n_unique=unique_count,
            p_unique=unique_count / _count if _count > 0 else 0,
            var_specific=init_dict,
        )
