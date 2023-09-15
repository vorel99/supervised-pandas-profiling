from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from ydata_profiling.config import Settings
from ydata_profiling.model.pandas.var_description.counts_pandas import VarCountsPandas
from ydata_profiling.model.var_description.default import (
    VarDescription,
    VarDescriptionHashable,
)


@dataclass
class VarDescriptionPandas(VarDescription):
    """Default description for pandas columns."""

    @classmethod
    def from_var_counts(
        cls, var_counts: VarCountsPandas, init_dict: dict
    ) -> VarDescriptionPandas:
        """Get a default description from a VarCountsPandas object."""
        return VarDescriptionPandas(
            n=var_counts.n,
            count=var_counts.count,
            n_missing=var_counts.n_missing,
            p_missing=var_counts.p_missing,
            hashable=var_counts.hashable,
            memory_size=var_counts.memory_size,
            ordering=var_counts.ordering,
            value_counts_index_sorted=var_counts.value_counts_index_sorted,
            value_counts_without_nan=var_counts.value_counts_without_nan,
            var_specific=init_dict,
        )


@dataclass
class VarDescriptionPandasHashable(VarDescriptionHashable):
    """Default description for pandas columns that are hashable (common types)."""

    @classmethod
    def from_var_counts(
        cls, var_counts: VarCountsPandas, init_dict: dict
    ) -> VarDescriptionPandasHashable:
        """Get a default description for a hashable column from a VarCountsPandas object."""
        _count = var_counts.count
        value_counts = var_counts.value_counts_without_nan
        distinct_count = len(value_counts)
        unique_count = value_counts.where(value_counts == 1).count()

        return VarDescriptionPandasHashable(
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
            p_distinct=distinct_count / _count if _count > 0 else 0,
            is_unique=unique_count == _count and _count > 0,
            n_unique=unique_count,
            p_unique=unique_count / _count if _count > 0 else 0,
            var_specific=init_dict,
            value_counts=var_counts.value_counts,
        )


def get_default_pandas_description(
    config: Settings, series: pd.Series, init_dict: dict
) -> VarDescriptionPandas | VarDescriptionPandasHashable:
    _var_counts = VarCountsPandas(config, series)
    if _var_counts.hashable:
        return VarDescriptionPandasHashable.from_var_counts(_var_counts, init_dict)
    return VarDescriptionPandas.from_var_counts(_var_counts, init_dict)
