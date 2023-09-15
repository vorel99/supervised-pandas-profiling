from dataclasses import dataclass

import pandas as pd

from ydata_profiling.config import Settings
from ydata_profiling.model.var_description.counts import VarCounts


@dataclass
class VarCountsPandas(VarCounts):
    value_counts_without_nan: pd.Series
    """Counts of values in the series without NaN."""
    value_counts_index_sorted: pd.Series
    """Sorted counts of values in the series without NaN."""

    def __init__(self, config: Settings, series: pd.Series):
        """Counts the values in a series (with and without NaN, distinct).

        Args:
            config: report Settings object
            series: Series for which we want to calculate the values.
            summary: series' summary

        Returns:
            A dictionary with the count values (with and without NaN, distinct).
        """
        length = len(series)

        try:
            value_counts_with_nan = series.value_counts(dropna=False)
            _ = set(value_counts_with_nan.index)
            hashable = True
        except:  # noqa: E722
            hashable = False

        value_counts_without_nan = None
        value_counts_index_sorted = None
        if hashable:
            value_counts_with_nan = value_counts_with_nan[value_counts_with_nan > 0]

            null_index = value_counts_with_nan.index.isnull()
            if null_index.any():
                n_missing = value_counts_with_nan[null_index].sum()
                value_counts_without_nan = value_counts_with_nan[~null_index]
            else:
                n_missing = 0
                value_counts_without_nan = value_counts_with_nan

            try:
                value_counts_index_sorted = value_counts_without_nan.sort_index(
                    ascending=True
                )
                ordering = True
            except TypeError:
                ordering = False
        else:
            n_missing = series.isna().sum()
            ordering = False

        super().__init__(
            hashable=hashable,
            value_counts_without_nan=value_counts_without_nan,
            value_counts_index_sorted=value_counts_index_sorted,
            ordering=ordering,
            n_missing=n_missing,
            n=length,
            p_missing=series.isna().sum() / length if length > 0 else 0,
            count=length - series.isna().sum(),
            memory_size=series.memory_usage(deep=config.memory_deep),
            value_counts=None,
        )
