from typing import Optional, Tuple

import numpy as np
import pandas as pd

from pandas_profiling.config import Settings
from pandas_profiling.model.description_target import TargetDescription
from pandas_profiling.model.pandas.description_variable_pandas import (
    NumDescriptionPandas,
    NumDescriptionSupervisedPandas,
)
from pandas_profiling.model.summary_algorithms import (
    chi_square,
    describe_date_1d,
    histogram_compute,
    series_handle_nulls,
    series_hashable,
)


@describe_date_1d.register
@series_hashable
@series_handle_nulls
def pandas_describe_date_1d(
    config: Settings,
    series: pd.Series,
    summary: dict,
    target_description: Optional[TargetDescription] = None,
) -> Tuple[Settings, pd.Series, dict, Optional[TargetDescription]]:
    """Describe a date series.

    Args:
        config: report Settings object
        series: The Series to describe.
        summary: The dict containing the series description so far.

    Returns:
        A dict containing calculated series description values.
    """
    summary.update(
        {
            "min": pd.Timestamp.to_pydatetime(series.min()),
            "max": pd.Timestamp.to_pydatetime(series.max()),
        }
    )

    summary["range"] = summary["max"] - summary["min"]

    values = pd.to_datetime(series).astype(np.int64) // 10**9

    if config.vars.num.chi_squared_threshold > 0.0:
        summary["chi_squared"] = chi_square(values)

    summary.update(histogram_compute(config, values, summary["n_distinct"]))

    if target_description is None:
        plot_bins = config.plot.histogram.bins
        summary["plot_description"] = NumDescriptionPandas(
            config.vars, values, plot_bins
        )
    else:
        plot_bins = config.plot.histogram.bins_supervised
        summary["plot_description"] = NumDescriptionSupervisedPandas(
            config.vars, values, plot_bins, target_description
        )

    return config, values, summary, target_description
