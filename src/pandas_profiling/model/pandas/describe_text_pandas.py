from typing import Optional, Tuple

import pandas as pd

from pandas_profiling.config import Settings
from pandas_profiling.model.description_target import TargetDescription
from pandas_profiling.model.pandas.describe_categorical_pandas import (
    length_summary_vc,
    unicode_summary_vc,
    word_summary_vc,
)
from pandas_profiling.model.pandas.description_variable_pandas import (
    TextDescriptionPandas,
    TextDescriptionSupervisedPandas,
)
from pandas_profiling.model.summary_algorithms import (
    describe_text_1d,
    histogram_compute,
    series_handle_nulls,
    series_hashable,
)


@describe_text_1d.register
@series_hashable
@series_handle_nulls
def pandas_describe_text_1d(
    config: Settings,
    series: pd.Series,
    summary: dict,
    target_description: Optional[TargetDescription] = None,
) -> Tuple[Settings, pd.Series, dict, Optional[TargetDescription]]:
    """Describe string series.

    Args:
        config: report Settings object
        series: The Series to describe.
        summary: The dict containing the series description so far.

    Returns:
        A dict containing calculated series description values.
    """

    series = series.astype(str)

    # Only run if at least 1 non-missing value
    value_counts = summary["value_counts_without_nan"]
    value_counts.index = value_counts.index.astype(str)

    summary.update({"first_rows": series.head(5)})

    if config.vars.text.length:
        summary.update(length_summary_vc(value_counts))
        summary.update(
            histogram_compute(
                config,
                summary["length_histogram"].index.values,
                len(summary["length_histogram"]),
                name="histogram_length",
                weights=summary["length_histogram"].values,
            )
        )

    if config.vars.text.characters:
        summary.update(unicode_summary_vc(value_counts))

    if config.vars.text.words:
        summary.update(word_summary_vc(value_counts, config.vars.cat.stop_words))

    if target_description:
        summary["plot_description"] = TextDescriptionSupervisedPandas(
            config.vars, series, target_description
        )
    else:
        summary["plot_description"] = TextDescriptionPandas(config.vars, series)

    return config, series, summary, target_description
