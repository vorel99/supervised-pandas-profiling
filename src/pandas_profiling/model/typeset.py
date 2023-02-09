import imghdr
import os
import warnings
from functools import partial, wraps
from typing import Callable, Sequence, Set
from urllib.parse import urlparse

import pandas as pd
import visions
from multimethod import multimethod
from pandas.api import types as pdt
from pandas_profiling.config import Settings
from pandas_profiling.model.typeset_relations import (
    category_is_numeric,
    category_to_numeric,
    numeric_is_category,
    series_is_string,
    string_is_bool,
    string_is_category,
    string_is_datetime,
    string_to_bool,
    string_to_datetime,
    to_bool,
    to_category,
)
from visions.backends.pandas.series_utils import series_not_empty
from visions.relations import IdentityRelation, InferenceRelation, TypeRelation

pandas_has_string_dtype_flag = hasattr(pdt, "is_string_dtype")


def series_handle_nulls(fn: Callable[..., bool]) -> Callable[..., bool]:
    """Decorator for nullable series"""

    @wraps(fn)
    def inner(series: pd.Series, state: dict, *args, **kwargs) -> bool:
        if "hasnans" not in state:
            state["hasnans"] = series.hasnans

        if state["hasnans"]:
            series = series.dropna()
            if series.empty:
                return False

        return fn(series, state, *args, **kwargs)

    return inner


def typeset_types(config: Settings) -> Set[visions.VisionsBaseType]:
    """Define types based on the config"""

    class Unsupported(visions.Generic):
        pass

    class Numeric(visions.VisionsBaseType):
        """Type for all numeric (float, int) columns.

        Examples
        --------
        >>> pd.Series([1, 2, 5, 3, 8, 9])
        >>> pd.Series([.34, 2.9, 55, 3.14, 89, 91])
        """

        @staticmethod
        def get_relations() -> Sequence[TypeRelation]:
            return [
                IdentityRelation(Unsupported),
                InferenceRelation(
                    Categorical,
                    relationship=lambda x, y: partial(category_is_numeric, k=config)(
                        x, y
                    ),
                    transformer=category_to_numeric,
                ),
            ]

        @staticmethod
        @multimethod
        @series_not_empty
        @series_handle_nulls
        def contains_op(series: pd.Series, state: dict) -> bool:
            return pdt.is_numeric_dtype(series) and not pdt.is_bool_dtype(series)

    class String(visions.VisionsBaseType):
        """Type for plaintext columns.
        Like name, note, string identifier, residence etc.

        Examples
        --------
        >>> pd.Series(["AX01", "BC32", "AC00"])
        >>> pd.Series([
                        "Peggy Holt",
                        "Marina Mayo",
                        "Hussain Baker",
                        "Salman Joyce",
                        "Keeley Stanley",
                        "Theodore Summers",
                        "Yasmine Delacruz",
                        "Mahdi Munoz",
                        "Riley Fulton",
                        "Tomos Butler",
                    ])
        """

        @staticmethod
        def get_relations() -> Sequence[TypeRelation]:
            return [
                IdentityRelation(Unsupported),
            ]

        @staticmethod
        @multimethod
        @series_not_empty
        @series_handle_nulls
        def contains_op(series: pd.Series, state: dict) -> bool:
            return series_is_string(series, state)

    class DateTime(visions.VisionsBaseType):
        @staticmethod
        def get_relations() -> Sequence[TypeRelation]:
            return [
                IdentityRelation(Unsupported),
                InferenceRelation(
                    String,
                    relationship=lambda x, y: partial(string_is_datetime)(x, y),
                    transformer=string_to_datetime,
                ),
            ]

        @staticmethod
        @multimethod
        @series_not_empty
        @series_handle_nulls
        def contains_op(series: pd.Series, state: dict) -> bool:
            if not pdt.is_object_dtype(series):
                return pandas_has_string_dtype_flag and pdt.is_string_dtype(series)
            return pdt.is_datetime64_any_dtype(series) or string_is_datetime(
                series, state
            )

    class Categorical(visions.VisionsBaseType):
        """Type for categorical columns.
        Categorical columns in pandas categorical format
        and columns in string format with small count of unique values.

        Examples
        --------
        >>> pd.Series(["male", "female", "female", "male", "female"])
        >>> pd.Series(["class 1", "class 3", "class 1", "class 2"])
        """

        @staticmethod
        def get_relations() -> Sequence[TypeRelation]:
            return [
                IdentityRelation(Unsupported),
                IdentityRelation(String),
                InferenceRelation(
                    Numeric,
                    relationship=lambda x, y: partial(numeric_is_category, k=config)(
                        x, y
                    ),
                    transformer=to_category,
                ),
                InferenceRelation(
                    String,
                    relationship=lambda x, y: partial(numeric_is_category, k=config)(
                        x, y
                    ),
                    transformer=to_category,
                ),
            ]

        @staticmethod
        @multimethod
        @series_not_empty
        @series_handle_nulls
        def contains_op(series: pd.Series, state: dict) -> bool:
            is_valid_dtype = pdt.is_categorical_dtype(series) and not pdt.is_bool_dtype(
                series
            )
            if is_valid_dtype:
                return True
            elif not pdt.is_object_dtype(series):
                return pandas_has_string_dtype_flag and pdt.is_string_dtype(series)

            return series_is_string(series, state) and string_is_category(series, state)

    class Boolean(visions.VisionsBaseType):
        @staticmethod
        def get_relations() -> Sequence[TypeRelation]:
            # Numeric [0, 1] goes via Categorical with distinct_count_without_nan <= 2
            mapping = config.vars.bool.mappings

            return [
                IdentityRelation(Unsupported),
                InferenceRelation(
                    Categorical,
                    relationship=lambda x, y: partial(string_is_bool, k=mapping)(x, y),
                    transformer=lambda s, st: to_bool(
                        partial(string_to_bool, k=mapping)(s, st)
                    ),
                ),
            ]

        @staticmethod
        @multimethod
        @series_not_empty
        @series_handle_nulls
        def contains_op(series: pd.Series, state: dict) -> bool:
            if pdt.is_object_dtype(series):
                try:
                    return series.isin({True, False}).all()
                except:  # noqa: E722
                    return False

            return pdt.is_bool_dtype(series)

    class URL(visions.VisionsBaseType):
        @staticmethod
        def get_relations() -> Sequence[TypeRelation]:
            return [IdentityRelation(Categorical)]

        @staticmethod
        @multimethod
        @series_handle_nulls
        def contains_op(series: pd.Series, state: dict) -> bool:
            # TODO: use coercion utils
            try:
                url_gen = (urlparse(x) for x in series)
                return all(x.netloc and x.scheme for x in url_gen)  # noqa: TC300
            except AttributeError:
                return False

    class Path(visions.VisionsBaseType):
        @staticmethod
        def get_relations() -> Sequence[TypeRelation]:
            return [IdentityRelation(Categorical)]

        @staticmethod
        @multimethod
        @series_handle_nulls
        def contains_op(series: pd.Series, state: dict) -> bool:
            # TODO: use coercion utils
            try:
                return all(os.path.isabs(p) for p in series)
            except TypeError:
                return False

    class File(visions.VisionsBaseType):
        @staticmethod
        def get_relations() -> Sequence[TypeRelation]:
            return [IdentityRelation(Path)]

        @staticmethod
        @multimethod
        @series_handle_nulls
        def contains_op(series: pd.Series, state: dict) -> bool:
            return all(os.path.exists(p) for p in series)

    class Image(visions.VisionsBaseType):
        @staticmethod
        def get_relations() -> Sequence[TypeRelation]:
            return [IdentityRelation(File)]

        @staticmethod
        @multimethod
        @series_handle_nulls
        def contains_op(series: pd.Series, state: dict) -> bool:
            return all(imghdr.what(p) for p in series)

    class TimeSeries(visions.VisionsBaseType):
        @staticmethod
        def get_relations() -> Sequence[TypeRelation]:
            return [IdentityRelation(Numeric)]

        @staticmethod
        @multimethod
        @series_not_empty
        @series_handle_nulls
        def contains_op(series: pd.Series, state: dict) -> bool:
            def is_timedependent(series: pd.Series) -> bool:
                autocorrelation_threshold = config.vars.timeseries.autocorrelation
                lags = config.vars.timeseries.lags
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    for lag in lags:
                        autcorr = series.autocorr(lag=lag)
                        if autcorr >= autocorrelation_threshold:
                            return True

                return False

            is_numeric = pdt.is_numeric_dtype(series) and not pdt.is_bool_dtype(series)
            return is_numeric and is_timedependent(series)

    types = {Unsupported, Boolean, Numeric, String, Categorical, DateTime}
    if config.vars.path.active:
        types.add(Path)
        if config.vars.file.active:
            types.add(File)
            if config.vars.image.active:
                types.add(Image)

    if config.vars.url.active:
        types.add(URL)

    if config.vars.timeseries.active:
        types.add(TimeSeries)

    return types


class ProfilingTypeSet(visions.VisionsTypeset):
    def __init__(self, config: Settings):
        self.config = config

        types = typeset_types(config)

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            super().__init__(types)
