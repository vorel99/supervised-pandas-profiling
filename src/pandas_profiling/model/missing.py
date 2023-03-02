import warnings
from abc import ABCMeta, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional

import numpy as np
import pandas as pd
from multimethod import multimethod
from pandas_profiling.config import Settings
from pandas_profiling.model.description_target import TargetDescription
from scipy.stats import chi2_contingency


@dataclass
class MissingConfMatrix:
    """Class for confusion matrix in absolute and relative numbers.

    Args:
        absolute_counts (pd.DataFrame): absolute counts of missing.
        relative_counts (pd.DataFrame): relative counts of missing.
    """

    absolute_counts: pd.DataFrame
    relative_counts: pd.DataFrame
    _p_value: Optional[float] = None
    _expected_counts: Optional[np.ndarray] = None

    @property
    def p_value(self) -> float:
        """P value from chi-square test of variables from contingency table."""
        if not self._p_value:
            self._p_value = chi2_contingency(self.absolute_counts)[1]
        return self._p_value

    @property
    def expected_counts(self):
        """Matrix of expected frequencies, based on the marginal sums of the table."""
        if not self._expected_counts:
            self._expected_counts = chi2_contingency(self.absolute_counts)[3]
        return self._expected_counts

    @property
    def plot_labels(self) -> list:
        """Labels for confusion matrix.
        Contain relative count and absolute count of values."""
        flat_abs = self.absolute_counts.values.flatten()
        flat_rel = self.relative_counts.values.flatten()
        labels = []
        for abs, rel in zip(flat_abs, flat_rel):
            labels.append("{0:.2%}\n{1}".format(rel, abs))
        return np.asarray(labels).reshape(self.absolute_counts.shape)


@dataclass
class MissingDescription(metaclass=ABCMeta):
    """Description of missing dependency on target.

    Args:
        missing_target: Dict[str, MissingConfMatrix]
            Confusion matrixes target x missing for variables with missing values.
            key: column name
            value: confusion matrix of missing vs target
    """

    missing_target: Dict[str, MissingConfMatrix]


@multimethod
def get_missing_description(
    config: Settings, df: Any, target_description: TargetDescription
) -> MissingDescription:
    """Describe relationship between missing values in variable and target variable.

    Args:
        config (Setting): Config of report
        df: (Any): Data, we are exploring.
        target_description (TargetDescription): Description of target column.
    """
    raise NotImplementedError


@multimethod
def missing_bar(config: Settings, df: Any) -> str:
    raise NotImplementedError()


@multimethod
def missing_matrix(config: Settings, df: Any) -> str:
    raise NotImplementedError()


@multimethod
def missing_heatmap(config: Settings, df: Any) -> str:
    raise NotImplementedError()


def get_missing_active(config: Settings, table_stats: dict) -> Dict[Any, Any]:
    """

    Args:
        config: report Settings object
        table_stats: The overall statistics for the DataFrame.

    Returns:

    """
    missing_map = {
        "bar": {
            "min_missing": 0,
            "name": "Count",
            "caption": "A simple visualization of nullity by column.",
            "function": missing_bar,
        },
        "matrix": {
            "min_missing": 0,
            "name": "Matrix",
            "caption": "Nullity matrix is a data-dense display which lets you quickly visually pick out patterns in data completion.",
            "function": missing_matrix,
        },
        "heatmap": {
            "min_missing": 2,
            "name": "Heatmap",
            "caption": "The correlation heatmap measures nullity correlation: how strongly the presence or absence of one variable affects the presence of another.",
            "function": missing_heatmap,
        },
    }

    missing_map = {
        name: settings
        for name, settings in missing_map.items()
        if (
            config.missing_diagrams[name]
            and table_stats["n_vars_with_missing"] >= settings["min_missing"]
        )
        and (
            name != "heatmap"
            or (
                table_stats["n_vars_with_missing"] - table_stats["n_vars_all_missing"]
                >= settings["min_missing"]
            )
        )
    }

    return missing_map


def handle_missing(name: str, fn: Callable) -> Callable:
    def inner(*args, **kwargs) -> Any:
        def warn_missing(missing_name: str, error: str) -> None:
            warnings.warn(
                f"""There was an attempt to generate the {missing_name} missing values diagrams, but this failed.
To hide this warning, disable the calculation
(using `df.profile_report(missing_diagrams={{"{missing_name}": False}}`)
If this is problematic for your use case, please report this as an issue:
https://github.com/ydataai/pandas-profiling/issues
(include the error message: '{error}')"""
            )

        try:
            return fn(*args, *kwargs)
        except ValueError as e:
            warn_missing(name, str(e))

    return inner


def get_missing_diagram(
    config: Settings, df: pd.DataFrame, settings: Dict[str, Any]
) -> Optional[Dict[str, Any]]:
    """Gets the rendered diagrams for missing values.

    Args:
        config: report Settings object
        df: The DataFrame on which to calculate the missing values.
        settings: missing diagram name, caption and function

    Returns:
        A dictionary containing the base64 encoded plots for each diagram that is active in the config (matrix, bar, heatmap).
    """

    if len(df) == 0:
        return None

    result = handle_missing(settings["name"], settings["function"])(config, df)
    if result is None:
        return None

    missing = {
        "name": settings["name"],
        "caption": settings["caption"],
        "matrix": result,
    }

    return missing
