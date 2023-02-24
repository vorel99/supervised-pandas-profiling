from abc import ABCMeta, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

from multimethod import multimethod
from pandas_profiling.config import Target


@dataclass
class TargetDescription(metaclass=ABCMeta):
    """Description for target column."""

    series: Any
    series_binary: Any
    config: Target
    name: str
    description: Dict[str, Any]
    positive_values: List[str]
    negative_values: List[str]

    def __init__(
        self,
        target_config: Target,
        series: Any,
    ) -> None:
        self.series = series
        self.config = target_config
        self.name = target_config.col_name
        self.description = {}
        self.positive_values, self.negative_values = self._infer_target_values()
        self.series_binary = self._get_bin_target()
        self._update_description_base()

    @abstractmethod
    def _infer_target_values(self) -> Tuple[List[str], List[str]]:
        """Infer positive and negative values in target column.

        Returns
        -------
        positive, negative : Tuple[List[str], List[str]]
            Positive and negative values.
        """
        pass

    @abstractmethod
    def _get_bin_target(self) -> Any:
        """Create binary target from column and positive/negative values.
        Positive values replace with 1, negative with -1."""
        pass

    @abstractmethod
    def _get_advanced_description(self) -> Dict[str, Any]:
        """Update description for target variable.
        Get target mean."""
        pass

    def _update_description_base(self) -> None:
        """Update description.
        Add positive and negative values."""
        _desc = {}
        _desc["positive_vals"] = self.positive_values
        _desc["negative_vals"] = self.negative_values
        _desc.update(self._get_advanced_description())

        self.description.update(_desc)


@multimethod
def describe_target(
    config: Target,
    data_frame: Any,
) -> TargetDescription:
    """Generate target description."""
    raise NotImplementedError()
