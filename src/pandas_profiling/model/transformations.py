from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Hashable, List, Optional

from multimethod import multimethod

from pandas_profiling.config import Settings
from pandas_profiling.model.description_target import TargetDescription


class Transformer(ABC):
    """Abstract class for transformers."""

    @abstractmethod
    def fit(self, col: Any, col_desc: Dict[str, Any]):
        """Fit col.

        Args:
            col (Any): Column with data to fit.
            col_desc (Dict[str, Any]): Description of col.
        """

    @abstractmethod
    def transform(self, col: Any, col_desc: Dict[str, Any]) -> Any:
        """Transform col.

        Args:
            col (Any): Column with data to transform.
            col_desc (Dict[str, Any]): Description of col.

        Returns:
            Any: Transformed col.
        """

    @abstractmethod
    def fit_transform(self, col: Any, col_desc: Dict[str, Any]) -> Any:
        """Fit and transform col.

        Args:
            col (Any): Column with data to fit and transform.
            col_desc (Dict[str, Any]): Description of col.
        Returns:
            Any: Transformed col.
        """


def normalize_transformation() -> Any:
    pass


def binning_transformation() -> Any:
    pass


def one_hot_transformation() -> Any:
    pass


def tf_idf_transformation() -> Any:
    pass


@dataclass
class TransformationDescription:
    original_data: Any
    new_data: Any
    used_transformation: Transformer


class TransformationsModule:
    transformations: List[TransformationDescription]
