from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Hashable, Optional

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


class BaseDataProcessor(ABC):
    transformations: Dict[Hashable, Transformer] = {}

    @abstractmethod
    def fit(self, data: Any, data_desc: Dict[str, Any]) -> None:
        """Train data processor on data."""

    @abstractmethod
    def fit_transform(self, data: Any, data_desc: Dict[str, Any]) -> Any:
        """Train data processor on data and return preprocessed data."""

    @abstractmethod
    def transform(self, test_data: Any) -> Any:
        """Transform test data same way as train data."""


@dataclass
class ModelEvaluation:
    """Class for data from model evaluations."""

    accuracy: float
    precision: float
    recall: float
    f1_score: float
    confusion_matrix: list


class Model(ABC):
    """Abstract class for models."""

    @abstractmethod
    def __init__(self, X: Any, y: Any) -> None:
        """Model creation.
        Train and test model.

        Args:
            X (Any): Features
            y (Any): Target
        """
        pass

    @abstractmethod
    def evaluate(self) -> ModelEvaluation:
        """Evaluate model.

        Returns:
            ModelEvaluation: evaluation of model
        """


@dataclass
class ModelModule:
    default_model: Model
    transformed_model: Optional[Model]


@multimethod
def get_model_module(
    config: Settings, target_description: TargetDescription, df: Any
) -> ModelModule:
    raise NotImplementedError()
