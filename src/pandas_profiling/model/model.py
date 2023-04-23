from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Optional

from multimethod import multimethod

from pandas_profiling.config import Settings
from pandas_profiling.model.description_target import TargetDescription


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
