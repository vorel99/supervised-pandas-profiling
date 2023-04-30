from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple

from multimethod import multimethod

from pandas_profiling.config import Model as ModelConfig
from pandas_profiling.config import Settings
from pandas_profiling.model.data import ConfMatrixData
from pandas_profiling.model.description_target import TargetDescription


@multimethod
def get_train_test_split(
    model_config: ModelConfig, df: Any, target_description: TargetDescription
) -> tuple[Any, Any, Any, Any]:
    """Split data to train and test subset

    Args:
        model_config (ModelConfig): Config of model containing seed.
        df (Any): DataFrame
        target_description (TargetDescription): Description of target variable.

    Returns:
        tuple[Any, Any, Any, Any]: X_train, X_test, y_train, y_test
    """
    raise NotImplementedError


@dataclass
class ModelEvaluation:
    """Class for data from model evaluations."""

    accuracy: float
    precision: float
    recall: float
    f1_score: float
    confusion_matrix: ConfMatrixData

    def get_evaluation_metric(self, config: Settings) -> float:
        match config.model.evaluation_metric:
            case "accuracy":
                return self.accuracy
            case "precision":
                return self.precision
            case "recall":
                return self.recall
            case "f1_score":
                return self.f1_score


class Model(ABC):
    """Abstract class for models."""

    @abstractmethod
    def __init__(self) -> None:
        """Model creation."""

    @abstractmethod
    def fit(self, X, y):
        pass

    @abstractmethod
    def transform(self, X):
        pass


class ModelData(ABC):
    """Data about model"""

    X_train: Any
    X_test: Any
    y_train: Any
    y_test: Any

    train_test_split_policy: str
    train_records: int
    test_records: int
    n_of_features: int

    boosting_type: str
    model_source: str

    @abstractmethod
    def evaluate(self) -> ModelEvaluation:
        """Evaluate model.

        Returns:
            ModelEvaluation: evaluation of model
        """

    @abstractmethod
    def get_feature_importances() -> List[Tuple[float, str]]:
        """Get feature importances for model training.

        Returns:
            Tuple[float, str]: Tuple of feature  importances and feature names.
        """


@dataclass
class ModelModule:
    default_model: ModelData
    transformed_model: Optional[ModelData]


@multimethod
def get_model_data(
    config: Settings, X_train: Any, X_test: Any, y_train: Any, y_test: Any
) -> ModelData:
    raise NotImplementedError()
