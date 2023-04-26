from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, Dict, Hashable, List, Optional

from multimethod import multimethod
from pandas_profiling.config import Settings
from pandas_profiling.model.description_target import TargetDescription
from pandas_profiling.model.model import ModelEvaluation


@dataclass
class TransformationData:
    col_name: str
    X_train: Any
    X_test: Any
    y_train: Any
    y_test: Any
    model_evaluation: ModelEvaluation

    def get_better(self, other: TransformationData) -> TransformationData:
        if self.model_evaluation.quality > other.model_evaluation.quality:
            return self
        return other


class Transformation:
    transformer: Any
    transformation_name: str

    @multimethod
    def fit(self, X: Any):
        raise NotImplementedError

    @multimethod
    def transform(self, X: Any) -> Any:
        raise NotImplementedError

    def supports_nan(self) -> bool:
        return True


class NormalizeTransformation(Transformation):
    transformation_name: str = (
        "Standardize features by removing the mean and scaling to unit variance."
    )


class BinningTransformation(Transformation):
    transformation_name: str = "Bin continuous data into intervals."

    def supports_nan(self) -> bool:
        return False


class OneHotTransformation(Transformation):
    transformation_name: str = "Encode categorical features as a one-hot numeric array."


class TfIdfTransformation(Transformation):
    pass


@multimethod
def get_train_test_split(df: Any, target_description: TargetDescription):
    raise NotImplementedError


@multimethod
def get_best_transformation(
    X_train: Any,
    X_test: Any,
    y_train: Any,
    y_test: Any,
    col_name: str,
    transformations: List[Callable],
) -> TransformationData:
    raise NotImplementedError


def get_transformations_map() -> Dict[str, List[Any]]:
    """Get valid transformations for column type.

    Returns:
        Dict[List[Callable]]: Valid transformations
    """
    return {
        "Numeric": [
            NormalizeTransformation,
            BinningTransformation,
        ],
        # "Text": [
        # TfIdfTransformation,
        # ],
        "Categorical": [
            OneHotTransformation,
        ],
    }


def get_transformations_module(
    config: Settings,
    variables_desc: Dict[str, Any],
    target_desc: TargetDescription,
    df: Any,
) -> List[TransformationData]:
    transformations = []
    transform_map = get_transformations_map()
    X_train, X_test, y_train, y_test = get_train_test_split(df, target_desc)
    for var_name, var_desc in variables_desc.items():
        var_type = var_desc["type"]
        if var_type in transform_map:
            best_transformation = get_best_transformation(
                X_train, X_test, y_train, y_test, var_name, transform_map[var_type]
            )
            if best_transformation is not None:
                transformations.append(best_transformation)

    return transformations
