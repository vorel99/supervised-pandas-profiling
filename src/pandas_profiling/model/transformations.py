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
    train_transformed_col: Any
    test_transformed_col: Any
    y_train: Any
    y_test: Any
    model_evaluation: ModelEvaluation

    def get_better(self, other: TransformationData) -> TransformationData:
        if self.model_evaluation.quality > other.model_evaluation.quality:
            return self
        return other


class Transformation:
    transformer: Any

    @multimethod
    def fit(self, X: Any):
        raise NotImplementedError

    @multimethod
    def transform(self, X: Any) -> Any:
        raise NotImplementedError


class NormalizeTransformation(Transformation):
    pass


class BinningTransformation(Transformation):
    pass


class OneHotTransformation(Transformation):
    pass


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


def get_transformations_map() -> Dict[List[Callable]]:
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
        # "Categorical": [
        # OneHotTransformation,
        # ],
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
            transformations.append(
                get_best_transformation(
                    X_train, X_test, y_train, y_test, var_name, transform_map[var_type]
                )
            )

    return transformations
