from __future__ import annotations

import pandas as pd
from lightgbm import LGBMClassifier
from pandas_profiling.config import Settings
from pandas_profiling.model.description_target import TargetDescription
from pandas_profiling.model.model import (
    Model,
    ModelData,
    ModelEvaluation,
    ModelModule,
    get_model_module,
)
from sklearn import metrics
from sklearn.model_selection import train_test_split


class ModelPandas(Model):
    model: LGBMClassifier

    def __init__(self) -> None:
        self.model = LGBMClassifier(
            max_depth=5, n_estimators=10, num_leaves=10, subsample_for_bin=None
        )

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        self.model.fit(X, y)

    def transform(self, X: pd.DataFrame):
        return self.model.predict(X)


class ModelDataPandas(ModelData):
    model: ModelPandas
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series

    def __init__(self, X_train, X_test, y_train, y_test) -> None:
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.model = ModelPandas()
        self.model.fit(X_train, y_train)
        self.y_pred = self.model.transform(X_test)

    def evaluate(self) -> ModelEvaluation:
        precision = metrics.precision_score(self.y_pred, self.y_pred)
        recall = metrics.recall_score(self.y_pred, self.y_pred)
        f1 = metrics.f1_score(self.y_pred, self.y_pred)
        accuracy = metrics.accuracy_score(self.y_pred, self.y_pred)

        conf_matrix = metrics.confusion_matrix(self.y_pred, self.y_pred)

        return ModelEvaluation(
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1_score=f1,
            confusion_matrix=conf_matrix,
        )

    @classmethod
    def get_model_from_df(
        cls,
        target_description: TargetDescription,
        df: pd.DataFrame,
    ) -> ModelDataPandas:
        X = df.drop(columns=target_description.name)
        y = target_description.series_binary
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=123
        )
        return ModelDataPandas(X_train, X_test, y_train, y_test)


class ModelModulePandas(ModelModule):
    def __init__(
        self,
        config: Settings,
        target_description: TargetDescription,
        df: pd.DataFrame,
    ):
        X = df.drop(columns=target_description.name)
        y = target_description.series_binary
        self.default_model = ModelDataPandas.get_model_from_df(target_description, df)
        self.transformed_model = None


@get_model_module.register
def get_model_module_pandas(
    config: Settings,
    target_description: TargetDescription,
    df: pd.DataFrame,
) -> ModelModule:
    object_cols = df.select_dtypes(include=["object"]).columns
    df[object_cols] = df[object_cols].astype("category")
    return ModelModulePandas(config, target_description, df)
