import pandas as pd
from lightgbm import LGBMClassifier
from sklearn import metrics
from sklearn.model_selection import train_test_split

from pandas_profiling.config import Settings
from pandas_profiling.model.description_target import TargetDescription
from pandas_profiling.model.model import (
    Model,
    ModelEvaluation,
    ModelModule,
    get_model_module,
)


class ModelPandas(Model):
    X: pd.DataFrame
    y: pd.Series
    model: LGBMClassifier

    def __init__(self, X: pd.DataFrame, y: pd.Series) -> None:
        if X.shape[0] != y.shape[0]:
            raise ValueError(
                "Cannot create model, X and y data should have same length."
            )
        self.X = X
        self.y = y
        self._train()

    def _train(self):
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y, test_size=0.25, random_state=123
        )
        self.model = LGBMClassifier(
            max_depth=5, n_estimators=10, num_leaves=10, subsample_for_bin=None
        )
        self.model.fit(X_train, y_train)
        self.real_y = y_test
        self.predicted_y = self.model.predict(X_test)

    def evaluate(self) -> ModelEvaluation:
        precision = metrics.precision_score(self.real_y, self.predicted_y)
        recall = metrics.recall_score(self.real_y, self.predicted_y)
        f1 = metrics.f1_score(self.real_y, self.predicted_y)
        accuracy = metrics.accuracy_score(self.real_y, self.predicted_y)

        conf_matrix = metrics.confusion_matrix(self.real_y, self.predicted_y)

        return ModelEvaluation(
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1_score=f1,
            confusion_matrix=conf_matrix,
        )


class ModelModulePandas(ModelModule):
    def __init__(
        self,
        config: Settings,
        target_description: TargetDescription,
        df: pd.DataFrame,
    ):
        X = df.drop(columns=target_description.name)
        y = target_description.series_binary
        self.default_model = ModelPandas(X, y)
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
