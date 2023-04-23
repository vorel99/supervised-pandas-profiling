from abc import ABC, abstractmethod
from typing import Any, Dict, Hashable, List

import pandas as pd
from lightgbm import LGBMClassifier
from sklearn import metrics
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

from pandas_profiling.config import Settings
from pandas_profiling.model.description import BaseDescription
from pandas_profiling.model.description_target import TargetDescription
from pandas_profiling.model.model import (
    BaseDataProcessor,
    Model,
    ModelEvaluation,
    ModelModule,
    get_model_module,
)


class TransformerPandas(ABC):
    @abstractmethod
    def fit(self, col: pd.Series, col_desc: Dict[str, Any]) -> None:
        pass

    @abstractmethod
    def transform(self, col: pd.Series, col_desc: Dict[str, Any]) -> pd.DataFrame:
        pass

    @abstractmethod
    def fit_transform(self, col: pd.Series, col_desc: Dict[str, Any]) -> pd.DataFrame:
        pass


class NumTransformerPandas(TransformerPandas):
    """Transformer for numeric type variables."""

    def fit(self, col: pd.Series, col_desc: Dict[str, Any]) -> None:
        pass

    def transform(self, col: pd.Series, col_desc: Dict[str, Any]) -> pd.DataFrame:
        return col.to_frame()

    def fit_transform(self, col: pd.Series, col_desc: Dict[str, Any]) -> pd.DataFrame:
        self.fit(col, col_desc)
        return self.transform(col, col_desc)


class CatTransformerPandas(TransformerPandas):
    """Transformer for category type variables.
    - use one-hot encoding
    """

    transformer: OneHotEncoder

    def fit(self, col: pd.Series, col_desc: Dict[str, Any]) -> None:
        self.transformer = OneHotEncoder(handle_unknown="ignore")
        self.transformer.fit(col.to_frame())

    def transform(self, col: pd.Series, col_desc: Dict[str, Any]) -> pd.DataFrame:
        transformed = self.transformer.transform(col.to_frame()).toarray()
        return pd.DataFrame(
            transformed, columns=self.transformer.get_feature_names_out()
        )

    def fit_transform(self, col: pd.Series, col_desc: Dict[str, Any]) -> pd.DataFrame:
        self.fit(col, col_desc)
        return self.transform(col, col_desc)


class TextTransformerPandas(TransformerPandas):
    """Transformer for category type variables.
    - use tf-idf transformation
    """

    transformer: TfidfVectorizer
    significant_words: List[str]

    def _add_prefix(self, df: pd.DataFrame, name: str) -> pd.DataFrame:
        return df.add_prefix(name + "_")

    def fit(self, col: pd.Series, col_desc: Dict[str, Any]) -> None:
        text_logodds: pd.DataFrame = col_desc["plot_description"].log_odds
        # TODO replace constant .5
        self.significant_words = text_logodds[
            text_logodds["log_odds_ratio"].abs() > 0.5
        ].index.to_list()

        self.transformer = TfidfVectorizer(token_pattern=r"(?u)\b[\w\./]+\b")
        # get tf-idf matrix of words
        self.transformer.fit(col.values.astype(str))

    def transform(self, col: pd.Series, col_desc: Dict[str, Any]) -> pd.DataFrame:
        transformed = self.transformer.transform(col.values.astype(str)).toarray()
        data = pd.DataFrame(
            transformed, columns=self.transformer.get_feature_names_out()
        )
        data = data[self.significant_words]
        return self._add_prefix(data, col.name)

    def fit_transform(self, col: pd.Series, col_desc: Dict[str, Any]) -> pd.DataFrame:
        self.fit(col, col_desc)
        return self.transform(col, col_desc)


class DataProcessorPandas(BaseDataProcessor):
    transform_map: Dict[str, TransformerPandas]
    transformations: Dict[Hashable, TransformerPandas] = {}

    def __init__(self) -> None:
        self.transform_map = {
            "Categorical": CatTransformerPandas,
            "Numeric": NumTransformerPandas,
            "Text": TextTransformerPandas,
        }

    def fit(self, data: pd.DataFrame, data_desc: Dict[str, Any]) -> None:
        """Train data processor on data and data description.

        Args:
            data (pd.DataFrame): Training data.
            data_desc (dict): Description of train data.
        """
        for col_name, col_desc in data_desc.items():
            if col_name not in data.columns:
                continue
            col = data[col_name]
            col_type = col_desc["type"]
            transformer = self.transform_map[col_type]()
            transformer.fit(col, col_desc)
            self.transformations[col_name] = transformer

    def transform(self, data: pd.DataFrame, data_desc: Dict[str, Any]) -> pd.DataFrame:
        """Transform test data.

        Args:
            test_data (pd.DataFrame): Test data, we want to transform.
            data_desc (dict): Description of test data.

        Returns:
            pd.DataFrame: Transformed test data.
        """

        transformed_test = pd.DataFrame()
        for col_name, col_desc in data_desc.items():
            if col_name not in data.columns:
                continue
            test_col = data[col_name]
            transformed_col = self.transformations[col_name].transform(
                test_col, col_desc
            )
            transformed_test = transformed_test.join(transformed_col, how="outer")

        return transformed_test

    def fit_transform(
        self, data: pd.DataFrame, data_desc: Dict[str, Any]
    ) -> pd.DataFrame:
        """Train data processor on data and data description.

        Args:
            data (pd.DataFrame): Training data.
            data_desc (dict): Description of train data.

        Returns:
            pd.DataFrame: Preprocessed train data.
        """
        self.fit(data, data_desc)
        return self.transform(data, data_desc)


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
