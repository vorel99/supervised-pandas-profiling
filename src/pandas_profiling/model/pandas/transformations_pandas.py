from typing import Any, Callable, Dict, Hashable, List

import pandas as pd
from pandas_profiling.config import Settings
from pandas_profiling.model.description import BaseDescription
from pandas_profiling.model.description_target import TargetDescription
from pandas_profiling.model.pandas.model_pandas import ModelDataPandas
from pandas_profiling.model.transformations import (  # Transformation,; TransformationsModule,; one_hot_transformation,; tf_idf_transformation,
    BinningTransformation,
    NormalizeTransformation,
    OneHotTransformation,
    TfIdfTransformation,
    Transformation,
    TransformationData,
    get_best_transformation,
    get_train_test_split,
)
from sklearn import metrics
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import KBinsDiscretizer, OneHotEncoder, StandardScaler

# class TfIdfTransformerPandas(TransformationPandas):
#     """Transformer for category type variables.
#     - use tf-idf transformation
#     """

#     transformer: TfidfVectorizer
#     significant_words: List[str]

#     def __init__(self) -> None:
#         self.transformer = TfidfVectorizer(token_pattern=r"(?u)\b[\w\./]+\b")

#     def _add_prefix(self, df: pd.DataFrame, name: str) -> pd.DataFrame:
#         return df.add_prefix(name + "_")

#     def fit(self, col: pd.Series, col_desc: Dict[str, Any]) -> None:
#         text_logodds: pd.DataFrame = col_desc["plot_description"].log_odds
#         # TODO replace constant .5
#         self.significant_words = text_logodds[
#             text_logodds["log_odds_ratio"].abs() > 0.5
#         ].index.to_list()

#         # get tf-idf matrix of words
#         self.transformer.fit(col.values.astype(str))

#     def transform(self, col: pd.Series, col_desc: Dict[str, Any]) -> pd.DataFrame:
#         transformed = self.transformer.transform(col.values.astype(str)).toarray()
#         data = pd.DataFrame(
#             transformed, columns=self.transformer.get_feature_names_out()
#         )
#         data = data[self.significant_words]
#         return self._add_prefix(data, col.name)


# NormalizeTransformation ==============================================================
# can handle nan
@NormalizeTransformation.fit.register
def fit_normalize_transform_pandas(self: NormalizeTransformation, X: pd.DataFrame):
    self.transformer = StandardScaler()
    self.transformer.fit(X)


@NormalizeTransformation.transform.register
def transform_normalize_transform_pandas(
    self: NormalizeTransformation, X: pd.DataFrame
):
    return pd.DataFrame(self.transformer.transform(X), index=X.index)


# BinningTransformation ================================================================
# cannot handle nan
@BinningTransformation.fit.register
def fit_binning_transform_pandas(self: BinningTransformation, X: pd.DataFrame):
    self.transformer = KBinsDiscretizer(encode="ordinal", strategy="quantile")
    self.transformer.fit(X)


@BinningTransformation.transform.register
def transform_binning_transform_pandas(self: BinningTransformation, X: pd.DataFrame):
    return pd.DataFrame(self.transformer.transform(X), index=X.index, columns=X.columns)


# OneHotTransformation =================================================================
@OneHotTransformation.fit.register
def fit_one_hot_transform_pandas(self: OneHotTransformation, X: pd.DataFrame):
    self.transformer = OneHotEncoder(handle_unknown="ignore")
    self.transformer.fit(X)


@OneHotTransformation.transform.register
def transform_one_hot_transform_pandas(self: OneHotTransformation, X: pd.DataFrame):
    return pd.DataFrame(
        self.transformer.transform(X).toarray(),
        index=X.index,
        columns=self.transformer.get_feature_names_out(),
    )


# @one_hot_transformation.register
# def one_hot_transformation_pandas(series: pd.Series) -> pd.DataFrame:
#     transformer = OneHotEncoder(handle_unknown="ignore")
#     transformer.fit(series.to_frame())
#     series_transformed = transformer.transform(series.to_frame()).toarray()
#     return pd.DataFrame(series_transformed, columns=transformer.get_feature_names_out())


# @tf_idf_transformation.register
# def tf_idf_transformation_pandas(series: pd.Series) -> pd.DataFrame:
#     transformer = TfidfVectorizer(token_pattern=r"(?u)\b[\w\./]+\b")
#     transformer.fit(series.values.astype(str))
#     series_transformed = transformer.transform(series.values.astype(str)).toarray()
#     return pd.DataFrame(series_transformed, columns=transformer.get_feature_names_out())


@get_train_test_split.register
def get_train_test_split_pandas(
    df: pd.DataFrame, target_description: TargetDescription
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    object_cols = df.select_dtypes(include=["object"]).columns
    df[object_cols] = df[object_cols].astype("category")

    X = df.drop(columns=target_description.name)
    y = target_description.series_binary
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=123
    )
    return X_train, X_test, y_train, y_test


@get_best_transformation.register
def get_best_transformation_pandas(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    col_name: str,
    transformations: List[Callable],
) -> TransformationData:
    best_transform = None

    for transform_class in transformations:
        transformer: Transformation = transform_class()
        if transformer.supports_nan():
            transformer.fit(X_train[col_name].to_frame())
            transformed_train = transformer.transform(X_train[col_name].to_frame())
            transformed_test = transformer.transform(X_test[col_name].to_frame())

            transformed_train = pd.concat(
                [X_train.drop(columns=col_name), transformed_train], axis=1
            )
            transformed_test = pd.concat(
                [X_test.drop(columns=col_name), transformed_test], axis=1
            )
            model_eval = ModelDataPandas(
                transformed_train, transformed_test, y_train, y_test
            ).evaluate()
            transformation = TransformationData(
                col_name=col_name,
                X_train=transformed_train,
                X_test=transformed_test,
                y_train=y_train,
                y_test=y_test,
                model_evaluation=model_eval,
                transform_name=transformer.transformation_name,
            )
    return transformation
