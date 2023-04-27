from typing import Any, Callable, Dict, Hashable, List, Optional

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
    return pd.DataFrame(self.transformer.transform(X), index=X.index, columns=X.columns)


# BinningTransformation ================================================================
# cannot handle nan
@BinningTransformation.fit.register
def fit_binning_transform_pandas(self: BinningTransformation, X: pd.DataFrame):
    self.transformer = KBinsDiscretizer(
        encode="ordinal", strategy="quantile", random_state=self.seed
    )
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
    ).add_prefix("{}_".format(X.columns[0]))


# TfIdfTransformation ==================================================================
@TfIdfTransformation.fit.register
def fit_otf_idf_transform_pandas(self: TfIdfTransformation, X: pd.DataFrame):
    self.transformer = TfidfVectorizer(token_pattern=r"(?u)\b[\w\./]+\b")
    # text_logodds: pd.DataFrame = col_desc["plot_description"].log_odds
    # # TODO replace constant .5
    # self.significant_words = text_logodds[
    #     text_logodds["log_odds_ratio"].abs() > 0.5
    # ].index.to_list()

    # get tf-idf matrix of words
    self.transformer.fit(X)


@TfIdfTransformation.transform.register
def transform_tf_idf_transform_pandas(self: TfIdfTransformation, X: pd.DataFrame):
    transformed = self.transformer.transform(X).toarray()
    data = pd.DataFrame(transformed, columns=self.transformer.get_feature_names_out())
    # data = data[self.significant_words]
    return data.add_prefix("{}_".format(X.columns[0]))


@get_train_test_split.register
def get_train_test_split_pandas(
    seed: int, df: pd.DataFrame, target_description: TargetDescription
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    object_cols = df.select_dtypes(include=["object"]).columns
    df[object_cols] = df[object_cols].astype("category")

    X = df.drop(columns=target_description.name)
    y = target_description.series_binary
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=seed
    )
    return X_train, X_test, y_train, y_test


@get_best_transformation.register
def get_best_transformation_pandas(
    config: Settings,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    col_name: str,
    transformations: List[Callable],
) -> Optional[TransformationData]:
    best_transform = None

    for transform_class in transformations:
        transformer: Transformation = transform_class(config.model_seed)
        # if data contains nan and transformation doesn't support nan, skip
        if (
            X_train.isnull().values.any() or X_test.isnull().values.any()
        ) and not transformer.supports_nan():
            continue
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
            config, transformed_train, transformed_test, y_train, y_test
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

        if best_transform is None:
            best_transform = transformation
        best_transform = best_transform.get_better(transformation)
    return best_transform
