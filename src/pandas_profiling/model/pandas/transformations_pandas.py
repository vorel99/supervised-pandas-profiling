from typing import Callable, List, Optional

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import (
    FunctionTransformer,
    KBinsDiscretizer,
    OneHotEncoder,
    StandardScaler,
)

from pandas_profiling.config import Settings
from pandas_profiling.model.pandas.model_pandas import ModelDataPandas
from pandas_profiling.model.transformations import (
    BinningTransformation,
    LogTransformation,
    NormalizeTransformation,
    OneHotTransformation,
    TfIdfTransformation,
    Transformation,
    TransformationData,
    get_best_transformation,
)


# NormalizeTransformation ==============================================================
# can handle nan
@NormalizeTransformation.fit.register
def fit_normalize_transform_pandas(self: NormalizeTransformation, X: pd.Series):
    self.transformer = StandardScaler()
    self.transformer.fit(X.to_frame())


@NormalizeTransformation.transform.register
def transform_normalize_transform_pandas(self: NormalizeTransformation, X: pd.Series):
    return pd.DataFrame(
        self.transformer.transform(X.to_frame()), index=X.index, columns=[X.name]
    )


# LogTransformation ================================================================
# cannot handle negative
@LogTransformation.fit.register
def fit_log_transform_pandas(self: LogTransformation, X: pd.Series):
    transformer = FunctionTransformer(np.log1p)
    self.transformer = transformer
    self.transformer.fit(X.to_frame())


@LogTransformation.transform.register
def transform_log_transform_pandas(self: LogTransformation, X: pd.Series):
    return pd.DataFrame(
        self.transformer.transform(X.to_frame()), index=X.index, columns=[X.name]
    )


# BinningTransformation ================================================================
# cannot handle nan
@BinningTransformation.fit.register
def fit_binning_transform_pandas(self: BinningTransformation, X: pd.Series):
    self.transformer = KBinsDiscretizer(
        encode="ordinal", strategy="quantile", random_state=self.seed
    )
    self.transformer.fit(X.to_frame())


@BinningTransformation.transform.register
def transform_binning_transform_pandas(self: BinningTransformation, X: pd.Series):
    return pd.DataFrame(
        self.transformer.transform(X.to_frame()), index=X.index, columns=[X.name]
    )


# OneHotTransformation =================================================================
@OneHotTransformation.fit.register
def fit_one_hot_transform_pandas(self: OneHotTransformation, X: pd.Series):
    self.transformer = OneHotEncoder(handle_unknown="ignore")
    self.transformer.fit(X.to_frame())


@OneHotTransformation.transform.register
def transform_one_hot_transform_pandas(self: OneHotTransformation, X: pd.Series):
    out_features = [str(item) for item in self.transformer.get_feature_names_out()]

    return pd.DataFrame(
        self.transformer.transform(X.to_frame()).toarray(),
        index=X.index,
        columns=out_features,
    )


# TfIdfTransformation ==================================================================
@TfIdfTransformation.fit.register
def fit_tf_idf_transform_pandas(self: TfIdfTransformation, X: pd.Series):
    self.transformer = TfidfVectorizer(
        token_pattern=r"(?u)\b[\w\./]+\b", stop_words="english", max_features=50
    )
    # text_logodds: pd.DataFrame = col_desc["plot_description"].log_odds
    # # TODO replace constant .5
    # self.significant_words = text_logodds[
    #     text_logodds["log_odds_ratio"].abs() > 0.5
    # ].index.to_list()

    # get tf-idf matrix of words
    self.transformer.fit(X)


@TfIdfTransformation.transform.register
def transform_tf_idf_transform_pandas(self: TfIdfTransformation, X: pd.Series):
    transformed = self.transformer.transform(X).toarray()
    out_features = [str(item) for item in self.transformer.get_feature_names_out()]
    data = pd.DataFrame(transformed, index=X.index, columns=out_features)
    # data = data[self.significant_words]
    return data.add_prefix("{}_".format(X.name))


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
        transformer: Transformation = transform_class(config.model.model_seed)
        train_col = X_train[col_name]
        test_col = X_test[col_name]
        # if data contains nan and transformation doesn't support nan, skip
        if (
            train_col.isnull().any() or test_col.isnull().any()
        ) and not transformer.supports_nan():
            continue
        # if data contains negative values and transformation doesn't support negative
        if (
            is_numeric_dtype(train_col)
            and (any(train_col < 0) or any(test_col < 0))
            and not transformer.supports_negative()
        ):
            continue
        transformer.fit(train_col)
        transformed_train = transformer.transform(train_col)
        transformed_test = transformer.transform(test_col)

        transformed_train = pd.concat(
            [X_train.drop(columns=col_name), transformed_train], axis=1
        )
        transformed_test = pd.concat(
            [X_test.drop(columns=col_name), transformed_test], axis=1
        )
        model_data = ModelDataPandas(
            config, transformed_train, transformed_test, y_train, y_test
        )
        transformation = TransformationData(
            col_name=col_name,
            X_train=transformed_train,
            X_test=transformed_test,
            y_train=y_train,
            y_test=y_test,
            model_data=model_data,
            transform_name=transformer.transformation_name,
            transform_desc=transformer.transformation_description,
        )

        if best_transform is None:
            best_transform = transformation
        best_transform = best_transform.get_better(transformation, config)
    return best_transform
