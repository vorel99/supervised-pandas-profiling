from typing import Any, Dict

import pandas as pd
from pandas_profiling.model.model import BaseDataProcessor
from sklearn.preprocessing import OneHotEncoder


class DataProcessorPandas(BaseDataProcessor):
    preprocessed_data: pd.DataFrame

    def __init__(self) -> None:
        super().__init__()

    @property
    def data(self) -> Any:
        return self.preprocessed_data

    def prepare_num(self, col: pd.Series, col_desc: Dict[str, Any]) -> pd.DataFrame:
        return col.to_frame()

    def prepare_cat(self, col: pd.Series, col_desc: Dict[str, Any]) -> pd.DataFrame:
        enc = OneHotEncoder(handle_unknown="ignore")
        transformed = enc.fit_transform(col.to_frame()).toarray()
        self.transformations[col.name] = enc
        new_data = pd.DataFrame(transformed, columns=enc.get_feature_names_out())
        return new_data

    def prepare_text(self, col: pd.Series, col_desc: Dict[str, Any]) -> pd.DataFrame:
        return col.to_frame()

    def _update_preprocessed_data(self, new_data: pd.DataFrame) -> None:
        self.preprocessed_data = self.preprocessed_data.join(new_data, how="outer")

    def fit(self, data: pd.DataFrame, data_desc: Dict[str, Any]) -> None:
        """Train data processor on data and data description.

        Args:
            data (pd.DataFrame): Training data.
            data_desc (dict): Description of train data.
        """
        self.fit_transform(data, data_desc)

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
        self.preprocessed_data = pd.DataFrame()

        for key, val in data_desc.items():
            if key not in data.columns:
                break
            col = data[key]
            self.prepare_column(col, val)
        return self.data

    def _transform_one_col(self, test_col: pd.Series) -> pd.DataFrame:
        """Transform one column with transformer from fit transformer."""
        transformed_col = test_col.to_frame()
        if test_col.name in self.transformations.keys():
            transformer = self.transformations[test_col.name]
            transformed = transformer.transform(test_col.to_frame()).toarray()
            transformed_col = pd.DataFrame(
                transformed, columns=transformer.get_feature_names_out()
            )
        return transformed_col

    def transform(
        self, test_data: pd.DataFrame, data_desc: Dict[str, Any]
    ) -> pd.DataFrame:
        """Transform test data.

        Args:
            test_data (pd.DataFrame): Test data, we want to transform.
            data_desc (dict): Description of test data.

        Returns:
            pd.DataFrame: Transformed test data.
        """
        if self.preprocessed_data is None:
            raise ValueError("You need t fit data first.")

        transformed_test = pd.DataFrame()
        for key, val in data_desc.items():
            if key not in test_data.columns:
                raise ValueError("'{}' is in description, but not in data.".format(key))
            test_col = test_data[key]
            transformed_col = self._transform_one_col(test_col)
            transformed_test = transformed_test.join(transformed_col, how="outer")

        return transformed_test
