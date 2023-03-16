from typing import Any, Dict

import pandas as pd
from pandas_profiling.model.model import BaseDataPreprocessing
from sklearn.preprocessing import OneHotEncoder


class DataPreprocessingPandas(BaseDataPreprocessing):
    preprocessed_data: pd.DataFrame
    transformations: Dict[str, Any] = {}

    def __init__(self, data: pd.DataFrame, vars_desc: Dict[str, Any]) -> None:
        super().__init__()
        self.preprocessed_data = pd.DataFrame()

        for key, val in vars_desc.items():
            col = data[key]
            self.prepare_column(col, val)

    @property
    def data(self) -> Any:
        return self.preprocessed_data

    def prepare_num(self, col: pd.Series, col_desc: Dict[str, Any]) -> pd.DataFrame:
        return col.to_frame()

    def prepare_cat(self, col: pd.Series, col_desc: Dict[str, Any]) -> pd.DataFrame:
        enc = OneHotEncoder(handle_unknown="ignore")
        transformed = enc.fit_transform(col.to_frame()).toarray()
        new_data = pd.DataFrame(transformed, columns=enc.get_feature_names_out())
        return new_data

    def prepare_text(self, col: pd.Series, col_desc: Dict[str, Any]) -> pd.DataFrame:
        return col.to_frame()

    def _update_preprocessed_data(self, new_data: pd.DataFrame) -> None:
        self.preprocessed_data = self.preprocessed_data.join(new_data, how="outer")
