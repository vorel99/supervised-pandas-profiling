from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, Dict, Hashable


class BaseDataProcessor(ABC):
    col_map: Dict[str, Callable]
    transformations: Dict[Hashable, Any] = {}

    def __init__(self) -> None:
        self.col_map = {
            "Categorical": self.prepare_cat,
            "Numeric": self.prepare_num,
            "Text": self.prepare_text,
        }

    @property
    @abstractmethod
    def data(self) -> Any:
        """Preprocessed train data."""

    @abstractmethod
    def prepare_num(self, col: Any, col_desc: Dict[str, Any]) -> Any:
        """Transform numeric type column.
        Save transformations to self.transformations.

        Args:
            col (Any): Data column.
            col_desc (dict): Description of column.

        Returns:
            preprocessed (Any): Preprocessed numeric data.
        """

    @abstractmethod
    def prepare_cat(self, col: Any, col_desc: Dict[str, Any]) -> Any:
        """Prepare categorical column.
        Apply one-hot encoding.
        Save transformations to self.transformations.

        Args:
            col (Any): Data column.
            col_desc (dict): Description of column.

        Returns:
            preprocessed (Any): Preprocessed categorical data.
        """

    @abstractmethod
    def prepare_text(self, col: Any, col_desc: Dict[str, Any]) -> Any:
        """Prepare text column.
        Save transformations to self.transformations.

        Args:
            col (Any): Data column.
            col_desc (dict): Description of column.

        Returns:
            preprocessed (Any): Preprocessed text data.
        """

    @abstractmethod
    def _update_preprocessed_data(self, new_data: Any) -> None:
        """Join preprocessed data with new data."""

    def prepare_column(self, col: Any, col_desc: Dict[str, Any]):
        """Prepare data column based on column type.
        Add new data to preprocessed data.

        Args:
            col (Any): Data of column.
            col_desc (Dict):  Description of column.
        """
        # call prepare function by column type
        col_type = col_desc["type"]
        prepare_func = self.col_map[col_type]
        data = prepare_func(col, col_desc)
        self._update_preprocessed_data(data)

    @abstractmethod
    def fit(self, data: Any, data_desc: Dict[str, Any]) -> None:
        """Train data processor on data."""

    @abstractmethod
    def fit_transform(self, data: Any, data_desc: Dict[str, Any]) -> Any:
        """Train data processor on data and return preprocessed data."""

    @abstractmethod
    def transform(self, test_data: Any):
        """Transform test data same way as train data."""


class BaseModel:
    data_preprocessing: BaseDataProcessor

    @property
    def data(self) -> Any:
        return self.data_preprocessing.data
