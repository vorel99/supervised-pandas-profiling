from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, Dict


@dataclass
class BaseDataPreprocessing(ABC):
    col_map: Dict[str, Callable]

    def __init__(self) -> None:
        self.col_map = {
            "Categorical": self.prepare_cat,
            "Numeric": self.prepare_num,
            "Text": self.prepare_text,
        }

    @property
    @abstractmethod
    def data(self) -> Any:
        """Preprocessed data."""

    @abstractmethod
    def prepare_num(self, col: Any, col_desc: Dict[str, Any]) -> Any:
        """Prepare numeric type column."""

    @abstractmethod
    def prepare_cat(self, col: Any, col_desc: Dict[str, Any]) -> Any:
        """Prepare categorical column.
        Apply one-hot encoding."""

    @abstractmethod
    def prepare_text(self, col: Any, col_desc: Dict[str, Any]) -> Any:
        """Prepare text column."""

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
        func = self.col_map[col_type]
        data = func(col, col_desc)
        self._update_preprocessed_data(data)
