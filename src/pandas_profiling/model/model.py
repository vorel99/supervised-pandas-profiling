from abc import ABC, abstractmethod
from typing import Any, Dict, Hashable


class Transformer(ABC):
    """Abstract class for transformers."""

    @abstractmethod
    def fit(self, col: Any, col_desc: Dict[str, Any]):
        """Fit col.

        Args:
            col (Any): Column with data to fit.
            col_desc (Dict[str, Any]): Description of col.
        """

    @abstractmethod
    def transform(self, col: Any, col_desc: Dict[str, Any]) -> Any:
        """Transform col.

        Args:
            col (Any): Column with data to transform.
            col_desc (Dict[str, Any]): Description of col.

        Returns:
            Any: Transformed col.
        """

    @abstractmethod
    def fit_transform(self, col: Any, col_desc: Dict[str, Any]) -> Any:
        """Fit and transform col.

        Args:
            col (Any): Column with data to fit and transform.
            col_desc (Dict[str, Any]): Description of col.
        Returns:
            Any: Transformed col.
        """


class BaseDataProcessor(ABC):
    transformations: Dict[Hashable, Transformer] = {}

    @abstractmethod
    def fit(self, data: Any, data_desc: Dict[str, Any]) -> None:
        """Train data processor on data."""

    @abstractmethod
    def fit_transform(self, data: Any, data_desc: Dict[str, Any]) -> Any:
        """Train data processor on data and return preprocessed data."""

    @abstractmethod
    def transform(self, test_data: Any) -> Any:
        """Transform test data same way as train data."""
