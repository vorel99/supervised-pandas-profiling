from __future__ import annotations

from collections import abc
from dataclasses import dataclass
from typing import Any, Iterator

from ydata_profiling.model.var_description.counts import VarCounts


@dataclass
class VarDescription(VarCounts):
    """Default description for one data column.
    Extends VarCounts class with information about distinct and unique values."""

    var_specific: dict

    def __getitem__(self, item: str):
        """Make the object subscriptable."""
        return self.var_specific[item]

    def __setitem__(self, key: str, value: Any):
        """Make the object subscriptable."""
        self.var_specific[key] = value

    def update(self, _dict: dict) -> None:
        """To support old dict like interface."""
        self.var_specific.update(_dict)

    def items(self) -> abc.ItemsView:
        """To support old dict like interface."""
        return self.var_specific.items()

    def get(self, key: str, default: Any = None) -> Any:
        """To support old dict like interface."""
        return self.var_specific.get(key, default)

    def __iter__(self) -> Iterator:
        """To support old dict like interface."""
        return self.var_specific.__iter__()


@dataclass
class VarDescriptionHashable(VarDescription):
    """Default description for one data column that is hashable (common types).
    Extends VarCounts class with information about distinct and unique values."""

    n_distinct: int | list | None
    """Number of distinct values"""
    p_distinct: float | list | None
    """Proportion of distinct values"""
    is_unique: bool | list | None
    """Whether the variable values are unique"""
    n_unique: int | list | None
    """Number of unique values"""
    p_unique: float | list | None
    """Proportion of unique values"""
