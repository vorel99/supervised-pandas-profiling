from dataclasses import dataclass
from typing import Any


@dataclass
class VarCounts:
    """Data about counts in variable column."""

    n: int
    """Count of rows in the series."""
    count: int
    """Count of not missing rows in the series."""
    n_missing: int
    """Count of missing rows in the series."""
    p_missing: float
    """Proportion of missing rows in the series."""

    hashable: bool
    value_counts_without_nan: Any
    """Counts of values in the series without NaN. Values as index, counts as values."""
    value_counts_index_sorted: Any
    """Sorted counts of values in the series without NaN. Sorted by counts."""
    ordering: bool
    memory_size: int
