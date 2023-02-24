"""Compute statistical description of datasets."""

from typing import Any, Dict, Optional

from multimethod import multimethod
from pandas_profiling.config import Settings
from pandas_profiling.model.description_target import TargetDescription
from pandas_profiling.model.summarizer import BaseSummarizer
from tqdm import tqdm
from visions import VisionsTypeset


@multimethod
def describe_1d(
    config: Settings,
    series: Any,
    summarizer: BaseSummarizer,
    typeset: VisionsTypeset,
    target_description: Optional[TargetDescription],
) -> dict:
    raise NotImplementedError()


@multimethod
def get_series_descriptions(
    config: Settings,
    df: Any,
    summarizer: BaseSummarizer,
    typeset: VisionsTypeset,
    pbar: tqdm,
) -> Dict[str, Any]:
    raise NotImplementedError()
