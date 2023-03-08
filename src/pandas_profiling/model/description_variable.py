from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, List, Optional

import numpy as np
from pandas_profiling.config import Univariate
from pandas_profiling.model.description_target import TargetDescription

import pandas as pd


@dataclass
class VariableDescription(ABC):
    """Base class for variable description.

    Attributes:
        config (Univariate): Setting of variables description.
        data_col (Any): Column with data values.
        data_col_name (str): Name of data column.

    """

    def __init__(self, config: Univariate, **kwargs) -> None:
        """Setup basic parameters for variable description.

        Args:
            config (Univariate): Config of variable description from report config.
        """
        self.config = config
        super().__init__(**kwargs)

    @property
    @abstractmethod
    def data_col(self) -> Any:
        pass

    @property
    @abstractmethod
    def data_col_name(self) -> str:
        pass

    def is_supervised(self) -> bool:
        """Return, if variable description is supervised, or not."""
        return False


class VariableDescriptionSupervised(VariableDescription):
    """Base class for supervised variable description.

    Attributes:
        target_col_name (str): Name of target column.
        p_target_value (int): Positive value of target column.
        n_target_value (int): Negative value of target column.
        target_description (TargetDescription): Description of target column.
    """

    target_description: TargetDescription

    def __init__(
        self, config: Univariate, target_description: TargetDescription, **kwargs
    ) -> None:
        """Setup basic parameters for plot description.s

        Args:
            data_col_name (str): Name of data column.
            data_col (Any): Column with data values.
            target_col_name (str or None): Name of target column.
        """
        self.target_description = target_description
        super().__init__(config=config, **kwargs)

    @property
    def target_col_name(self) -> str:
        return self.target_description.name

    @property
    def p_target_value(self) -> int:
        """Positive binary target value."""
        if self.target_description:
            return self.target_description.bin_positive
        raise ValueError(
            "target description is not defined at '{}' column".format(
                self.data_col_name
            )
        )

    @property
    def n_target_value(self) -> int:
        """Negative binary target value."""
        if self.target_description:
            return self.target_description.bin_negative
        raise ValueError(
            "target description is not defined at '{}' column".format(
                self.data_col_name
            )
        )

    def is_supervised(self) -> bool:
        return True
        return (
            self.target_description is not None
            and self.target_description.name != self.data_col_name
        )


class CatDescription(VariableDescription):
    """Abstract class for categorical unsupervised variable description.

    Attributes:
        distribution (pd.DataFrame): Distribution DataFrame preprocessed for plotting.
    """

    def __init__(self, config: Univariate, **kwargs) -> None:
        super().__init__(config=config, **kwargs)
        distribution = self._generate_distribution()
        self.__validate_distribution(distribution)

    __distribution: pd.DataFrame
    count_col_name: str = "count"

    @property
    def distribution(self) -> pd.DataFrame:
        """Returns preprocessed DataFrame for plotting

        distribution: pd.DataFrame with 2 or 3 columns (data_col, target_col or None, count)
        in format:
            col_name,   target_name,    count
            1           0               10
            1           1               5
            2           0               8
            ..."""
        if self.__distribution is None:
            raise ValueError(
                "Distribution not set at '{}' variable.".format(self.data_col_name)
            )
        return self.__distribution

    @abstractmethod
    def _generate_distribution(self) -> pd.DataFrame:
        """Generate distribution of variable.
        Distribution contains following columns:
            self.data_col_name: column with categories
            self.target_col_name: column with target binary values
            self.count_col_name: column with counts.
        Distribution DataFrame should contain all data_col and target_col combinations,
        even if count is 0.

        Examples:
            col_name,       target_name,    count
            1               0               10
            1               1               5
            2               0               8
            2               1               0
        """
        pass

    def __validate_distribution(self, distribution: pd.DataFrame) -> None:
        """Validate and set distribution DataFrame.
        - check if there are all needed columns
        - if report is supervised, generate log_odds

        Args:
            distribution (pd.DataFrame) : DataFrame, we want to validate.
        """
        if not isinstance(distribution, pd.DataFrame):
            raise ValueError("Preprocessed plot must be pd.DataFrame instance.")
        self._check_columns(distribution)
        self.__distribution = distribution.reset_index(drop=True)

    def _check_columns(self, df: pd.DataFrame):
        """Checks if df contains all columns (data_col, target_col, count_col)."""
        if self.data_col_name not in df:
            raise ValueError(
                "Data column '{}' not in DataFrame.".format(self.data_col_name)
            )
        if self.count_col_name not in df:
            raise ValueError(
                "Count column not in DataFrame. '{}'".format(self.data_col_name)
            )


class CatDescriptionSupervised(VariableDescriptionSupervised, CatDescription):
    """Abstract class for supervised categorical variable description.

    Attributes:
        distribution (pd.DataFrame): Distribution DataFrame preprocessed for plotting.
        log_odds (pd.DataFrame): Log2odds DataFrame preprocessed for plotting.
    """

    __log_odds: pd.DataFrame = None
    log_odds_col_name: str = "log_odds_ratio"
    log_odds_text_col: str = "text_position"

    def __init__(
        self, config: Univariate, target_description: TargetDescription, **kwargs
    ) -> None:
        super().__init__(config=config, target_description=target_description, **kwargs)
        self.__generate_log_odds_ratio()

    @property
    def log_odds(self) -> pd.DataFrame:
        """Returns DataFrame with relative log2odds for data column.
        format:
            col_name,   log_odds
            male        -2
            female      2
        """
        return self.__log_odds

    def __generate_log_odds_ratio(self):
        """Generates log2 odds ratio preprocessed DataFrame based on distribution.
        Compute odds for whole population and for every category.
        From that compute log odds ratio.
        """
        log_odds = pd.pivot_table(
            self.distribution,
            values=self.count_col_name,
            index=self.data_col_name,
            columns=self.target_col_name,
            sort=False,
        ).reset_index()
        log_odds.columns.name = ""

        # there is possibility, that positive, or negative values will not be present
        if not self.p_target_value in log_odds:
            log_odds[self.p_target_value] = 0
        if not self.n_target_value in log_odds:
            log_odds[self.n_target_value] = 0

        # Laplace smoothing for odds
        laplace_smoothing_alpha = self.config.base.log_odds_laplace_smoothing_alpha

        population_odds = (
            log_odds[self.p_target_value].sum() + laplace_smoothing_alpha
        ) / (log_odds[self.n_target_value].sum() + laplace_smoothing_alpha)

        # odds of groups
        _odds_col_name = "odds"
        log_odds[_odds_col_name] = (
            log_odds[self.p_target_value] + laplace_smoothing_alpha
        ) / (log_odds[self.n_target_value] + laplace_smoothing_alpha)

        # odds ratio
        _odds_ratio_col_name = "odds_ratio"
        log_odds[_odds_ratio_col_name] = log_odds[_odds_col_name] / population_odds

        # log odds ratio
        log_odds[self.log_odds_col_name] = np.log2(log_odds[_odds_ratio_col_name])
        log_odds[self.log_odds_col_name] = log_odds[self.log_odds_col_name].round(2)

        # replace all special values with 0
        log_odds.fillna(0, inplace=True)
        log_odds.replace([np.inf, -np.inf], 0, inplace=True)

        # add text position for log2odds
        log_odds[self.log_odds_text_col] = "left"
        log_odds.loc[
            log_odds[self.log_odds_col_name] < 0, self.log_odds_text_col
        ] = "right"
        self.__log_odds = log_odds

    def _check_columns(self, df: pd.DataFrame):
        """Checks if df contains all columns (data_col, target_col, count_col)."""
        if self.data_col_name not in df:
            raise ValueError(
                "Data column '{}' not in DataFrame.".format(self.data_col_name)
            )
        if self.target_description.name not in df:
            raise ValueError(
                "Target column '{}' not in DataFrame.".format(
                    self.target_description.name
                )
            )
        if self.count_col_name not in df:
            raise ValueError(
                "Count column not in DataFrame. '{}'".format(self.data_col_name)
            )


class TextDescription(VariableDescription):
    """Abstract class for unsupervised text variable description.

    Attributes:
        count_col_name (str): Name of column with word counts.
        words_counts (pd.DataFrame):
            Sorted data with words in data_col_name and counts in count_col_name.
    """

    _words_counts: pd.DataFrame

    def __init__(self, config: Univariate, **kwargs) -> None:
        super().__init__(config=config, **kwargs)
        self._words_counts = self.get_word_counts()

    @property
    def count_col_name(self) -> str:
        """Name of column with absolute count of word."""
        return "count"

    @property
    def words_counts(self) -> pd.DataFrame:
        return self._words_counts

    @abstractmethod
    def get_word_counts(self) -> pd.DataFrame:
        """Generate word counts for input series.

        Returns:
            Series with unique words as index and the computed frequency as value.
        """
        pass


class TextDescriptionSupervised(VariableDescriptionSupervised, TextDescription):
    """Abstract class for supervised text variable description.

    Attributes:
        positive_col_name (str): Name of column with word count for positive outcome.
        negative_col_name (str): Name of column with word count for negative outcome.
        words_counts (pd.DataFrame): Sorted words and counts of those words.
    """

    def __init__(
        self, config: Univariate, target_description: TargetDescription, **kwargs
    ) -> None:
        super().__init__(config=config, target_description=target_description, **kwargs)

    @property
    def positive_col_name(self) -> str:
        """Name of column with count of word for positive target."""
        return "positive"

    @property
    def negative_col_name(self) -> str:
        """Name of column with count of word for negative target."""
        return "negative"
