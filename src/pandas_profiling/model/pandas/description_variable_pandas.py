import string
from typing import List

from pandas_profiling.config import Univariate
from pandas_profiling.model.description_variable import (
    CatDescription,
    CatDescriptionSupervised,
    TextDescription,
    TextDescriptionSupervised,
    VariableDescription,
    VariableDescriptionSupervised,
)
from pandas_profiling.model.pandas.description_target_pandas import (
    TargetDescriptionPandas,
)

import pandas as pd


class VariableDescriptionPandas(VariableDescription):
    """Base class for pandas plot description.
    Specify type of data_col.

    Attributes:
        data_col (pd.Series): Series with data values.
        data_col_name (str): Name of column with data.
    """

    _data_col_name: str
    _data_col: pd.Series

    def __init__(self, config: Univariate, data_col: pd.Series, **kwargs) -> None:
        self._data_col_name = self._prepare_data_col_name(data_col)
        self._data_col = data_col

        super().__init__(config=config, **kwargs)

    @property
    def data_col(self) -> pd.Series:
        return self._data_col

    @property
    def data_col_name(self) -> str:
        return self._data_col_name

    @staticmethod
    def _prepare_data_col_name(data_col: pd.Series) -> str:
        """Fills col name, if None.

        Returns column name
        """
        if data_col.name is None:
            data_col.name = "data_col"
        return str(data_col.name)


class VariableDescriptionSupervisedPandas(
    VariableDescriptionSupervised, VariableDescriptionPandas
):
    """Base class for pandas plot description.
    Specify type of data_col and target_description.

    Attributes:
        target_description (TargetDescriptionPandas):
            Description of pandas target series.
    """

    target_description: TargetDescriptionPandas

    def __init__(
        self,
        config: Univariate,
        data_col: pd.Series,
        target_description: TargetDescriptionPandas,
        **kwargs
    ) -> None:
        super().__init__(
            config=config,
            target_description=target_description,
            data_col=data_col,
            **kwargs
        )


class CatDescriptionPandas(VariableDescriptionPandas, CatDescription):
    """Class for unsupervised categorical pandas variable description."""

    _other_placeholder: str = "other ..."
    _max_cat_to_plot: int

    def __init__(self, config: Univariate, data_col: pd.Series, **kwargs) -> None:
        """Prepare categorical data for plotting

        Args:
            config (Univariate): Config of variables from report setting.
            data_col (pd.Series): Series with data, from processed column.
        """
        self._max_cat_to_plot = config.cat.n_obs
        data_col = data_col.astype(str)
        super().__init__(config=config, data_col=data_col, **kwargs)

    def _limit_count(self, df: pd.DataFrame) -> pd.DataFrame:
        """Limit count of displayed categories to max_cat.
        All other categories groups to one category 'other'.

        Args:
            df (pd.DataFrame): Categories with counts, we would like to limit size of.
        """
        top_n_classes = df.drop_duplicates(self.data_col_name)[self.data_col_name].head(
            self._max_cat_to_plot
        )
        if top_n_classes.size < df[self.data_col_name].nunique():
            # select rows, that are not in top n classes and group them
            other = df[~df[self.data_col_name].isin(top_n_classes)]

            sum = other[self.count_col_name].sum()
            other = pd.DataFrame(
                data={
                    self.count_col_name: [sum],
                    self.data_col_name: [self._other_placeholder],
                }
            )
            # drop all categories, that are not in top_n_categories
            df = df[df[self.data_col_name].isin(top_n_classes)]
            # merge top n categories and other
            df = pd.concat([df, other])
        return df

    def _generate_distribution(self) -> pd.DataFrame:
        """Generate grouped distribution DataFrame.
        Limit count of showed categories. Other are merged and showed as last.

        Returns:
            distribution (pd.DataFrame): Sorted DataFrame with aggregated categories.
        """

        distribution = self.data_col.groupby(self.data_col).size()
        distribution = distribution.reset_index(name=self.count_col_name)
        # sorts plot
        distribution.sort_values(by=self.count_col_name, inplace=True, ascending=False)
        # limit the count of categories
        distribution = self._limit_count(distribution)
        return distribution


class CatDescriptionSupervisedPandas(
    VariableDescriptionSupervisedPandas, CatDescriptionPandas, CatDescriptionSupervised
):
    """Class for supervised categorical pandas variable description."""

    def __init__(
        self,
        config: Univariate,
        data_col: pd.Series,
        target_description: TargetDescriptionPandas,
        **kwargs
    ) -> None:
        """Describe categorical data.

        Args:
            config (Univariate): Config of variables from report setting.
            data_col (pd.Series): Series with data, from processed column.
            target_description (TargetDescriptionPandas): Description of target series.
        """
        super().__init__(
            config=config,
            data_col=data_col,
            target_description=target_description,
            **kwargs
        )

    def _limit_count(self, df: pd.DataFrame) -> pd.DataFrame:
        """Limit count of displayed categories to max_cat.
        All other categories groups to one category 'other'
        """
        top_n_classes = df.drop_duplicates(self.data_col_name)[self.data_col_name].head(
            self._max_cat_to_plot
        )
        if top_n_classes.size < df[self.data_col_name].nunique():
            # select rows, that are not in top n classes and group them
            other = df[~df[self.data_col_name].isin(top_n_classes)]
            # TODO check if target col != data col
            if self.is_supervised():
                other = (
                    other.groupby(self.target_description.name)[self.count_col_name]
                    .sum()
                    .reset_index()
                )
                other[self.data_col_name] = self._other_placeholder
            else:
                sum = other[self.count_col_name].sum()
                other = pd.DataFrame(
                    data={
                        self.count_col_name: [sum],
                        self.data_col_name: [self._other_placeholder],
                    }
                )
            # drop all categories, that are not in top_n_categories
            df = df[df[self.data_col_name].isin(top_n_classes)]
            # merge top n categories and other
            df = pd.concat([df, other])
        return df

    def _generate_distribution(self) -> pd.DataFrame:
        """Generate grouped distribution DataFrame.
        Limit count of showed categories. Other are merged and showed as last.

        Returns:
            distribution (pd.DataFrame): Sorted DataFrame with aggregated categories.
        """
        # we have 2 different columns
        if self.data_col_name != self.target_col_name:
            # join columns by id
            data = self.data_col.to_frame().join(
                self.target_description.series_binary, how="inner"
            )
            distribution = data.groupby(data.columns.to_list()).size()
            # add zero values
            distribution = distribution.unstack(fill_value=0).stack().reset_index()
            distribution.rename(columns={0: self.count_col_name}, inplace=True)
        else:
            distribution = self.data_col.groupby(self.data_col).size()
            distribution = distribution.reset_index(name=self.count_col_name)

        # sorts plot
        distribution.sort_values(by=self.count_col_name, inplace=True, ascending=False)

        # limit the count of categories
        distribution = self._limit_count(distribution)
        return distribution


class NumDescriptionPandas(VariableDescriptionPandas, CatDescription):
    """Class for unsupervised numeric pandas variable description."""

    _bars: int

    def __init__(
        self, config: Univariate, data_col: pd.Series, bar_count: int, **kwargs
    ) -> None:
        """Describe numerical data.

        Args:
            config (Univariate): Config of variables from report setting.
            data_col (pd.Series): Series with data, from processed column.
            bar_count (int): Count of bars for distribution plot.
        """
        self._bars = bar_count
        super().__init__(config=config, data_col=data_col, **kwargs)

    def _group_distribution(self, data: pd.DataFrame) -> pd.DataFrame:
        """Get unsupervised distribution."""
        # replace bins with middle value (10, 20] -> 15
        data[self.data_col_name] = data[self.data_col_name].apply(lambda x: x.mid)
        data[self.data_col_name] = data[self.data_col_name].astype(float)
        sub = [self.data_col_name]
        # aggregate bins
        data_series = data.groupby(sub)[self.count_col_name].size()
        data = data_series.reset_index(name=self.count_col_name)
        return data

    def _generate_distribution(self) -> pd.DataFrame:
        """Cut continuous variable to bins.
        data_col is set to mid of generated cut.

        Returns:
            data (pd.DataFrame): Binned and grouped data.
        """

        # join columns by id
        data = pd.DataFrame()
        # TODO probably delete
        # set precision for col
        # range > 100 -> precision = 1
        # range < 100 -> precision = 2
        # range < 10 -> precision = 3
        range = self.data_col.max() - self.data_col.min()
        if range < 10:
            precision = 3
        elif range < 100:
            precision = 2
        else:
            precision = 1
        # add bins to data_col
        data[self.data_col_name] = pd.cut(
            self.data_col, bins=self._bars, precision=precision
        )
        data[self.count_col_name] = 0
        # group data
        data = self._group_distribution(data)
        return data


class NumDescriptionSupervisedPandas(
    VariableDescriptionSupervisedPandas, NumDescriptionPandas, CatDescriptionSupervised
):
    """Class for supervised numeric pandas variable description."""

    def __init__(
        self,
        config: Univariate,
        data_col: pd.Series,
        bar_count: int,
        target_description: TargetDescriptionPandas,
        **kwargs
    ) -> None:
        """Describe numerical supervised data.

        Args:
            config (Univariate): Config of variables from report setting.
            data_col (pd.Series): Series with data, from processed column.
            bar_count (int): Count of bars for distribution plot.
            target_description (TargetDescriptionPandas): Description of target series.
        """
        self._bars = bar_count
        super().__init__(
            config=config,
            data_col=data_col,
            bar_count=bar_count,
            target_description=target_description,
            **kwargs
        )

    def _group_distribution(self, data: pd.DataFrame) -> pd.DataFrame:
        """Group supervised numeric value.

        Args:
            data (pd.DataFrame): DataFrame with binned data_col
                and count_col with zeroes.

        Returns:
            data (pd.DataFrame):
                Grouped DataFrame by target_col and data_col.
                Column count_col contains counts for every target data combination.
                Even zero values.
        """
        data = data.join(self.target_description.series_binary, how="left")
        sub = [self.data_col_name, self.target_description.name]
        # aggregate bins
        data_series = data.groupby(sub)[self.count_col_name].size()
        # add zero values
        data = data_series.unstack(fill_value=0).stack().reset_index()
        data.rename(columns={0: self.count_col_name}, inplace=True)
        data[self.data_col_name] = data[self.data_col_name].astype(str)
        return data


class TextDescriptionPandas(VariableDescriptionPandas, TextDescription):
    words: pd.DataFrame
    stop_words: List[str] = []

    def __init__(self, config: Univariate, data_col: pd.Series, **kwargs) -> None:
        self.stop_words = config.cat.stop_words
        super().__init__(config=config, data_col=data_col, **kwargs)

    def _get_word_counts(self, data: pd.Series) -> pd.Series:
        """Count the number of occurrences of each individual word across
        all lines of the data Series, then sort from the word with the most
        occurrences to the word with the least occurrences. If a list of
        stop words is given, they will be ignored.

        Args:
            data: Series with data, we want to processed.

        Returns:
            Series with unique words as index and the computed frequency as value.
        """
        # get count of values
        value_counts = data.value_counts(dropna=True)
        series = pd.Series(value_counts.index, index=value_counts)
        word_lists = series.str.lower().str.split()
        words = word_lists.explode().str.strip(string.punctuation + string.whitespace)
        word_counts = pd.Series(words.index, index=words)
        # fix for pandas 1.0.5
        word_counts = word_counts[word_counts.index.notnull()]
        word_counts = word_counts.groupby(level=0, sort=False).sum()
        word_counts = word_counts.sort_values(ascending=False)

        # Remove stop words
        if len(self.stop_words) > 0:
            self.stop_words = [x.lower() for x in self.stop_words]
            word_counts = word_counts.loc[~word_counts.index.isin(self.stop_words)]
        return word_counts

    def get_word_counts(self) -> pd.DataFrame:
        return self._get_word_counts(self.data_col).to_frame(name=self.count_col_name)


class TextDescriptionSupervisedPandas(
    VariableDescriptionSupervisedPandas,
    TextDescriptionPandas,
    TextDescriptionSupervised,
):
    def __init__(
        self,
        config: Univariate,
        data_col: pd.Series,
        target_description: TargetDescriptionPandas,
        **kwargs
    ) -> None:
        super().__init__(
            config=config,
            data_col=data_col,
            target_description=target_description,
            **kwargs
        )

    def _get_word_counts_supervised(self) -> pd.DataFrame:
        if not self.target_description:
            raise ValueError("target not found in {}".format(self.data_col_name))
        data = self.data_col.to_frame().join(self.target_description.series_binary)
        positive_vals = data.loc[
            data[self.target_description.name] == self.target_description.bin_positive,
            self.data_col_name,
        ]
        negative_vals = data.loc[
            data[self.target_description.name] == self.target_description.bin_negative,
            self.data_col_name,
        ]
        positive_counts = self._get_word_counts(positive_vals).to_frame(
            name=self.positive_col_name
        )
        negative_counts = self._get_word_counts(negative_vals).to_frame(
            name=self.negative_col_name
        )
        word_counts = positive_counts.join(negative_counts)
        word_counts.fillna(0, inplace=True)
        word_counts[self.count_col_name] = (
            word_counts[self.positive_col_name] + word_counts[self.negative_col_name]
        )
        return word_counts.sort_values(by=self.count_col_name, ascending=False)

    def get_word_counts(self) -> pd.DataFrame:
        return self._get_word_counts_supervised()
