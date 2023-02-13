from typing import Optional

import numpy as np
import pandas as pd
from pandas_profiling.model.description_plot import BasePlotDescription


class PlotDescriptionPandas(BasePlotDescription):
    """Base class for pandas plot description."""

    _data_col: pd.Series
    _target_col: Optional[pd.Series]

    def __init__(self, data_col: pd.Series, target_col: Optional[pd.Series]) -> None:
        self._data_col = data_col
        self._target_col = target_col
        data_col_name = self.__prepare_data_col_name(data_col)
        target_col_name = self.__prepare_target_col_name(target_col)
        super().__init__(data_col_name, target_col_name)

    @classmethod
    def __prepare_data_col_name(cls, data_col: pd.Series) -> str:
        """Fills col name, if None.

        Returns column name
        """
        if data_col.name is None:
            data_col.name = "data_col"
        return str(data_col.name)

    @classmethod
    def __prepare_target_col_name(
        cls, target_col: Optional[pd.Series]
    ) -> Optional[str]:
        if target_col is None:
            return None
        if target_col.name is None:
            target_col.name = "target_col"
        return str(target_col.name)


class CategoricalPlotDescriptionPandas(PlotDescriptionPandas):
    _other_placeholder: str = "other ..."
    _max_cat_to_plot: int

    def __init__(
        self, data_col: pd.Series, target_col: Optional[pd.Series], max_cat_to_plot: int
    ) -> None:
        """Prepare categorical data for plotting

        Parameters
        ----------
        data_col : pd.Series
            series with data, from processed column
        target_col : pd.Series or None
            series with target column, if is set, else None
        max_cat_to_plot : int
            limit for plotting. If we have more categories, than max_cat_to_plot,
            all below threshold will be merged to other category
        """
        data_col = data_col.astype(str)
        super().__init__(data_col, target_col)

        self._max_cat_to_plot = max_cat_to_plot

        # we have 2 different columns
        if self._target_col is not None and self.data_col_name != self.target_col_name:
            # join columns by id
            data = (
                self._data_col.to_frame()
                .join(self._target_col, how="inner")
                .astype(str)
            )
            distribution = data.groupby(data.columns.to_list()).size().reset_index()
            distribution.rename(columns={0: self.count_col_name}, inplace=True)
        else:
            distribution = data_col.groupby(data_col).size()
            distribution = distribution.rename(index=self.count_col_name).reset_index()

        # sorts plot
        distribution.sort_values(by=self.count_col_name, inplace=True, ascending=False)

        # limit the count of plotted categories
        distribution = self._limit_count(distribution)

        # add column for label position
        distribution = self._add_labels_location(distribution)
        self._set_distribution(distribution)

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
            if (
                self.target_col_name is not None
                and self.data_col_name != self.target_col_name
            ):
                other = (
                    other.groupby(self.target_col_name)[self.count_col_name]
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

    def _add_labels_location(self, df: pd.DataFrame):
        col_name = "labels_location"
        df[col_name] = "right"
        df.loc[
            df[self.count_col_name] < df[self.count_col_name].max() / 4, col_name
        ] = "left"
        return df


class NumericPlotDescriptionPandas(PlotDescriptionPandas):
    def __init__(
        self, data_col: pd.Series, target_col: Optional[pd.Series], max_bar_count: int
    ) -> None:
        super().__init__(data_col, target_col)
        self._bars = max_bar_count

        distribution = self._get_distribution()
        self._set_distribution(distribution)

    def _get_distribution(self) -> pd.DataFrame:
        """Cut continuous variable to bins.

        Returns
        -------
        data : pd.DataFrame
            Binned and grouped data.
        """
        # join columns by id
        data = pd.DataFrame()
        data[self.data_col_name] = pd.cut(self._data_col, bins=self._bars, precision=0)
        data[self.count_col_name] = 0
        if self._target_col is not None:
            data = data.join(self._target_col, how="inner")
            sub = [self.data_col_name, self.target_col_name]
        else:
            sub = [self.data_col_name]
        data = data.groupby(sub)[self.count_col_name].count().reset_index()
        return data
