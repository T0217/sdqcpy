import pandas as pd
from typing import Tuple
from scipy.stats import chisquare
from .base_statistical_test import BaseStatisticalTest


class CategoricalTest(BaseStatisticalTest):
    """
    Statistical test class for categorical data.

    This class provides methods to calculate basic statistical indicators and
    compare the distributions of raw and synthetic categorical data.
    """

    def basis(self, data: pd.Series) -> pd.Series:
        """
        Calculate basic statistical indicators for categorical data.

        Parameters
        ----------
        data : pd.Series
            Input categorical data series.

        Returns
        -------
        pd.Series
            Series containing various statistical indicators.
        """
        indicators = ['count', 'missing', 'unique', 'top', 'freq']
        objcounts = data.value_counts()
        count_unique = len(objcounts[objcounts != 0])

        # Determine the most frequent category and its frequency
        top = objcounts.index[0] if count_unique > 0 else pd.NA
        freq = objcounts.iloc[0] if count_unique > 0 else pd.NA

        values = [data.count(), data.isna().sum(), count_unique, top, freq]
        return pd.Series(data=values, index=indicators, name=data.name)

    def distribution(
            self,
            raw_data: pd.Series,
            synthetic_data: pd.Series
    ) -> Tuple[float, float]:
        """
        Compare the distributions of raw and synthetic categorical data.

        Parameters
        ----------
        raw_data : pd.Series
            Original categorical data series.
        synthetic_data : pd.Series
            Synthetic categorical data series.

        Returns
        -------
        Tuple[float, float]
            Jaccard index and chi-square goodness of fit test p-value.
        """
        def jaccard_index(data1, data2):
            """
            Calculate the Jaccard index between two sets of data.

            The Jaccard index measures the similarity between two sets by
            calculating the ratio of the size of their intersection to the
            size of their union.

            Parameters
            ----------
            data1 : pd.Series
                First input data series.
            data2 : pd.Series
                Second input data series.

            Returns
            -------
            float
                Jaccard index, or pd.NA if the union is empty.
            """
            set_a = set(data1)
            set_b = set(data2)
            intersection = len(set_a.intersection(set_b))
            union = len(set_a.union(set_b))
            return intersection / union if union != 0 else pd.NA

        # Calculate the Jaccard index between raw and synthetic data
        jaccard_value = jaccard_index(raw_data, synthetic_data)
        if jaccard_value == 1:
            # Perform chi-square goodness of fit test
            _, p_value = chisquare(
                raw_data.value_counts().values,
                synthetic_data.value_counts().values
            )
        else:
            p_value = 0

        return jaccard_value, p_value
