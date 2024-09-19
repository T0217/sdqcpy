import pandas as pd
import numpy as np
from typing import Tuple
from scipy.stats import wasserstein_distance
from .base_statistical_test import BaseStatisticalTest


class NumericalTest(BaseStatisticalTest):
    """
    Statistical test class for numerical data.

    This class provides methods to calculate basic statistical indicators and
    compare the distributions of raw and synthetic numerical data.
    """

    def basis(self, data: pd.Series) -> pd.Series:
        """
        Calculate basic statistical indicators for numerical data.

        Parameters
        ----------
        data : pd.Series
            Input numerical data series.

        Returns
        -------
        pd.Series
            Series containing various statistical indicators.
        """
        # Define percentiles and their formatted names
        percentiles = [0.25, 0.5, 0.75]
        formatted_percentiles = [f'{p:.0%}' for p in percentiles]

        # Define indicators for the statistical indicators
        indicators = ['count', 'missing', 'min'] + formatted_percentiles + \
            ['max', 'mean', 'var', 'cv', 'skew', 'kurt']

        # Calculate the statistical indicators
        count = data.count()
        missing = data.isna().sum()
        min_val = data.min()
        quantiles = data.quantile(percentiles).tolist()
        max_val = data.max()
        mean = data.mean()
        var = data.var()
        cv = data.std() / mean
        skew = data.skew()
        kurt = data.kurt()

        # Round the calculated values
        values = [count, missing, min_val] + quantiles + [
            max_val, round(mean, 2), round(var, 2), round(cv, 2),
            round(skew, 2), round(kurt, 2)
        ]

        return pd.Series(data=values, index=indicators, name=data.name)

    def distribution(
            self,
            raw_data: pd.Series,
            synthetic_data: pd.Series
    ) -> Tuple[float, float]:
        """
        Compare the distributions of raw and synthetic numerical data.

        Parameters
        ----------
        raw_data : pd.Series
            Original numerical data series.
        synthetic_data : pd.Series
            Synthetic numerical data series.

        Returns
        -------
        Tuple[float, float]
            Wasserstein distance and Hellinger distance.
        """
        # Calculate Wasserstein distance
        wasserstein_value = wasserstein_distance(raw_data, synthetic_data)

        # Calculate Hellinger distance
        raw_data = raw_data/np.sum(raw_data)
        synthetic_data = synthetic_data/np.sum(synthetic_data)
        hellinger_value = np.sqrt(
            np.sum((np.sqrt(raw_data) - np.sqrt(synthetic_data)) ** 2)
        ) / np.sqrt(2)

        return wasserstein_value, hellinger_value
