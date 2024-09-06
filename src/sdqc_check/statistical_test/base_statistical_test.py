import pandas as pd
from typing import Tuple
from abc import ABC, abstractmethod


class BaseStatisticalTest(ABC):
    """
    Abstract base class for statistical tests.

    This class defines the interface for statistical tests to be performed on data.
    Subclasses should implement the `basis` and `distribution` methods.
    """

    @abstractmethod
    def basis(self, data: pd.Series) -> pd.Series:
        """
        Calculate basic statistical measures for a given data series.

        Parameters
        ----------
        data : pd.Series
            Input data series.

        Returns
        -------
        pd.Series
            Series containing basic statistical measures.
        """
        raise NotImplementedError(
            "Subclasses must implement the 'basis' method."
        )

    @abstractmethod
    def distribution(
            self,
            raw_data: pd.Series,
            synthetic_data: pd.Series
    ) -> Tuple[float, float]:
        """
        Compare the distributions of raw and synthetic data.

        Parameters
        ----------
        raw_data : pd.Series
            Original data series.
        synthetic_data : pd.Series
            Synthetic data series.

        Returns
        -------
        Tuple[float, float]
            Two measures of distribution similarity.
        """
        raise NotImplementedError(
            "Subclasses must implement the 'distribution' method."
        )
