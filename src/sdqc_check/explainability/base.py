from abc import ABC, abstractmethod
from typing import Any
import pandas as pd


class BaseFeatureImportance(ABC):
    """
    Abstract base class for feature importance calculation.

    This class defines the interface for feature importance calculators.
    Subclasses should implement the compute_feature_importance method.

    Parameters
    ----------
    model : Any
        The trained model object.
    X_train : pd.DataFrame
        Training data.
    X_test : pd.DataFrame
        Test data.
    y_test : pd.Series
        Test labels.
    random_seed : int, optional
        Random seed for reproducibility (default is 17).
    """

    def __init__(
            self,
            model: Any,
            X_train: pd.DataFrame,
            X_test: pd.DataFrame,
            y_test: pd.Series,
            random_seed: int = 17
    ) -> None:
        self.model = model
        self.X_train = X_train
        self.X_test = X_test
        self.y_test = y_test
        self.random_seed = random_seed

    @abstractmethod
    def compute_feature_importance(self) -> pd.DataFrame:
        """
        Compute feature importance.

        Returns
        -------
        pd.DataFrame
            A DataFrame containing feature importance scores.
        """
        raise NotImplementedError("Subclasses must implement this method.")
