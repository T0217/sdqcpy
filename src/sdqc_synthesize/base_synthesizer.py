import os
import random
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Union

import numpy as np
import pandas as pd
import tensorflow as tf
import torch
from sdv.metadata import SingleTableMetadata


class BaseSynthesizer(ABC):
    """
    Abstract base class for synthesizers that generate synthetic data.

    This class provides a common interface and utility methods for synthesizers.
    Subclasses should implement the `fit` and `generate` methods.

    Parameters
    ----------
    data : pd.DataFrame
        The input data to synthesize from.
    metadata : Optional[Union[Dict[str, Any], str, SingleTableMetadata]]
        Metadata describing the data structure. It can be a dictionary, a JSON file path,
        or an instance of SingleTableMetadata. If not provided, metadata will be inferred
        from the input data.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        metadata: Optional[Union[Dict[str, Any],
                                 str, SingleTableMetadata]] = None,
    ) -> None:
        self.data = data
        self.metadata = metadata

    def validate_arg(self) -> None:
        """
        Validate the model name argument for the synthesizer.

        This method checks if the provided model name is in the list of available models.

        Raises
        ------
        ValueError
            If the model name is not found in the available models.
        """
        self.model_name = self.model_name.lower()
        if self.model_name not in self.modelList:
            raise ValueError(
                f"Model {self.model_name} not found. "
                f"Available models are: {', '.join(self.modelList)}."
            )

    def create_metadata(self) -> SingleTableMetadata:
        """
        Create metadata from the input DataFrame if no metadata is provided.

        This method detects the metadata from the input DataFrame using the
        `detect_from_dataframe` method of `SingleTableMetadata`.

        Returns
        -------
        SingleTableMetadata
            The detected metadata from the input DataFrame.

        Raises
        ------
        TypeError
            If the input data is not a pandas DataFrame.
        """
        if not isinstance(self.data, pd.DataFrame):
            raise TypeError("Input data must be a pandas DataFrame")

        metadata = SingleTableMetadata()
        metadata.detect_from_dataframe(self.data)
        return metadata

    def check_metadata(self) -> SingleTableMetadata:
        """
        Validate and return the provided metadata, creating it from the data if necessary.

        This method checks the type of the provided metadata and returns an instance of
        `SingleTableMetadata`. If no metadata is provided, it creates the metadata from
        the input data using the `create_metadata` method.

        Returns
        -------
        SingleTableMetadata
            The metadata to be used for the synthesizer.

        Raises
        ------
        ValueError
            If the metadata format is not supported.
        """
        if self.metadata is None:
            return self.create_metadata()
        elif isinstance(self.metadata, SingleTableMetadata):
            return self.metadata
        elif isinstance(self.metadata, str):
            metadata = SingleTableMetadata()
            metadata.load_from_json(self.metadata)
            return metadata
        elif isinstance(self.metadata, dict):
            metadata = SingleTableMetadata()
            metadata.load_from_dict(self.metadata)
            return metadata
        else:
            raise ValueError(
                'Input must be a dictionary, a JSON file path, '
                'or an instance of SingleTableMetadata.'
            )

    @abstractmethod
    def fit(self) -> None:
        """
        Fit the synthesizer to the input data.

        This method must be implemented by subclasses to train the synthesizer
        on the input data.

        Raises
        ------
        NotImplementedError
            If the method is not implemented in the subclass.
        """
        raise NotImplementedError(
            "Subclasses must implement the 'fit' method."
        )

    @abstractmethod
    def generate(self) -> Union[pd.DataFrame, Dict[str, pd.DataFrame]]:
        """
        Generate synthetic data using the synthesizer.

        This method must be implemented by subclasses to generate synthetic data
        based on the trained synthesizer.

        Returns
        -------
        Union[pd.DataFrame, Dict[str, pd.DataFrame]]
            A DataFrame of synthetic data or a dictionary of results.

        Raises
        ------
        NotImplementedError
            If the method is not implemented in the subclass.
        """
        raise NotImplementedError(
            "Subclasses must implement the 'generate' method."
        )

    @staticmethod
    def set_seed(seed: int) -> None:
        """
        Set the seed for reproducibility across different libraries.

        This method sets the random seed for Python's built-in random module,
        NumPy, PyTorch, TensorFlow, and the PYTHONHASHSEED environment variable.

        Parameters
        ----------
        seed : int
            The seed value to be set for random number generation.
        """
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        tf.random.set_seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
