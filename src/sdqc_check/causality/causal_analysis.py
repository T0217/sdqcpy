import warnings
from typing import Tuple

import numpy as np
import pandas as pd

import castle
from castle.algorithms import (
    DirectLiNGAM,
    GAE,
    GOLEM,
    GraNDAG,
    Notears
)
from castle.metrics import MetricsDAG

# Ignore warnings
warnings.filterwarnings('ignore')


class CausalAnalysis:
    """
    Causal analysis using various causal discovery algorithms.

    Parameters
    ----------
    raw_data : pd.DataFrame
        The input raw data for causal analysis.
    synthetic_data : pd.DataFrame
        The input synthetic data for causal analysis.
    model_name : str, optional
        The name of the causal discovery model to use (default is 'dlg').
        Available models: 'dlg', 'notears', 'golem', 'grandag', 'gae'.
    random_seed : int, optional
        The random seed for reproducibility (default is 17).
    device_type : str, optional
        The type of device to use (default is 'cpu').
    device_id : int, optional
        The ID of the device to use (default is 0).
    """

    def __init__(
            self,
            raw_data: pd.DataFrame,
            synthetic_data: pd.DataFrame,
            model_name: str = 'dlg',
            random_seed: int = 17,
            device_type: str = 'cpu',
            device_id: int = 0
    ) -> None:
        self.raw_data = raw_data.to_numpy()
        self.synthetic_data = synthetic_data.to_numpy()
        self.model_name = model_name.lower()
        self.random_seed = random_seed
        self.device_type = device_type
        self.device_id = device_id
        self.modelList = ['dlg', 'notears', 'golem', 'grandag', 'gae']

        if self.model_name not in self.modelList:
            raise ValueError(
                f"Model {self.model_name} not found. "
                f"Available models are: {', '.join(self.modelList)}."
            )

    def compare_adjacency_matrices(self) -> None:
        """
        Compare the adjacency matrices of the raw and synthetic data using the selected causal discovery algorithm.
        """
        model = self._get_model(self.model_name)
        raw_matrix, synthetic_matrix = self._compute_causal_matrices(
            model,
            self.raw_data,
            self.synthetic_data
        )
        mt = MetricsDAG(raw_matrix, synthetic_matrix)
        return mt

    def _get_model(self, model_name: str) -> castle.common.BaseLearner:
        """
        Get the causal discovery model based on the model name.

        Parameters:
        -----------
        model_name : str
            The name of the causal discovery model.

        Returns:
        --------
        model : castle.common.BaseLearner
            The causal discovery model instance.
        """
        if model_name == 'dlg':
            return DirectLiNGAM()
        elif model_name == 'notears':
            return Notears()
        elif model_name == 'golem':
            return GOLEM(
                num_iter=1e4,
                seed=self.random_seed,
                device_type=self.device_type,
                device_id=self.device_id
            )
        elif model_name == 'grandag':
            return GraNDAG(
                input_dim=self.raw_data.shape[1],
                model_name='NonLinGaussANM',
                random_seed=self.random_seed,
                iterations=5000
            )
        elif model_name == 'gae':
            return GAE(
                update_freq=500,
                epochs=4,
                seed=self.random_seed,
                device_type=self.device_type,
                device_id=self.device_id
            )

    @staticmethod
    def _compute_causal_matrices(
            method: castle.common.BaseLearner,
            data1: np.ndarray,
            data2: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute the causal matrices using the specified causal discovery method.

        Parameters:
        -----------
        method : castle.common.BaseLearner
            The causal discovery method.
        data1 : numpy.ndarray
            The first input data for causal discovery.
        data2 : numpy.ndarray
            The second input data for causal discovery.

        Returns:
        --------
        Tuple[numpy.ndarray, numpy.ndarray]
            A tuple containing the causal matrices for data1 and data2.
        """
        method.learn(data1)
        causal_matrix1 = method.causal_matrix
        method.learn(data2)
        causal_matrix2 = method.causal_matrix

        return causal_matrix1, causal_matrix2
