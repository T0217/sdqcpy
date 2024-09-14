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
        raw_matrix, synthetic_matrix = self.compute_causal_matrices()
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
                device_ids=self.device_id
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
                device_ids=self.device_id
            )

    def compute_causal_matrices(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute causal matrices for raw and synthetic data using the specified causal discovery method.

        Parameters:
        -----------
        raw_data : np.ndarray
            The raw input data for causal discovery.
        synthetic_data : np.ndarray
            The synthetic input data for causal discovery.

        Returns:
        --------
        Tuple[np.ndarray, np.ndarray]
            A tuple containing the causal matrices for raw and synthetic data.
        """
        model = self._get_model(self.model_name)

        # Compute causal matrix for raw data
        model.learn(self.raw_data)
        raw_causal_matrix = model.causal_matrix

        # Compute causal matrix for synthetic data
        model.learn(self.synthetic_data)
        synthetic_causal_matrix = model.causal_matrix

        return raw_causal_matrix, synthetic_causal_matrix
