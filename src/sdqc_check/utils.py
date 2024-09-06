import pandas as pd
import numpy as np
from typing import Tuple


def combine_data_and_labels(
        real_data: pd.DataFrame,
        synthetic_data: pd.DataFrame
) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Combines real and synthetic data into a single DataFrame and creates corresponding labels.

    Parameters:
    -----------
    real_data : pd.DataFrame
        DataFrame containing real data samples.
    synthetic_data : pd.DataFrame
        DataFrame containing synthetic data samples.

    Returns:
    --------
    Tuple[pd.DataFrame, np.ndarray]
        A tuple containing:
        1. X: Combined DataFrame with real and synthetic data.
        2. y: Array of labels, where 1 represents real data and 0 represents synthetic data.
    """
    # Combine real and synthetic data into a single DataFrame
    X = pd.concat([real_data, synthetic_data], ignore_index=True)

    # Create labels: 1 for real data, 0 for synthetic data
    y = np.concatenate(
        [np.ones(len(real_data)), np.zeros(len(synthetic_data))])

    return X, y


def set_seed(seed: int) -> None:
    """
    Set the seed for reproducibility across different libraries.

    This method sets the random seed for Python's built-in random module, NumPy, PyTorch, 
    and the PYTHONHASHSEED environment variable to ensure consistent results across runs.

    Parameters:
    -----------
    seed : int
        The seed value to be set for random number generation. This value will be used to initialize 
        the random number generators of various libraries.

    Returns:
    --------
    None
        This function does not return any value. It sets the random seed for multiple libraries and 
        the PYTHONHASHSEED environment variable.
    """
    import random
    import os
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
