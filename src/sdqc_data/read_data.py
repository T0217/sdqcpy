import pandas as pd
import importlib
import os


def read_data(id: str) -> pd.DataFrame:
    """
    Read data from CSV files based on the provided ID.

    This function dynamically loads CSV files from the 'sdqc_data' package
    directory based on the given ID. It supports reading different datasets
    for various analysis purposes.

    Parameters
    ----------
    id : str
        The identifier for the dataset to be read. Valid options are:
        '1': For the first dataset
        '2': For the second dataset
        '3_raw': For the raw version of the third dataset
        '3_synth': For the synthetic version of the third dataset

    Returns
    -------
    pd.DataFrame
        A pandas DataFrame containing the requested dataset.

    Raises
    ------
    ValueError
        If an invalid ID is provided.
    """
    package_name = 'sdqc_data'
    package = importlib.import_module(package_name)
    package_file = os.path.dirname(package.__file__)

    # Define a mapping of IDs to file paths
    id_to_file = {
        '1': 'data/1.csv',
        '2': 'data/2.csv',
        '3_raw': 'data/3_raw.csv',
        '3_synth': 'data/3_synth.csv'
    }

    if id in id_to_file:
        file_path = os.path.join(package_file, id_to_file[id])
        return pd.read_csv(file_path)
    else:
        raise ValueError(
            f"Invalid id: {id}. Valid options are: {', '.join(id_to_file.keys())}")
