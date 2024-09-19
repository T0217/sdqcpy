import pandas as pd
from typing import Dict, List


def identify_data_types(data: pd.DataFrame) -> Dict[str, List[str]]:
    """
    Identify the data types of columns in a DataFrame.

    This function analyzes each column in the input DataFrame and categorizes
    them as 'categorical', 'numerical', or 'problem' based on their characteristics.

    Parameters
    ----------
    data : pd.DataFrame
        The input DataFrame to analyze.

    Returns
    -------
    Dict[str, List[str]]
        A dictionary with keys 'categorical', 'numerical', and 'problem',
        each containing a list of column names corresponding to that data type.

    Notes
    -----
    - Boolean columns are always considered categorical.
    - Integer and float columns are categorized based on their uniqueness and range.
    - Object columns are categorized based on their uniqueness relative to the column length.
    - Columns with less than 10 non-null values are labeled as 'problem'.
    """
    col_dtype = {
        'categorical': [],
        'numerical': [],
        'problem': []
    }

    for col_name in data.columns:
        col_data = data[col_name].dropna()
        data_type = col_data.infer_objects().dtype.kind

        if len(col_data) <= 10:
            col_dtype['problem'].append(col_name)
            continue

        if data_type == 'b':
            col_dtype['categorical'].append(col_name)
        elif data_type in ['i', 'f']:
            if (
                    (col_data == col_data.round()).all() and
                    (col_data >= 0).all() and
                    (col_data.nunique() <= min(round(len(col_data) / 10), 10))
            ):
                col_dtype['categorical'].append(col_name)
            else:
                col_dtype['numerical'].append(col_name)
        elif data_type == 'O':
            if col_data.nunique() <= max(round(len(col_data) / 10), 10):
                col_dtype['categorical'].append(col_name)
            else:
                col_dtype['problem'].append(col_name)
        else:
            col_dtype['problem'].append(col_name)

    return col_dtype
