import itertools
import numpy as np
import pandas as pd
from typing import Dict, List
from scipy.stats import chi2_contingency, pearsonr


def data_corr(data: pd.DataFrame, col_dtypes: Dict[str, List]) -> pd.DataFrame:
    """
    Calculate correlation coefficients between columns in a DataFrame.

    This function computes the following correlations:
    - Cramer's V for categorical-categorical pairs
    - Pearson's correlation coefficient for numerical-numerical pairs
    - Eta coefficient for categorical-numerical pairs

    Parameters
    ----------
    data : pd.DataFrame
        Input data.
    col_dtypes : Dict[str, List]
        Dictionary specifying the column types with keys 'numerical' and 'categorical'.

    Returns
    -------
    pd.DataFrame
        DataFrame containing correlation coefficients between columns.
    """
    numerical_cols = col_dtypes['numerical']
    categorical_cols = col_dtypes['categorical']
    col_corr = pd.DataFrame(
        columns=['column1', 'column2', 'method', 'corr_coefficient'])

    # Calculate correlations for each pair of columns
    for col1, col2 in itertools.combinations(data.columns, 2):
        if col1 in categorical_cols and col2 in categorical_cols:
            # Cramer's V for categorical-categorical pairs
            crosstab = pd.crosstab(data[col1], data[col2])
            chi2, _, _, _ = chi2_contingency(crosstab)
            n = np.sum(data.values)
            min_dim = min(data.shape) - 1
            corr_coef = np.sqrt((chi2 / n) / min_dim)
            method = "Cramer's V"
        elif col1 in numerical_cols and col2 in numerical_cols:
            # Pearson's correlation coefficient for numerical-numerical pairs
            corr_coef, _ = pearsonr(data[col1], data[col2])
            method = 'Pearson'
        else:
            # Eta coefficient for categorical-numerical pairs
            if col1 in categorical_cols:
                cat_col, num_col = col1, col2
            else:
                cat_col, num_col = col2, col1

            group_means = data.groupby(cat_col)[num_col].mean()
            grand_mean = data[num_col].mean()

            between_group_variance = (
                (group_means - grand_mean)**2 * data.groupby(cat_col).size()
            ).sum()
            total_variance = ((data[num_col] - grand_mean)**2).sum()
            corr_coef = np.sqrt(between_group_variance / total_variance)
            method = 'Eta'

        # Add the correlation to the result DataFrame
        col_corr = pd.concat([
            col_corr,
            pd.DataFrame({
                'column1': [col1],
                'column2': [col2],
                'method': [method],
                'corr_coefficient': [corr_coef]
            })
        ], ignore_index=True)

    return col_corr
