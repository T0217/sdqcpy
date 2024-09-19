import pandas as pd
import numpy as np
from sdqc_check import data_corr


def test_data_corr(raw_data, col_dtypes):
    result = data_corr(raw_data, col_dtypes)

    assert isinstance(result, pd.DataFrame)
    assert set(result.columns) == {'column1',
                                   'column2', 'method', 'corr_coefficient'}

    expected_pairs = [
        ('A', 'B'), ('A', 'C'), ('A', 'D'),
        ('B', 'C'), ('B', 'D'), ('C', 'D')
    ]
    assert set(zip(result['column1'], result['column2'])
               ) == set(expected_pairs)

    assert set(result['method']) == {"Cramer's V", 'Pearson', 'Eta'}
    assert (result['corr_coefficient'] >= -
            1).all() and (result['corr_coefficient'] <= 1).all()


def test_data_corr_perfect_correlation():
    perfect_data = pd.DataFrame({
        'cat1': ['A', 'B'] * 50,
        'cat2': ['X', 'Y'] * 50,
        'num1': list(range(100)),
        'num2': list(range(0, 200, 2))
    })

    col_dtypes = {
        'categorical': ['cat1', 'cat2'],
        'numerical': ['num1', 'num2']
    }

    result = data_corr(perfect_data, col_dtypes)

    cramer_v = result[(result['column1'] == 'cat1') & (
        result['column2'] == 'cat2')]['corr_coefficient'].values[0]
    assert np.isclose(cramer_v, 1.0, atol=0.02)

    pearson = result[(result['column1'] == 'num1') & (
        result['column2'] == 'num2')]['corr_coefficient'].values[0]
    assert np.isclose(pearson, 1.0)
