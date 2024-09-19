import pytest
import pandas as pd
from typing import Dict
from sdqc_check.statistical_test.utils import identify_data_types


@pytest.fixture
def sample_data():
    return pd.DataFrame({
        'bool_col': [True, False] * 10,
        'int_col': range(1, 21),
        'float_col': [i * 1.1 for i in range(1, 21)],
        'cat_col': ['A', 'B', 'C', 'D'] * 5,
        'cat_problem_col': [str(i) for i in range(20)],
        'na_problem_col': [1, 2, 3] + [pd.NA] * 17
    })


def test_identify_data_types(sample_data):
    result = identify_data_types(sample_data)

    assert isinstance(result, Dict)
    assert set(result.keys()) == {'categorical', 'numerical', 'problem'}

    assert 'bool_col' in result['categorical']
    assert 'cat_col' in result['categorical']
    assert 'int_col' in result['numerical']
    assert 'float_col' in result['numerical']
    assert 'cat_problem_col' in result['problem']
    assert 'na_problem_col' in result['problem']
