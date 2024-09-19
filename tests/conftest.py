import pytest
import pandas as pd
import numpy as np


@pytest.fixture
def raw_data():
    return pd.DataFrame({
        'A': np.random.randint(0, 100, 100),
        'B': np.random.choice([0, 1, 2], 100),
        'C': np.random.uniform(0, 1, 100),
        'D': np.random.choice([0, 1], 100)
    })


@pytest.fixture
def synthetic_data():
    return pd.DataFrame({
        'A': np.random.randint(0, 100, 100),
        'B': np.random.choice([0, 1, 2], 100),
        'C': np.random.uniform(0, 1, 100),
        'D': np.random.choice([0, 1], 100)
    })


@pytest.fixture
def col_dtypes():
    return {
        'categorical': ['B', 'D'],
        'numerical': ['A', 'C']
    }


@pytest.fixture
def sample_categorical_data1():
    return pd.Series(['A'] * 40 + ['B'] * 30 + ['C'] * 20 + ['D'] * 10)


@pytest.fixture
def sample_categorical_data2():
    return pd.Series(
        np.random.choice(
            ['A', 'B', 'C', 'D'], 100, p=[0.4, 0.3, 0.2, 0.1]
        )
    )


@pytest.fixture
def sample_numerical_data1():
    return pd.Series([1, 2, 3, 4, 5] * 20)


@pytest.fixture
def sample_numerical_data2():
    return pd.Series([1, 2, 3, 4, 5, 5, 4, 3, 2, 1] * 10)
