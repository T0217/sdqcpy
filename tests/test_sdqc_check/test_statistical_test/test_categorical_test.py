import pytest
import pandas as pd
from sdqc_check import CategoricalTest


@pytest.fixture
def categorical_test():
    return CategoricalTest()


def test_basis(categorical_test, sample_categorical_data1):
    result = categorical_test.basis(sample_categorical_data1)
    assert isinstance(result, pd.Series)
    assert result['count'] == 100
    assert result['missing'] == 0
    assert result['unique'] == 4
    assert result['top'] == 'A'
    assert result['freq'] == 40


def test_distribution(
        categorical_test,
        sample_categorical_data1,
        sample_categorical_data2
):
    jaccard, p_value = categorical_test.distribution(
        sample_categorical_data1, sample_categorical_data2
    )
    assert isinstance(jaccard, float)
    assert isinstance(p_value, float)
    assert 0 <= jaccard <= 1
    assert 0 <= p_value <= 1


def test_jaccard_index_identical(categorical_test):
    data = pd.Series(['A', 'B', 'C'])
    jaccard, _ = categorical_test.distribution(data, data)
    assert jaccard == 1


def test_jaccard_index_disjoint(categorical_test):
    data1 = pd.Series(['A', 'B', 'C'])
    data2 = pd.Series(['D', 'E', 'F'])
    jaccard, _ = categorical_test.distribution(data1, data2)
    assert jaccard == 0


def test_jaccard_index_different(categorical_test):
    data1 = pd.Series(['A', 'B', 'A', 'C', 'B', 'A', 'C', 'A', 'B', 'C'])
    data2 = pd.Series(['A', 'A', 'A', 'A', 'A', 'B', 'B', 'B', 'C', 'D'])
    jaccard, p_value = categorical_test.distribution(data1, data2)
    assert jaccard < 1
    assert p_value == 0


def test_chi_square_identical(categorical_test):
    data1 = pd.Series(['A'] * 4 + ['B'] * 2 + ['C'] * 3 + ['D'] * 1)
    data2 = pd.Series(['D'] * 1 + ['C'] * 3 + ['B'] * 2 + ['A'] * 4)

    _, p_value = categorical_test.distribution(data1, data2)
    assert p_value == 1


def test_chi_square_different(categorical_test):
    data1 = pd.Series(['A', 'B', 'A', 'C', 'B', 'A', 'D', 'A', 'B', 'C'])
    data2 = pd.Series(['A', 'A', 'A', 'A', 'A', 'B', 'B', 'B', 'C', 'D'])
    _, p_value = categorical_test.distribution(data1, data2)
    assert p_value < 1
