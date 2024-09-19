import pytest
import pandas as pd
from sdqc_check import NumericalTest


@pytest.fixture
def numerical_test():
    return NumericalTest()


def test_basis(numerical_test, sample_numerical_data1):
    result = numerical_test.basis(sample_numerical_data1)
    assert isinstance(result, pd.Series)
    assert result['count'] == 100
    assert result['missing'] == 0
    assert result['min'] == 1
    assert result['max'] == 5
    assert result['mean'] == 3
    assert round(result['var']) == 2
    assert round(result['cv'], 1) == 0.5
    assert result['skew'] == 0
    assert round(result['kurt']) == -1


def test_distribution(
        numerical_test,
        sample_numerical_data1,
        sample_numerical_data2
):
    wasserstein, hellinger = numerical_test.distribution(
        sample_numerical_data1, sample_numerical_data2
    )
    assert isinstance(wasserstein, float)
    assert isinstance(hellinger, float)
    assert wasserstein >= 0
    assert 0 <= hellinger <= 1


def test_distances_identical(numerical_test):
    data = pd.Series([1, 2, 3, 4, 5])
    wasserstein, hellinger = numerical_test.distribution(data, data)
    assert wasserstein == 0
    assert hellinger == 0


def test_distances_different(numerical_test):
    data1 = pd.Series([1, 2, 3, 4, 5])
    data2 = pd.Series([6, 7, 8, 9, 10])
    wasserstein, hellinger = numerical_test.distribution(data1, data2)
    assert wasserstein > 0
    assert hellinger > 0
