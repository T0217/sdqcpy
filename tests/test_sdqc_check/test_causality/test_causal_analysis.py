import pytest
import castle
import numpy as np
import pandas as pd
from sdqc_check import CausalAnalysis
from sdqc_data import read_data


@pytest.fixture
def sample_data(raw_data, synthetic_data):
    return raw_data, synthetic_data


def test_causal_analysis_initialization(sample_data):
    raw_data, synthetic_data = sample_data
    ca = CausalAnalysis(raw_data, synthetic_data)
    assert isinstance(ca, CausalAnalysis)
    assert np.array_equal(ca.raw_data, raw_data.to_numpy())
    assert np.array_equal(ca.synthetic_data, synthetic_data.to_numpy())
    assert ca.model_name == 'dlg'
    assert ca.random_seed == 17
    assert ca.device_type == 'cpu'
    assert ca.device_id == 0


def test_causal_analysis_invalid_model():
    with pytest.raises(ValueError):
        CausalAnalysis(
            pd.DataFrame(), pd.DataFrame(), model_name='invalid_model'
        )


@pytest.mark.parametrize('model_name', ['dlg', 'notears', 'golem', 'grandag', 'gae'])
def test_get_model(sample_data, model_name):
    raw_data, synthetic_data = sample_data
    ca = CausalAnalysis(raw_data, synthetic_data, model_name=model_name)
    model = ca._get_model(model_name)
    assert isinstance(model, castle.common.BaseLearner)


def test_compute_causal_matrices(sample_data):
    raw_data, synthetic_data = sample_data
    ca = CausalAnalysis(raw_data, synthetic_data)
    raw_matrix, synthetic_matrix = ca.compute_causal_matrices()
    assert isinstance(raw_matrix, np.ndarray)
    assert isinstance(synthetic_matrix, np.ndarray)
    assert raw_matrix.shape == (4, 4)
    assert synthetic_matrix.shape == (4, 4)

def test_compare_adjacency_matrices():
    raw_data = read_data('3_raw')
    synthetic_data = read_data('3_synth')
    ca = CausalAnalysis(raw_data, synthetic_data)
    mt = ca.compare_adjacency_matrices()
    assert isinstance(mt, castle.MetricsDAG)
