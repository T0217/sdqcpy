import pandas as pd
from typing import Dict
from sdqc_synthesize import SDVSynthesizer


def test_sdv_synthesizer_initialization(raw_data):
    synthesizer = SDVSynthesizer(data=raw_data)
    assert isinstance(synthesizer, SDVSynthesizer)
    assert synthesizer.data.equals(raw_data)
    assert synthesizer.model_name == 'tvae'
    assert synthesizer.random_seed == 17


def test_sdv_synthesizer_fit(raw_data):
    synthesizer = SDVSynthesizer(data=raw_data)
    fitted_model = synthesizer.fit()
    assert fitted_model is not None


def test_sdv_synthesizer_generate(raw_data):
    synthesizer = SDVSynthesizer(data=raw_data)
    synthetic_data = synthesizer.generate()
    assert isinstance(synthetic_data, pd.DataFrame)
    assert synthetic_data.shape[0] == raw_data.shape[0]
    assert set(synthetic_data.columns) == set(raw_data.columns)


def test_sdv_synthesizer_multiple_models(raw_data):
    synthesizer = SDVSynthesizer(
        data=raw_data, model_name=['gaussiancopula', 'ctgan']
    )
    results = synthesizer.generate()
    assert isinstance(results, Dict)
    assert set(results.keys()) == {'gaussiancopula', 'ctgan'}
    for _, synthetic_data in results.items():
        assert isinstance(synthetic_data, pd.DataFrame)
        assert synthetic_data.shape[0] == raw_data.shape[0]
        assert set(synthetic_data.columns) == set(raw_data.columns)


def test_sdv_synthesizer_custom_num_rows(raw_data):
    num_rows = 10
    synthesizer = SDVSynthesizer(data=raw_data, num_rows=num_rows)
    synthetic_data = synthesizer.generate()
    assert isinstance(synthetic_data, pd.DataFrame)
    assert synthetic_data.shape[0] == num_rows


def test_sdv_synthesizer_custom_model_args(raw_data):
    model_args = {'epochs': 5}
    synthesizer = SDVSynthesizer(data=raw_data, model_args=model_args)
    assert synthesizer.model_args == model_args
    synthetic_data = synthesizer.generate()
    assert isinstance(synthetic_data, pd.DataFrame)
