import pandas as pd
from typing import Dict
from sdqc_synthesize import YDataSynthesizer


def test_ydata_synthesizer_initialization(raw_data):
    synthesizer = YDataSynthesizer(data=raw_data)
    assert isinstance(synthesizer, YDataSynthesizer)
    assert synthesizer.data.equals(raw_data)
    assert synthesizer.model_name == 'fast'
    assert synthesizer.random_seed == 17


def test_ydata_synthesizer_fit(raw_data):
    synthesizer = YDataSynthesizer(data=raw_data)
    fitted_model = synthesizer.fit()
    assert fitted_model is not None


def test_ydata_synthesizer_generate(raw_data):
    synthesizer = YDataSynthesizer(data=raw_data)
    synthetic_data = synthesizer.generate()
    assert isinstance(synthetic_data, pd.DataFrame)
    assert synthetic_data.shape[0] == raw_data.shape[0]
    assert set(synthetic_data.columns) == set(raw_data.columns)


def test_ydata_synthesizer_multiple_models(raw_data):
    synthesizer = YDataSynthesizer(data=raw_data, model_name=['gan', 'wgan'])
    results = synthesizer.generate()
    assert isinstance(results, Dict)
    assert set(results.keys()) == {'gan', 'wgan'}
    for _, synthetic_data in results.items():
        assert isinstance(synthetic_data, pd.DataFrame)
        assert synthetic_data.shape[0] == raw_data.shape[0]
        assert set(synthetic_data.columns) == set(raw_data.columns)


def test_ydata_synthesizer_custom_num_rows(raw_data):
    num_rows = 10
    synthesizer = YDataSynthesizer(data=raw_data, num_rows=num_rows)
    synthetic_data = synthesizer.generate()
    assert isinstance(synthetic_data, pd.DataFrame)
    assert synthetic_data.shape[0] == num_rows


def test_ydata_synthesizer_custom_model_args(raw_data):
    model_args = {'lr': 1e-3}
    synthesizer = YDataSynthesizer(
        data=raw_data, model_name='gan', model_args=model_args
    )
    assert synthesizer.model_args['lr'] == 1e-3
    synthetic_data = synthesizer.generate()
    assert isinstance(synthetic_data, pd.DataFrame)


def test_ydata_synthesizer_custom_train_args(raw_data):
    train_args = {'epochs': 50}
    synthesizer = YDataSynthesizer(
        data=raw_data, model_name='gan', train_args=train_args
    )
    assert synthesizer.train_args['epochs'] == 50
    synthetic_data = synthesizer.generate()
    assert isinstance(synthetic_data, pd.DataFrame)
