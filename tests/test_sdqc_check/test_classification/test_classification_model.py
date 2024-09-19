import pytest
import pandas as pd
from sdqc_check import ClassificationModel


@pytest.fixture
def sample_data(raw_data, synthetic_data):
    return raw_data, synthetic_data


def test_classification_model_initialization(sample_data):
    raw_data, synthetic_data = sample_data
    cm = ClassificationModel(raw_data, synthetic_data)
    assert isinstance(cm, ClassificationModel)
    assert cm.model_name == 'rf'
    assert cm.test_size == 0.2
    assert cm.random_seed == 17


def test_classification_model_invalid_model():
    with pytest.raises(ValueError):
        ClassificationModel(
            pd.DataFrame(), pd.DataFrame(), model_name='invalid_model'
        )


@pytest.mark.parametrize('model_name', ['svm', 'rf', 'xgb', 'lgbm'])
def test_classification_model_single_model(sample_data, model_name):
    raw_data, synthetic_data = sample_data
    cm = ClassificationModel(raw_data, synthetic_data, model_name=model_name)
    metrics, models = cm.train_and_evaluate_models()
    assert isinstance(metrics, pd.DataFrame)
    assert len(models) == 1
    assert metrics['Model'].iloc[0] == model_name


def test_classification_model_multiple_models(sample_data):
    raw_data, synthetic_data = sample_data
    model_names = ['svm', 'rf', 'xgb', 'lgbm']
    cm = ClassificationModel(raw_data, synthetic_data, model_name=model_names)
    metrics, models = cm.train_and_evaluate_models()
    assert isinstance(metrics, pd.DataFrame)
    assert len(models) == len(model_names)
    assert set(metrics['Model']) == set(model_names)


def test_classification_model_custom_params(sample_data):
    raw_data, synthetic_data = sample_data
    custom_params = {
        'rf': {'n_estimators': 100, 'max_depth': 5},
        'xgb': {'n_estimators': 100, 'max_depth': 5}
    }
    cm = ClassificationModel(
        raw_data, synthetic_data,
        model_name=['rf', 'xgb'],
        model_params=custom_params
    )
    metrics, models = cm.train_and_evaluate_models()
    assert isinstance(metrics, pd.DataFrame)
    assert len(models) == 2
    assert set(metrics['Model']) == {'rf', 'xgb'}


def test_classification_model_metrics(sample_data):
    raw_data, synthetic_data = sample_data
    cm = ClassificationModel(raw_data, synthetic_data)
    metrics, _ = cm.train_and_evaluate_models()
    expected_columns = [
        'Model', 'Accuracy', 'Precision', 'Recall', 'F1', 'AUC'
    ]
    assert set(metrics.columns) == set(expected_columns)
