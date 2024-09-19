import pytest
import pandas as pd
import numpy as np
from typing import List, Dict
from sdqc_integration import SequentialAnalysis
from sdqc_data import read_data


@pytest.fixture
def sample_data():
    raw_data = read_data('3_raw')
    synthetic_data = read_data('3_synth')
    return raw_data, synthetic_data


def test_sequential_analysis_initialization(sample_data):
    raw_data, synthetic_data = sample_data
    analysis = SequentialAnalysis(raw_data, synthetic_data)
    assert isinstance(analysis, SequentialAnalysis)
    assert analysis.raw_data.equals(raw_data)
    assert analysis.synthetic_data.equals(synthetic_data)


def test_statistical_test_step(sample_data):
    raw_data, synthetic_data = sample_data
    analysis = SequentialAnalysis(raw_data, synthetic_data)
    result = analysis.statistical_test_step()
    assert isinstance(result, dict)
    assert 'column_types' in result
    assert 'raw_correlation' in result
    assert 'synthetic_correlation' in result
    assert 'results' in result


def test_classification_step(sample_data):
    raw_data, synthetic_data = sample_data
    analysis = SequentialAnalysis(raw_data, synthetic_data)
    metrics, models = analysis.classification_step()
    assert isinstance(metrics, pd.DataFrame)
    assert isinstance(models, List)
    assert len(models) > 0


def test_causal_analysis_step(sample_data):
    raw_data, synthetic_data = sample_data
    analysis = SequentialAnalysis(raw_data, synthetic_data)
    metrics, raw_matrix, synthetic_matrix = analysis.causal_analysis_step()
    assert isinstance(metrics, Dict)
    assert isinstance(raw_matrix, np.ndarray)
    assert isinstance(synthetic_matrix, np.ndarray)


@pytest.mark.parametrize('explainability_algorithm', ['shap', 'pfi'])
def test_explainability_step(sample_data, explainability_algorithm):
    raw_data, synthetic_data = sample_data
    analysis = SequentialAnalysis(
        raw_data, synthetic_data, explainability_algorithm=explainability_algorithm
    )
    analysis.run()
    assert isinstance(analysis.results['Explainability'], pd.DataFrame)


def test_run(sample_data):
    raw_data, synthetic_data = sample_data
    analysis = SequentialAnalysis(raw_data, synthetic_data)
    results = analysis.run()
    assert isinstance(results, Dict)
    assert 'Statistical Test' in results
    assert 'Classification' in results
    assert 'Explainability' in results
    assert 'Causal Analysis' in results


def test_visualize_html(sample_data, tmp_path):
    raw_data, synthetic_data = sample_data
    analysis = SequentialAnalysis(raw_data, synthetic_data)
    analysis.run()
    output_path = tmp_path / 'test_report.html'
    analysis.visualize_html(str(output_path))
    assert output_path.exists()
