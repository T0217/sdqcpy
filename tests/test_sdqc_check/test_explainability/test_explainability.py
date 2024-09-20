import pytest
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sdqc_check import (
    ShapFeatureImportance, PFIFeatureImportance, ModelBasedFeatureImportance
)


@pytest.fixture
def sample_data():
    X, y = make_classification(
        n_samples=100, n_features=10, n_informative=5, random_state=17
    )
    X = pd.DataFrame(X, columns=[f'feature_{i + 1}' for i in range(10)])
    y = pd.Series(y)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=17
    )
    model = RandomForestClassifier(random_state=17)
    model.fit(X_train, y_train)
    return model, X_train, X_test, y_train, y_test


@pytest.mark.parametrize("FeatureImportance", [
    ShapFeatureImportance, PFIFeatureImportance, ModelBasedFeatureImportance
])
def test_feature_importance(sample_data, FeatureImportance):
    model, X_train, X_test, y_train, y_test = sample_data
    importance = FeatureImportance(model, X_train, X_test, y_test)
    result = importance.compute_feature_importance()

    assert isinstance(result, pd.DataFrame)
    assert len(result) == 10
    assert set(result.columns) == {'feature', 'importance'}
    assert result['feature'].nunique() == 10
    assert all(
        f'feature_{i + 1}' in result['feature'].values for i in range(10)
    )
    assert result['importance'].is_monotonic_decreasing


@pytest.mark.parametrize("FeatureImportance", [
    ShapFeatureImportance, PFIFeatureImportance, ModelBasedFeatureImportance
])
def test_feature_importance_random_seed(sample_data, FeatureImportance):
    model, X_train, X_test, y_train, y_test = sample_data
    importance1 = FeatureImportance(
        model, X_train, X_test, y_test, random_seed=17
    )
    importance2 = FeatureImportance(
        model, X_train, X_test, y_test, random_seed=17
    )

    scores1 = importance1.compute_feature_importance()
    scores2 = importance2.compute_feature_importance()

    pd.testing.assert_frame_equal(scores1, scores2)


@pytest.mark.parametrize("model_class", [RandomForestClassifier, SVC])
def test_shap_feature_importance_models(sample_data, model_class):
    _, X_train, X_test, y_train, y_test = sample_data
    if model_class == SVC:
        model = model_class(random_state=17, probability=True)
    else:
        model = model_class(random_state=17)
    model.fit(X_train, y_train)

    shap_importance = ShapFeatureImportance(model, X_train, X_test, y_test)
    importance = shap_importance.compute_feature_importance()

    assert isinstance(importance, pd.DataFrame)
    assert len(importance) == 10
    assert set(importance.columns) == {'feature', 'importance'}
    assert importance['feature'].nunique() == 10
    assert all(
        f'feature_{i + 1}' in importance['feature'].values for i in range(10)
    )
    assert importance['importance'].is_monotonic_decreasing
