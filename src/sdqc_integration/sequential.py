import warnings
from typing import Dict, Any, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split
from sdqc_check import (
    CausalAnalysis,
    ClassificationModel,
    ShapFeatureImportance,
    data_corr,
    identify_data_types,
    CategoricalTest,
    NumericalTest,
    combine_data_and_labels
)

# Ignore warnings
warnings.filterwarnings('ignore')


class SequentialAnalysis:
    def __init__(
        self,
        raw_data: pd.DataFrame,
        synthetic_data: pd.DataFrame,
        random_seed: int = 17
    ) -> None:
        self.raw_data = raw_data
        self.synthetic_data = synthetic_data
        self.random_seed = random_seed

    def statistical_test_step(self) -> Dict[str, Any]:
        raw_data = pd.DataFrame(self.raw_data)
        synthetic_data = pd.DataFrame(self.synthetic_data)

        raw_col_dtypes = identify_data_types(raw_data)
        synthetic_col_dtypes = identify_data_types(synthetic_data)

        raw_correlation = data_corr(raw_data, raw_col_dtypes)
        synthetic_correlation = data_corr(synthetic_data, synthetic_col_dtypes)

        categorical_test = CategoricalTest()
        numerical_test = NumericalTest()

        results = {
            'raw': {'categorical': {}, 'numerical': {}},
            'synthetic': {'categorical': {}, 'numerical': {}},
            'distribution_comparison': {'categorical': {}, 'numerical': {}}
        }

        for col in raw_data.columns:
            print(col)
            if col in raw_col_dtypes['categorical']:
                results['raw']['categorical'][col] = categorical_test.basis(
                    raw_data[col]).to_dict()
                results['synthetic']['categorical'][col] = categorical_test.basis(
                    synthetic_data[col]).to_dict()

                jaccard_value, p_value = categorical_test.distribution(
                    raw_data[col], synthetic_data[col])
                results['distribution_comparison']['categorical'][col] = {
                    'jaccard_index': jaccard_value,
                    'chi_square_p_value': p_value
                }

            elif col in raw_col_dtypes['numerical']:
                results['raw']['numerical'][col] = numerical_test.basis(
                    raw_data[col]).to_dict()
                results['synthetic']['numerical'][col] = numerical_test.basis(
                    synthetic_data[col]).to_dict()

                wasserstein_value, hellinger_value = numerical_test.distribution(
                    raw_data[col], synthetic_data[col])
                results['distribution_comparison']['numerical'][col] = {
                    'wasserstein_distance': wasserstein_value,
                    'hellinger_distance': hellinger_value
                }

        return {
            'column_types': raw_col_dtypes,
            'raw_correlation': raw_correlation.to_dict(orient='records'),
            'synthetic_correlation': synthetic_correlation.to_dict(orient='records'),
            'results': results
        }

    def classification_step(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        classification = ClassificationModel(
            raw_data=self.raw_data,
            synthetic_data=self.synthetic_data,
            random_seed=self.random_seed,
            model_name=['svm', 'rf', 'xgb', 'lgbm']
        )
        metrics_results, model_results = classification.train_and_evaluate_models()
        return metrics_results, model_results

    def explainability_step(self, model, X_train, X_test, y_test):
        shap_explainability = ShapFeatureImportance(
            model=model,
            X_train=X_train,
            X_test=X_test,
            y_test=y_test,
            random_seed=self.random_seed
        )
        shap_results = shap_explainability.compute_feature_importance()
        return shap_results

    def causal_analysis_step(self):
        causal_analysis = CausalAnalysis(
            raw_data=self.raw_data,
            synthetic_data=self.synthetic_data,
            model_name='dlg',
            random_seed=self.random_seed
        )
        mt = causal_analysis.compare_adjacency_matrices()
        print('Comparative results:', mt.metrics)

    def run(self):
        statistical_test_results = self.statistical_test_step()
        print(1)
        metrics_results, model_results = self.classification_step()
        print(2)
        X, y = combine_data_and_labels(self.raw_data, self.synthetic_data)

        X_train, X_test, _, y_test = train_test_split(
            X, y, random_state=self.random_seed
        )

        shap_results = self.explainability_step(
            model_results[metrics_results['AUC'].argmax()],
            X_train,
            X_test,
            y_test
        )
        print(3)
        self.causal_analysis_step()

        return statistical_test_results, metrics_results, shap_results
