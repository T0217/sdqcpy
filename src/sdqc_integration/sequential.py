import warnings
from typing import Dict, Any, Tuple

from tqdm import tqdm
import pandas as pd
from sklearn.model_selection import train_test_split
from sdqc_check import (
    CausalAnalysis,
    ClassificationModel,
    ShapFeatureImportance,
    PFIFeatureImportance,
    LimeFeatureImportance,
    data_corr,
    identify_data_types,
    CategoricalTest,
    NumericalTest,
    combine_data_and_labels
)

# Ignore warnings
warnings.filterwarnings('ignore')


class SequentialAnalysis:
    """
    A class to perform sequential analysis on raw and synthetic data.

    This class provides methods for statistical testing, classification,
    explainability analysis, and causal analysis.

    Parameters
    ----------
    raw_data : pd.DataFrame
        The original raw data.
    synthetic_data : pd.DataFrame
        The synthetic data generated from the raw data.
    random_seed : int, optional
        Random seed for reproducibility (default is 17).
    causal_model_name : str, optional
        Name of the causal model to use (default is 'dlg').
    explainability_algorithm : str, optional
        Algorithm to use for explainability analysis (default is 'shap').
    """

    def __init__(
        self,
        raw_data: pd.DataFrame,
        synthetic_data: pd.DataFrame,
        random_seed: int = 17,
        causal_model_name: str = 'dlg',
        explainability_algorithm: str = 'shap'
    ) -> None:
        self.raw_data = raw_data
        self.synthetic_data = synthetic_data
        self.random_seed = random_seed
        self.causal_model_name = causal_model_name
        self.explainability_algorithm = explainability_algorithm
        self.results = {}

    def statistical_test_step(self) -> Dict[str, Any]:
        """
        Perform statistical tests on raw and synthetic data.

        Returns
        -------
        Dict[str, Any]
            A dictionary containing the results of statistical tests.
        """
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
            if col in raw_col_dtypes['categorical']:
                self._process_categorical_column(
                    col, raw_data, synthetic_data, categorical_test, results)
            elif col in raw_col_dtypes['numerical']:
                self._process_numerical_column(
                    col, raw_data, synthetic_data, numerical_test, results)

        return {
            'column_types': raw_col_dtypes,
            'raw_correlation': raw_correlation.to_dict(orient='records'),
            'synthetic_correlation': synthetic_correlation.to_dict(orient='records'),
            'results': results
        }

    def _process_categorical_column(self, col, raw_data, synthetic_data, categorical_test, results):
        """
        Process categorical columns.
        """
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

    def _process_numerical_column(self, col, raw_data, synthetic_data, numerical_test, results):
        """
        Process numerical columns.
        """
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

    def classification_step(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Perform classification on raw and synthetic data.

        Returns
        -------
        Tuple[pd.DataFrame, pd.DataFrame]
            A tuple containing metrics results and model results.
        """
        classification = ClassificationModel(
            raw_data=self.raw_data,
            synthetic_data=self.synthetic_data,
            random_seed=self.random_seed,
            model_name=['svm', 'rf', 'xgb', 'lgbm']
        )
        metrics_results, model_results = classification.train_and_evaluate_models()
        return metrics_results, model_results

    def explainability_step(self, model, X_train, X_test, y_test):
        """
        Perform explainability analysis on the best model.

        Parameters
        ----------
        model : object
            The trained model to explain.
        X_train : pd.DataFrame
            Training data features.
        X_test : pd.DataFrame
            Test data features.
        y_test : pd.Series
            Test data labels.

        Returns
        -------
        pd.DataFrame
            Feature importance scores.
        """
        explainability_class = self._get_explainability_class()
        explainer = explainability_class(
            model=model,
            X_train=X_train,
            X_test=X_test,
            y_test=y_test,
            random_seed=self.random_seed
        )
        return explainer.compute_feature_importance()

    def _get_explainability_class(self):
        """
        Get the appropriate explainability class.
        """
        explainability_classes = {
            'shap': ShapFeatureImportance,
            'pfi': PFIFeatureImportance,
            'lime': LimeFeatureImportance
        }
        if self.explainability_algorithm not in explainability_classes:
            raise ValueError(
                f"Algorithm {self.explainability_algorithm} not supported. "
                "Please choose from 'shap', 'pfi', or 'lime'"
            )
        return explainability_classes[self.explainability_algorithm]

    def causal_analysis_step(self):
        """
        Perform causal analysis on raw and synthetic data.
        """
        causal_analysis = CausalAnalysis(
            raw_data=self.raw_data,
            synthetic_data=self.synthetic_data,
            model_name=self.causal_model_name,
            random_seed=self.random_seed
        )
        mt = causal_analysis.compare_adjacency_matrices()
        return mt.metrics

    def run(self):
        """
        Run the complete sequential analysis.

        Returns
        -------
        Dict[str, Any]
            A dictionary containing results from all analysis steps.
        """
        steps = [
            ('Statistical Test', self.statistical_test_step),
            ('Classification', self.classification_step),
            ('Explainability', self._perform_explainability),
            ('Causal Analysis', self.causal_analysis_step)
        ]

        for step_name, step_function in tqdm(steps, desc="Executing steps"):
            self.results[step_name] = step_function()

        return self.results

    def _perform_explainability(self):
        """
        Perform explainability analysis.
        """
        X, y = combine_data_and_labels(self.raw_data, self.synthetic_data)
        X_train, X_test, _, y_test = train_test_split(
            X, y, random_state=self.random_seed
        )
        best_model = self.results['Classification'][1][self.results['Classification'][0]['AUC'].argmax(
        )]
        return self.explainability_step(best_model, X_train, X_test, y_test)
