import warnings
from typing import (
    Dict, Any, Tuple, List, Type, Optional, Union
)

from tqdm import tqdm
import pandas as pd
import numpy as np
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

import io
import json
import base64
import itertools
import matplotlib.pyplot as plt
import seaborn as sns
from jinja2 import Environment, PackageLoader, select_autoescape

# Ignore warnings
warnings.filterwarnings('ignore')
# Set font and minus sign for plots
plt.rcParams['font.sans-serif'] = 'SimHei'
plt.rcParams['axes.unicode_minus'] = False


class SequentialAnalysis:
    """
    A class to perform sequential analysis on raw and synthetic data.

    This class provides methods for statistical testing, classification,
    explainability analysis, and causal analysis. It compares the raw data
    with synthetic data generated from it across multiple dimensions.

    Parameters
    ----------
    raw_data : pd.DataFrame
        The original raw data.
    synthetic_data : pd.DataFrame
        The synthetic data generated from the raw data.
    use_cols : Optional[Union[List[str], str]], optional
        Columns to use for analysis. If None, all columns are used.
    random_seed : int, optional
        Random seed for reproducibility (default is 17).
    causal_model_name : str, optional
        Name of the causal model to use (default is 'dlg').
    explainability_algorithm : str, optional
        Algorithm to use for explainability analysis (default is 'shap').

    Attributes
    ----------
    use_cols : List[str]
        Columns used for analysis.
    raw_data : pd.DataFrame
        The original raw data (subset to use_cols).
    synthetic_data : pd.DataFrame
        The synthetic data (subset to use_cols).
    random_seed : int
        Random seed for reproducibility.
    causal_model_name : str
        Name of the causal model used.
    explainability_algorithm : str
        Algorithm used for explainability analysis.
    results : Dict[str, Any]
        Dictionary to store results from various analysis steps.

    Methods
    -------
    statistical_test_step():
        Perform statistical tests on raw and synthetic data.
    classification_step():
        Perform classification on raw and synthetic data.
    explainability_step():
        Perform explainability analysis on raw and synthetic data.
    causal_analysis_step():
        Perform causal analysis on raw and synthetic data.
    run():
        Run the full sequential analysis pipeline.
    generate_html_report(output_path: str):
        Generate an HTML report of the analysis results.
    """

    def __init__(
        self,
        raw_data: pd.DataFrame,
        synthetic_data: pd.DataFrame,
        use_cols: Optional[Union[List[str], str]] = None,
        random_seed: int = 17,
        causal_model_name: str = 'dlg',
        explainability_algorithm: str = 'shap'
    ) -> None:
        self.use_cols = use_cols if use_cols is not None else raw_data.columns
        self.raw_data = raw_data[self.use_cols]
        self.synthetic_data = synthetic_data[self.use_cols]
        self.random_seed = random_seed
        self.causal_model_name = causal_model_name
        self.explainability_algorithm = explainability_algorithm
        self.results = {}

    def statistical_test_step(self) -> Dict[str, Any]:
        """
        Perform statistical tests on raw and synthetic data.

        This method conducts various statistical tests to compare the distributions
        of raw and synthetic data for each column. It handles both categorical and
        numerical data types appropriately.

        Returns
        -------
        Dict[str, Any]
            A dictionary containing the results of statistical tests, including:
            - Column types (categorical or numerical)
            - Correlation matrices for raw and synthetic data
            - Detailed statistical results for each column
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
                    col, raw_data, synthetic_data, categorical_test, results
                )
            elif col in raw_col_dtypes['numerical']:
                self._process_numerical_column(
                    col, raw_data, synthetic_data, numerical_test, results
                )

        return {
            'column_types': raw_col_dtypes,
            'raw_correlation': raw_correlation,
            'synthetic_correlation': synthetic_correlation,
            'results': results
        }

    def _process_categorical_column(
            self,
            col: str,
            raw_data: pd.DataFrame,
            synthetic_data: pd.DataFrame,
            categorical_test: CategoricalTest,
            results: Dict[str, Any]
    ) -> None:
        """
        Process a categorical column for statistical testing.

        This method computes basic statistics and distribution comparisons
        for a categorical column in both raw and synthetic data.

        Parameters
        ----------
        col : str
            The name of the categorical column to process.
        raw_data : pd.DataFrame
            The raw data DataFrame.
        synthetic_data : pd.DataFrame
            The synthetic data DataFrame.
        categorical_test : CategoricalTest
            An instance of CategoricalTest for performing statistical tests.
        results : Dict[str, Any]
            A dictionary to store the computed results.
        """
        results['raw']['categorical'][col] = categorical_test.basis(
            raw_data[col]).to_dict()
        results['synthetic']['categorical'][col] = categorical_test.basis(
            synthetic_data[col]).to_dict()

        jaccard_value, p_value = categorical_test.distribution(
            raw_data[col], synthetic_data[col]
        )
        results['distribution_comparison']['categorical'][col] = {
            'jaccard_index': jaccard_value,
            'chi_square_p_value': p_value
        }

    def _process_numerical_column(
            self,
            col: str,
            raw_data: pd.DataFrame,
            synthetic_data: pd.DataFrame,
            numerical_test: NumericalTest,
            results: Dict[str, Any]
    ) -> None:
        """
        Process a numerical column for statistical testing.

        This method computes basic statistics and distribution comparisons
        for a numerical column in both raw and synthetic data.

        Parameters
        ----------
        col : str
            The name of the numerical column to process.
        raw_data : pd.DataFrame
            The raw data DataFrame.
        synthetic_data : pd.DataFrame
            The synthetic data DataFrame.
        numerical_test : NumericalTest
            An instance of NumericalTest for performing statistical tests.
        results : Dict[str, Any]
            A dictionary to store the computed results.
        """
        results['raw']['numerical'][col] = numerical_test.basis(
            raw_data[col]).to_dict()
        results['synthetic']['numerical'][col] = numerical_test.basis(
            synthetic_data[col]).to_dict()

        wasserstein_value, hellinger_value = numerical_test.distribution(
            raw_data[col], synthetic_data[col]
        )
        results['distribution_comparison']['numerical'][col] = {
            'wasserstein_distance': wasserstein_value,
            'hellinger_distance': hellinger_value
        }

    def classification_step(self) -> Tuple[pd.DataFrame, List[object]]:
        """
        Perform classification on raw and synthetic data.

        This method trains and evaluates multiple classification models
        (SVM, Random Forest, XGBoost, LightGBM) on the combined dataset
        of raw and synthetic data.

        Returns
        -------
        Tuple[pd.DataFrame, List[object]]
            A tuple containing:
            - DataFrame with classification metrics for each model
            - List of trained model objects
        """
        classification = ClassificationModel(
            raw_data=self.raw_data,
            synthetic_data=self.synthetic_data,
            random_seed=self.random_seed,
            model_name=['svm', 'rf', 'xgb', 'lgbm']
        )
        return classification.train_and_evaluate_models()

    def explainability_step(
            self,
            model: object,
            X_train: pd.DataFrame,
            X_test: pd.DataFrame,
            y_test: pd.Series
    ) -> pd.DataFrame:
        """
        Perform explainability analysis on the best model.

        This method applies the specified explainability algorithm
        (SHAP, PFI, or LIME) to interpret the predictions of the best
        performing model.

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
            Feature importance values computed by the explainability algorithm.
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

    def _get_explainability_class(self) -> Type:
        """
        Get the appropriate explainability class based on the specified algorithm.

        Returns
        -------
        Type
            The explainability class (SHAP, PFI, or LIME) to be used.

        Raises
        ------
        ValueError
            If the specified explainability algorithm is not supported.
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

    def _perform_explainability(self) -> pd.DataFrame:
        """
        Perform explainability analysis on the best model.

        This method combines raw and synthetic data, splits it into
        training and test sets, selects the best model based on AUC,
        and applies the explainability algorithm.

        Returns
        -------
        pd.DataFrame
            Feature importance values for the best model.
        """
        X, y = combine_data_and_labels(self.raw_data, self.synthetic_data)
        X_train, X_test, _, y_test = train_test_split(
            X, y, random_state=self.random_seed
        )
        best_model = self.results['Classification'][1][self.results[
            'Classification'][0]['AUC'].argmax()]
        return self.explainability_step(best_model, X_train, X_test, y_test)

    def causal_analysis_step(
            self
    ) -> Tuple[Dict[str, Any], np.ndarray, np.ndarray]:
        """
        Perform causal analysis on raw and synthetic data.

        This method computes and compares causal relationships in
        raw and synthetic data using the specified causal model.

        Returns
        -------
        Tuple[Dict[str, Any], np.ndarray, np.ndarray]
            A tuple containing:
            - Dictionary of causal metrics
            - Raw data causal matrix
            - Synthetic data causal matrix
        """
        causal_analysis = CausalAnalysis(
            raw_data=self.raw_data,
            synthetic_data=self.synthetic_data,
            model_name=self.causal_model_name,
            random_seed=self.random_seed
        )
        mt = causal_analysis.compare_adjacency_matrices()
        raw_causal_matrix, synthetic_causal_matrix = causal_analysis.compute_causal_matrices()
        return mt.metrics, raw_causal_matrix, synthetic_causal_matrix

    def run(self) -> Dict[str, Any]:
        """
        Run the complete sequential analysis pipeline.

        This method executes all analysis steps in sequence:
        statistical testing, classification, explainability,
        and causal analysis.

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

        for step_name, step_function in tqdm(steps, desc="Executing analysis steps"):
            self.results[step_name] = step_function()

        return self.results

    def _generate_column_stats(
            self,
            column_types: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Generate statistical plots and metrics for each column in the dataset.

        This method creates probability mass function (PMF) plots for categorical data
        and kernel density estimation (KDE) plots for numerical data. It also compiles
        various statistical metrics for each column.

        Parameters
        ----------
        column_types : Dict[str, Any]
            A dictionary specifying the type of each column ('categorical' or 'numerical').

        Returns
        -------
        Dict[str, Any]
            A dictionary containing plots and statistics for each column.
        """
        stats = {}
        for col in self.raw_data.columns:
            if col in column_types['categorical']:
                # Generate PMF plots for categorical data
                plt.figure(figsize=(8, 5))
                self.raw_data[col].value_counts(normalize=True).plot(
                    kind='bar',
                    alpha=0.8,
                    label='Raw Data',
                    color='lightblue',
                    edgecolor='black',
                    linewidth=1
                )
                self.synthetic_data[col].value_counts(normalize=True).plot(
                    kind='bar',
                    alpha=0.8,
                    label='Synthetic Data',
                    color='mistyrose',
                    edgecolor='black',
                    linewidth=1
                )
                plt.legend()
                plt.title(f'{col} PMF Comparison')
                plt.xlabel('Category')
                plt.ylabel('Probability')

                # Retrieve statistical metrics
                distribution_comparison = self.results['Statistical Test'][
                    'results']['distribution_comparison']['categorical'][col]
                raw_stats = self.results['Statistical Test']['results'][
                    'raw']['categorical'][col]
                synth_stats = self.results['Statistical Test']['results'][
                    'synthetic']['categorical'][col]

                # Convert plot to base64 encoded string
                buffer = io.BytesIO()
                plt.savefig(buffer, format='jpg')
                buffer.seek(0)
                image_base64 = base64.b64encode(
                    buffer.getvalue()).decode('utf-8')

                stats[col] = {
                    'plot': f'data:image/jpg;base64,{image_base64}',
                    'distribution_comparison': distribution_comparison,
                    'raw_stats': raw_stats,
                    'synth_stats': synth_stats
                }

                plt.close()

            elif col in column_types['numerical']:
                # Generate KDE plots for numerical data
                plt.figure(figsize=(8, 5))
                self.raw_data[col].plot(
                    kind='kde',
                    label='Raw',
                    color='blue'
                )
                self.synthetic_data[col].plot(
                    kind='kde',
                    label='Synthetic',
                    color='red'
                )
                plt.legend()
                plt.title(f'{col} KDE Comparison')
                plt.xlabel('Value')
                plt.ylabel('Density')

                # Retrieve statistical metrics
                distribution_comparison = self.results['Statistical Test'][
                    'results']['distribution_comparison']['numerical'][col]
                raw_stats = self.results['Statistical Test']['results'][
                    'raw']['numerical'][col]
                synth_stats = self.results['Statistical Test']['results'][
                    'synthetic']['numerical'][col]

                # Convert plot to base64 encoded string
                buffer = io.BytesIO()
                plt.savefig(buffer, format='jpg')
                buffer.seek(0)
                image_base64 = base64.b64encode(
                    buffer.getvalue()).decode('utf-8')

                stats[col] = {
                    'plot': f'data:image/jpg;base64,{image_base64}',
                    'distribution_comparison': distribution_comparison,
                    'raw_stats': raw_stats,
                    'synth_stats': synth_stats
                }

                plt.close()
            else:
                # Skip columns that are neither categorical nor numerical
                pass
        return stats

    def _generate_correlation_plots(
            self,
            column_types: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Generate correlation plots for pairs of columns in the dataset.

        This method creates various types of plots depending on the data types of the columns:
        - Heatmaps for categorical vs categorical
        - Scatter plots for numerical vs numerical
        - Box plots for categorical vs numerical (and vice versa)

        Parameters
        ----------
        column_types : Dict[str, Any]
            A dictionary specifying the type of each column ('categorical' or 'numerical').

        Returns
        -------
        Dict[str, Any]
            A dictionary containing correlation plots for each pair of columns.
        """
        correlation_plots = {}
        raw_correlation = self.results['Statistical Test']['raw_correlation']
        synthetic_correlation = self.results['Statistical Test']['synthetic_correlation']

        for col1, col2 in itertools.combinations(self.raw_data.columns, 2):
            if col1 in column_types['categorical'] and col2 in column_types['categorical']:
                # Generate heatmaps for categorical vs categorical
                fig, (ax1, ax2) = plt.subplots(figsize=(12, 5), ncols=2)
                ax1.set_title(f'{col1} vs {col2} (Raw Data)')
                sns.heatmap(
                    pd.crosstab(self.raw_data[col1], self.raw_data[col2]),
                    annot=True,
                    ax=ax1,
                    cbar=False
                )
                ax1.text(
                    0.5, -0.125,
                    f"{raw_correlation[(raw_correlation['column1'] == col1) & (raw_correlation['column2'] == col2)]['method'].values[0]} "
                    f"Correlation Coefficient: "
                    f"{raw_correlation[(raw_correlation['column1'] == col1) & (raw_correlation['column2'] == col2)]['corr_coefficient'].values[0] :.4f}",
                    ha='center', va='center',
                    transform=ax1.transAxes
                )

                ax2.set_title(f'{col1} vs {col2} (Synthetic Data)')
                sns.heatmap(
                    pd.crosstab(
                        self.synthetic_data[col1], self.synthetic_data[col2]),
                    annot=True,
                    ax=ax2,
                    cbar=False
                )
                ax2.text(
                    0.5, -0.125,
                    f"{synthetic_correlation[(synthetic_correlation['column1'] == col1) & (synthetic_correlation['column2'] == col2)]['method'].values[0]} "
                    f"Correlation Coefficient: "
                    f"{synthetic_correlation[(synthetic_correlation['column1'] == col1) & (synthetic_correlation['column2'] == col2)]['corr_coefficient'].values[0] :.4f}",
                    ha='center', va='center',
                    transform=ax2.transAxes
                )
                buffer = io.BytesIO()
                plt.savefig(buffer, format='jpg')
                buffer.seek(0)
                image_base64 = base64.b64encode(
                    buffer.getvalue()).decode('utf-8')
                correlation_plots[f'{col1} vs {col2}'] = f'data:image/jpg;base64,{image_base64}'
                correlation_plots[f'{col2} vs {col1}'] = f'data:image/jpg;base64,{image_base64}'
                plt.close()
            elif col1 in column_types['numerical'] and col2 in column_types['numerical']:
                # Generate scatter plots for numerical vs numerical
                fig, (ax1, ax2) = plt.subplots(figsize=(12, 5), ncols=2)
                ax1.set_title(f'{col1} vs {col2} (Raw Data)')
                ax1.scatter(
                    self.raw_data[col1],
                    self.raw_data[col2],
                    alpha=0.5,
                    color='blue'
                )
                ax1.set_xlabel(col1)
                ax1.set_ylabel(col2)
                ax1.text(
                    0.5, -0.125,
                    f"{raw_correlation[(raw_correlation['column1'] == col1) & (raw_correlation['column2'] == col2)]['method'].values[0]} "
                    f"Correlation Coefficient: "
                    f"{raw_correlation[(raw_correlation['column1'] == col1) & (raw_correlation['column2'] == col2)]['corr_coefficient'].values[0] :.4f}",
                    ha='center', va='center',
                    transform=ax1.transAxes
                )

                ax2.set_title(f'{col1} vs {col2} (Synthetic Data)')
                ax2.scatter(
                    self.synthetic_data[col1],
                    self.synthetic_data[col2],
                    alpha=0.5,
                    color='red'
                )
                ax2.set_xlabel(col1)
                ax2.set_ylabel(col2)
                ax2.text(
                    0.5, -0.125,
                    f"{synthetic_correlation[(synthetic_correlation['column1'] == col1) & (synthetic_correlation['column2'] == col2)]['method'].values[0]} "
                    f"Correlation Coefficient: "
                    f"{synthetic_correlation[(synthetic_correlation['column1'] == col1) & (synthetic_correlation['column2'] == col2)]['corr_coefficient'].values[0] :.4f}",
                    ha='center', va='center',
                    transform=ax2.transAxes
                )
                buffer = io.BytesIO()
                plt.savefig(buffer, format='jpg')
                buffer.seek(0)
                image_base64 = base64.b64encode(
                    buffer.getvalue()
                ).decode('utf-8')
                correlation_plots[f'{col1} vs {col2}'] = f'data:image/jpg;base64,{image_base64}'
                correlation_plots[f'{col2} vs {col1}'] = f'data:image/jpg;base64,{image_base64}'

                plt.close()

            elif col1 in column_types['categorical'] and col2 in column_types['numerical']:
                # Generate box plots for categorical vs numerical
                fig, (ax1, ax2) = plt.subplots(figsize=(12, 5), ncols=2)
                ax1.set_title(f'{col1} vs {col2} (Raw Data)')
                sns.boxplot(
                    x=self.raw_data[col1],
                    y=self.raw_data[col2],
                    ax=ax1
                )
                ax1.set_xlabel(col1)
                ax1.set_ylabel(col2)
                ax1.text(
                    0.5, -0.125,
                    f"{raw_correlation[(raw_correlation['column1'] == col1) & (raw_correlation['column2'] == col2)]['method'].values[0]} "
                    f"Correlation Coefficient: "
                    f"{raw_correlation[(raw_correlation['column1'] == col1) & (raw_correlation['column2'] == col2)]['corr_coefficient'].values[0] :.4f}",
                    ha='center', va='center',
                    transform=ax1.transAxes
                )

                ax2.set_title(f'{col1} vs {col2} (Synthetic Data)')
                sns.boxplot(
                    x=self.synthetic_data[col1],
                    y=self.synthetic_data[col2],
                    ax=ax2
                )
                ax2.set_xlabel(col1)
                ax2.set_ylabel(col2)
                ax2.text(
                    0.5, -0.125,
                    f"{synthetic_correlation[(synthetic_correlation['column1'] == col1) & (synthetic_correlation['column2'] == col2)]['method'].values[0]} "
                    f"Correlation Coefficient: "
                    f"{synthetic_correlation[(synthetic_correlation['column1'] == col1) & (synthetic_correlation['column2'] == col2)]['corr_coefficient'].values[0] :.4f}",
                    ha='center', va='center',
                    transform=ax2.transAxes
                )
                buffer = io.BytesIO()
                plt.savefig(buffer, format='jpg')
                buffer.seek(0)
                image_base64 = base64.b64encode(
                    buffer.getvalue()).decode('utf-8')
                correlation_plots[f'{col1} vs {col2}'] = f'data:image/jpg;base64,{image_base64}'
                correlation_plots[f'{col2} vs {col1}'] = f'data:image/jpg;base64,{image_base64}'

                plt.close()

            elif col1 in column_types['numerical'] and col2 in column_types['categorical']:
                # Generate box plots for numerical vs categorical
                fig, (ax1, ax2) = plt.subplots(figsize=(12, 5), ncols=2)
                ax1.set_title(f'{col1} vs {col2} (Raw Data)')
                sns.boxplot(
                    x=self.raw_data[col2],
                    y=self.raw_data[col1],
                    ax=ax1
                )
                ax1.set_xlabel(col2)
                ax1.set_ylabel(col1)
                ax1.text(
                    0.5, -0.125,
                    f"{raw_correlation[(raw_correlation['column1'] == col1) & (raw_correlation['column2'] == col2)]['method'].values[0]} "
                    f"Correlation Coefficient: "
                    f"{raw_correlation[(raw_correlation['column1'] == col1) & (raw_correlation['column2'] == col2)]['corr_coefficient'].values[0] :.4f}",
                    ha='center', va='center',
                    transform=ax1.transAxes
                )

                ax2.set_title(f'{col1} vs {col2} (Synthetic Data)')
                sns.boxplot(
                    x=self.synthetic_data[col2],
                    y=self.synthetic_data[col1],
                    ax=ax2
                )
                ax2.set_xlabel(col2)
                ax2.set_ylabel(col1)
                ax2.text(
                    0.5, -0.125,
                    f"{synthetic_correlation[(synthetic_correlation['column1'] == col1) & (synthetic_correlation['column2'] == col2)]['method'].values[0]} "
                    f"Correlation Coefficient: "
                    f"{synthetic_correlation[(synthetic_correlation['column1'] == col1) & (synthetic_correlation['column2'] == col2)]['corr_coefficient'].values[0] :.4f}",
                    ha='center', va='center',
                    transform=ax2.transAxes
                )
                buffer = io.BytesIO()
                plt.savefig(buffer, format='jpg')
                buffer.seek(0)
                image_base64 = base64.b64encode(
                    buffer.getvalue()
                ).decode('utf-8')
                correlation_plots[f'{col1} vs {col2}'] = f'data:image/jpg;base64,{image_base64}'
                correlation_plots[f'{col2} vs {col1}'] = f'data:image/jpg;base64,{image_base64}'
                plt.close()
            else:
                # Skip pairs that don't fit into the above categories
                pass
        return correlation_plots

    def _generate_feature_importance(
            self
    ) -> List[Any]:
        """
        Generate a feature importance plot and data.

        This method creates a horizontal bar plot of feature importances
        and returns both the plot and the raw data.

        Returns
        -------
        List[Any]
            A list containing the feature importance data and the plot as a base64 encoded string.
        """
        feature_importance = [
            self.results['Explainability'].to_dict(orient='records')]
        features = []
        importances = []

        plt.figure(figsize=(10, 6))
        for item in feature_importance[0]:
            if item['importance'] >= 0.01:
                features.append(item['feature'])
                importances.append(item['importance'])

        plt.barh(features[::-1], importances[::-1])
        plt.ylabel('Features')
        plt.xlabel('Importance')
        plt.title('Feature Importance')

        buffer = io.BytesIO()
        plt.savefig(buffer, format='jpg')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        feature_importance.append(f'data:image/jpg;base64,{image_base64}')
        plt.close()

        return feature_importance

    def _generate_causal_comparison(
            self
    ) -> Dict[str, Any]:
        """
        Generate causal comparison metrics and adjacency matrix plots.

        This method creates adjacency matrix plots for both raw and synthetic data,
        and compiles causal metrics.

        Returns
        -------
        Dict[str, Any]
            A dictionary containing causal metrics and adjacency matrix plots.
        """
        causal_comparison = {}
        causal_comparison['causal_metrics'] = self.results['Causal Analysis'][0]
        raw_causal_matrix = self.results['Causal Analysis'][1]
        synthetic_causal_matrix = self.results['Causal Analysis'][2]

        # Set diagonal elements to 0
        np.fill_diagonal(raw_causal_matrix, 0)
        np.fill_diagonal(synthetic_causal_matrix, 0)

        fig, (ax1, ax2) = plt.subplots(figsize=(8, 4), ncols=2)

        ax1.set_title('Raw Adjacency Matrix')
        ax1.imshow(raw_causal_matrix, cmap='Greys', interpolation='none')

        ax2.set_title('Synthetic Adjacency Matrix')
        ax2.imshow(synthetic_causal_matrix, cmap='Greys', interpolation='none')

        buffer = io.BytesIO()
        plt.savefig(buffer, format='jpg')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        causal_comparison['Adjacency_matrices'] = f'data:image/jpg;base64,{image_base64}'
        plt.close()

        return causal_comparison

    def visualize_html(
            self,
            output_path: str
    ) -> None:
        """
        Visualize the comparison results between raw and synthetic data in an HTML file.

        This method generates various plots and metrics to compare the raw and synthetic data, 
        including statistical metrics, classification metrics, feature importance, and causal analysis.
        The results are then rendered into an HTML template and saved to a file.

        Parameters
        ----------
        output_path : str
            The path where the HTML file will be saved.

        Raises
        ------
        ValueError
            If the `run` method has not been executed before calling this method.
        """
        if not self.results:
            raise ValueError('Please execute the run method first.')

        # Detect duplicate situation
        detect_df = pd.concat(
            [self.raw_data.drop_duplicates(), self.synthetic_data]
        )
        duplicate_situation = True if detect_df.duplicated().sum() > 0 else False

        # Define column types
        column_types = self.results['Statistical Test']['column_types']
        # Generate basic statistical metrics
        stats = self._generate_column_stats(column_types)

        # Generate correlation plots
        correlation_plots = self._generate_correlation_plots(column_types)

        classification_metrics = self.results['Classification'][0].to_dict(
            orient='records')

        feature_importance = self._generate_feature_importance()

        causal_comparison = self._generate_causal_comparison()

        all_data = {
            'duplicate_situation': duplicate_situation,
            'column_types': column_types,
            'stats': stats,
            'correlation_plots': correlation_plots,
            'classification_metrics': classification_metrics,
            'feature_importance': feature_importance,
            'causal_comparison': causal_comparison
        }

        json_data = json.dumps(all_data)

        env = Environment(
            loader=PackageLoader('sdqc_integration', 'templates'),
            autoescape=select_autoescape(['html', 'xml'])
        )
        template = env.get_template('index.html')

        html_content = template.render(json_data=json_data)

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)

        print(f"Results have been saved to {output_path}")
