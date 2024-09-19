import warnings
from typing import Optional, Dict, Union, List, Tuple

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score,
    f1_score
)
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from ..utils import combine_data_and_labels

# Ignore warnings
warnings.filterwarnings('ignore')


class ClassificationModel:
    """
    A class for training and evaluating binary classification models to distinguish between real and synthetic data.

    Parameters
    ----------
    raw_data : pd.DataFrame
        DataFrame containing real data samples.
    synthetic_data : pd.DataFrame
        DataFrame containing synthetic data samples.
    model_name : Union[str, List[str]], optional
        Name(s) of the model(s) to use for classification (default is 'rf').
        Available models: 'svm', 'rf', 'xgb', 'lgbm'.
    test_size : float, optional
        The proportion of the dataset to include in the test split (default is 0.2).
    random_seed : int, optional
        The random seed for reproducibility (default is 17).
    model_params : Dict[str, dict], optional
        A dictionary containing model names as keys and their corresponding parameter dictionaries as values.
        If not provided, default parameters will be used for each model.
    """

    def __init__(
            self,
            raw_data: pd.DataFrame,
            synthetic_data: pd.DataFrame,
            model_name: Union[str, List[str]] = 'rf',
            test_size: float = 0.2,
            random_seed: int = 17,
            model_params: Optional[Dict[str, dict]] = None
    ) -> None:
        self.raw_data = raw_data
        self.synthetic_data = synthetic_data
        self.model_name = model_name
        self.test_size = test_size
        self.random_seed = random_seed
        self.model_params = model_params
        self.modelList = ['svm', 'rf', 'xgb', 'lgbm']

        # Combine data and labels, and split into train and test sets
        X, y = combine_data_and_labels(self.raw_data, self.synthetic_data)

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_seed
        )

        self.modelDict = self._initialize_models()

    def _initialize_models(self) -> Dict[str, object]:
        """
        Initialize classification models with custom parameters if provided, else use default parameters.

        Returns
        -------
        Dict[str, object]
            A dictionary containing model names as keys and their corresponding initialized model objects as values.
        """
        for model_name in self.modelList:
            if self.model_params is not None:
                if model_name not in self.model_params:
                    self.model_params[model_name] = {}
            else:
                self.model_params = {}
                self.model_params[model_name] = {}

        for model_name in self.model_params.keys():
            if 'random_state' not in self.model_params[model_name]:
                self.model_params[model_name]['random_state'] = self.random_seed

            if model_name == 'svm':
                if ('probability' not in self.model_params[model_name]) or\
                        (not self.model_params[model_name]['probability']):
                    self.model_params[model_name]['probability'] = True

            if (model_name == 'lgbm') and\
                    ('verbose' not in self.model_params[model_name]):
                self.model_params[model_name]['verbose'] = -1

        # Initialize models with parameters
        return {
            'svm': SVC(**self.model_params['svm']),
            'rf': RandomForestClassifier(**self.model_params['rf']),
            'xgb': XGBClassifier(**self.model_params['xgb']),
            'lgbm': LGBMClassifier(**self.model_params['lgbm'])
        }

    def _evaluate_model(self, model_name: str, model: object) -> pd.DataFrame:
        """
        Train and evaluate a single classification model.

        Parameters
        ----------
        model_name : str
            The name of the model.
        model : object
            The initialized model object.

        Returns
        -------
        pd.DataFrame
            A DataFrame containing performance metrics for the model.
        """
        model.fit(self.X_train, self.y_train)
        y_pred = model.predict(self.X_test)
        y_proba = model.predict_proba(self.X_test)[:, 1]

        # Calculate performance metrics
        metrics = {
            'Model': model_name,
            'Accuracy': accuracy_score(self.y_test, y_pred),
            'Precision': precision_score(self.y_test, y_pred),
            'Recall': recall_score(self.y_test, y_pred),
            'F1': f1_score(self.y_test, y_pred),
            'AUC': roc_auc_score(self.y_test, y_proba)
        }

        return pd.DataFrame([metrics])

    def train_and_evaluate_models(self) -> Tuple[pd.DataFrame, List[object]]:
        """
        Train and evaluate all specified classification models.

        Returns
        -------
        Tuple[pd.DataFrame, List[object]]
            A tuple containing:
            - pd.DataFrame: A DataFrame containing performance metrics for all the models.
            - List[object]: A list of trained model objects.
        """
        metrics_results = []
        model_results = []

        if isinstance(self.model_name, str):
            self.model_name = [self.model_name]

        # Train and evaluate model(s)
        for model_name in self.model_name:
            if model_name not in self.modelList:
                raise ValueError(
                    f"Model '{model_name}' not found. "
                    f"Available models are: {', '.join(self.modelList)}."
                )
            model = self.modelDict[model_name]
            model_metrics = self._evaluate_model(model_name, model)
            metrics_results.append(model_metrics)
            model_results.append(model)

        return pd.concat(metrics_results, ignore_index=True), model_results
