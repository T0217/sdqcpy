from .base import BaseFeatureImportance
import pandas as pd
import numpy as np


class ModelBasedFeatureImportance(BaseFeatureImportance):
    """
    Model-Based Feature Importance calculator.

    This class uses model-specific attributes to compute feature importance scores.

    Parameters
    ----------
    See base class BaseFeatureImportance for parameter details.
    """

    def compute_feature_importance(self) -> pd.DataFrame:
        """
        Compute Model-Based Feature Importance.

        This method uses model-specific attributes to calculate feature importance.

        Returns
        -------
        pd.DataFrame
            A DataFrame with features and their model-based importance scores, sorted in descending order.

        Raises
        ------
        AttributeError:
            If the model does not have the required attribute for feature importance.
        """
        if hasattr(self.model, "feature_importances_"):
            importances = self.model.feature_importances_
        elif hasattr(self.model, "coef_"):
            importances = np.abs(self.model.coef_[0])
        else:
            raise AttributeError(
                "No attribute 'feature_importances_' or 'coef_' found for the model."
            )
        importance = pd.DataFrame({
            'feature': self.X_train.columns,
            'importance': importances
        })
        return importance.sort_values('importance', ascending=False)
