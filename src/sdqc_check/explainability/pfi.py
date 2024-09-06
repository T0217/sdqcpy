from .base import BaseFeatureImportance
import pandas as pd
from sklearn.inspection import permutation_importance


class PFIFeatureImportance(BaseFeatureImportance):
    """
    Permutation Feature Importance (PFI) calculator.

    This class uses scikit-learn's permutation_importance to compute feature importance scores.

    Parameters
    ----------
    See base class BaseFeatureImportance for parameter details.
    """

    def compute_feature_importance(self) -> pd.DataFrame:
        """
        Compute Permutation Feature Importance.

        This method uses permutation_importance from scikit-learn to calculate feature importance.

        Returns
        -------
        pd.DataFrame
            A DataFrame with features and their PFI importance scores, sorted in descending order.
        """
        pfi = permutation_importance(
            self.model,
            self.X_test,
            self.y_test,
            n_repeats=5,
            random_state=self.random_seed
        )
        importance = pd.DataFrame({
            'feature': self.X_test.columns,
            'importance': pfi.importances_mean
        })
        return importance.sort_values('importance', ascending=False)
