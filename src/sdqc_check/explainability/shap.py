from .base import BaseFeatureImportance
import pandas as pd
import numpy as np
import shap
from sklearn.svm import SVC


class ShapFeatureImportance(BaseFeatureImportance):
    """
    SHAP (SHapley Additive exPlanations) feature importance calculator.

    This class uses the SHAP library to compute feature importance scores.

    Parameters
    ----------
    See base class BaseFeatureImportance for parameter details.
    """

    def compute_feature_importance(self) -> pd.DataFrame:
        """
        Compute SHAP feature importance.

        This method uses either KernelExplainer for SVC models or TreeExplainer for tree-based models.

        Returns
        -------
        pd.DataFrame
            A DataFrame with features and their SHAP importance scores, sorted in descending order.
        """
        if isinstance(self.model, SVC):
            explainer = shap.KernelExplainer(
                model=self.model.predict_proba,
                data=shap.sample(
                    self.X_train, 30) if self.X_train.shape[0] > 100 else self.X_train,
                link='logit'
            )
            shap_values = explainer.shap_values(self.X_test)[:, :, 0]

        else:
            explainer = shap.TreeExplainer(
                model=self.model,
                data=self.X_train,
                model_output='proability'
            )
            shap_values = explainer.shap_values(self.X_test)

        shap_df = pd.DataFrame(data=shap_values,
                               columns=self.X_test.columns)

        importance = pd.DataFrame({
            'feature': shap_df.columns,
            'importance': np.abs(shap_df.values).mean(axis=0)
        })
        return importance.sort_values('importance', ascending=False)
