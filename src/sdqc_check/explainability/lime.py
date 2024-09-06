from .base import BaseFeatureImportance
import pandas as pd
import lime.lime_tabular


class LimeFeatureImportance(BaseFeatureImportance):
    """
    LIME (Local Interpretable Model-agnostic Explanations) feature importance calculator.

    This class uses the LIME library to compute feature importance scores.

    Parameters
    ----------
    See base class BaseFeatureImportance for parameter details.
    """

    def compute_feature_importance(self) -> pd.DataFrame:
        """
        Compute LIME feature importance.

        This method explains each instance in the test set and averages the importance scores.

        Returns
        -------
        pd.DataFrame
            A DataFrame with features and their LIME importance scores, sorted in descending order.
        """
        explainer = lime.lime_tabular.LimeTabularExplainer(
            self.X_train.values,
            feature_names=self.X_train.columns,
            class_names=['0', '1'],
            mode='classification',
            random_state=self.random_seed
        )
        importances = [dict(explainer.explain_instance(
            self.X_test.iloc[i],
            self.model.predict_proba,
            num_features=len(self.X_test.columns)
        ).as_list()) for i in range(len(self.X_test))]

        importance = pd.DataFrame(importances).mean().reset_index()
        importance.columns = ['feature', 'importance']
        return importance.sort_values('importance', ascending=False)
