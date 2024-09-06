import warnings
from typing import Any, Dict, List, Optional, Union

import pandas as pd
from sdv.metadata import SingleTableMetadata
from sdv.single_table import (
    CopulaGANSynthesizer,
    CTGANSynthesizer,
    GaussianCopulaSynthesizer,
    TVAESynthesizer,
)

from .base_synthesizer import BaseSynthesizer

# Ignore warnings
warnings.filterwarnings('ignore')


class SDVSynthesizer(BaseSynthesizer):
    """
    A synthesizer for generating synthetic data using various generative models from SDV.

    Parameters
    ----------
    data : pd.DataFrame
        The input data to synthesize from.
    metadata : Optional[Union[Dict[str, Any], str, SingleTableMetadata]]
        Metadata describing the input data.
    random_seed : int, optional
        Seed for random number generation (default is 17).
    model_name : Union[List[str], str], optional
        Name(s) of the model(s) to use for synthesis (default is 'tvae').
    model_args : Optional[Dict[str, Dict]], optional
        Arguments for the model.
    num_rows : Optional[int], optional
        Number of synthetic rows to generate (default is the same as input data).
    """

    def __init__(
        self,
        data: pd.DataFrame,
        metadata: Optional[Union[Dict[str, Any],
                                 str, SingleTableMetadata]] = None,
        random_seed: int = 17,
        model_name: Union[List[str], str] = "tvae",
        model_args: Optional[Dict[str, Dict]] = None,
        num_rows: Optional[int] = None,
    ) -> None:
        super().__init__(data, metadata)
        self.metadata = self.check_metadata()
        self.random_seed = random_seed
        self.model_name = model_name
        self.model_args = model_args
        self.num_rows = num_rows
        self.modelDict = {
            'copulagan': CopulaGANSynthesizer,
            'gaussiancopula': GaussianCopulaSynthesizer,
            'ctgan': CTGANSynthesizer,
            'tvae': TVAESynthesizer
        }
        self.modelList = list(self.modelDict.keys())

    def validate_arg(self) -> None:
        """
        Validate and set the model arguments.
        """
        super().validate_arg()

        # Validate model arguments
        if not isinstance(self.model_args, Dict):
            self.model_args = {}

    def fit(self) -> Any:
        """
        Fit the specified model to the data.

        Returns
        -------
        Any
            The fitted synthesizer model.
        """
        self.validate_arg()

        synth = self.modelDict[self.model_name](
            metadata=self.metadata, **self.model_args
        )

        synth.fit(data=self.data)
        return synth

    def generate(self) -> Union[pd.DataFrame, Dict[str, pd.DataFrame]]:
        """
        Generate synthetic data based on the fitted model.

        Returns
        -------
        Union[pd.DataFrame, Dict[str, pd.DataFrame]]
            A DataFrame of synthetic data or a dictionary of results.
        """
        self.num_rows = self.num_rows if self.num_rows else self.data.shape[0]
        self.set_seed(self.random_seed)

        if isinstance(self.model_name, str):
            return self.fit().sample(self.num_rows)
        elif isinstance(self.model_name, List):
            model_name = self.model_name.copy()
            model_args = self.model_args.copy() if self.model_args else None
            results = {}

            for _model_name in model_name:
                self.model_name = _model_name

                if isinstance(model_args, Dict):
                    self.model_args = model_args.get(_model_name, None)
                else:
                    self.model_args = None

                result = self.fit().sample(self.num_rows)
                results[_model_name] = result
            return results
