import warnings
from typing import Any, Dict, List, Optional, Union

import pandas as pd
from sdv.metadata import SingleTableMetadata
from ydata_synthetic.synthesizers import ModelParameters, TrainParameters
from ydata_synthetic.synthesizers.regular import RegularSynthesizer

from .base_synthesizer import BaseSynthesizer

# Ignore warnings
warnings.filterwarnings('ignore')


class YDataSynthesizer(BaseSynthesizer):
    """
    A synthesizer for generating synthetic data using various generative models from YData Synthetic.

    Parameters
    ----------
    data : pd.DataFrame
        The input data to synthesize from.
    metadata : Optional[Union[Dict[str, Any], str, SingleTableMetadata]]
        Metadata describing the input data. It can be a dictionary, a JSON file path,
        or an instance of SingleTableMetadata. If not provided, metadata will be inferred
        from the input data.
    random_seed : int, optional
        Seed for random number generation (default is 17).
    model_name : Union[List[str], str], optional
        Name(s) of the model(s) to use for synthesis (default is 'fast').
    model_args : Optional[Union[Dict[str, Any], ModelParameters]], optional
        Arguments for the model. It can be a dictionary or an instance of ModelParameters.
    train_args : Optional[Union[Dict[str, Any], TrainParameters]], optional
        Training arguments for the model. It can be a dictionary or an instance of TrainParameters.
    addition_args : Optional[Dict[str, Any]], optional
        Additional arguments for the model.
    epochs : Optional[int], optional
        Number of epochs for training (default is 100).
    num_rows : Optional[int], optional
        Number of synthetic rows to generate (default is the same as input data).
    """

    def __init__(
        self,
        data: pd.DataFrame,
        metadata: Optional[Union[Dict[str, Any],
                                 str, SingleTableMetadata]] = None,
        random_seed: int = 17,
        model_name: Union[List[str], str] = 'fast',
        model_args: Optional[Union[Dict[str, Any], ModelParameters]] = None,
        train_args: Optional[Union[Dict[str, Any], TrainParameters]] = None,
        addition_args: Optional[Dict[str, Any]] = None,
        epochs: Optional[int] = 100,
        num_rows: Optional[int] = None,
    ) -> None:
        super().__init__(data, metadata)
        self.metadata = self.check_metadata().to_dict()
        self.random_seed = random_seed
        self.model_name = model_name
        self.model_args = model_args
        self.train_args = train_args
        self.addition_args = addition_args
        self.epochs = epochs
        self.num_rows = num_rows
        self.modelList = [
            'gan', 'wgan', 'wgangp', 'dragan', 'cramer', 'ctgan', 'fast'
        ]

    def validate_arg(self) -> None:
        """
        Validate and set the model, training, and additional arguments.
        """
        super().validate_arg()

        # Validate model arguments
        if isinstance(self.model_args, ModelParameters):
            self.model_args = self.model_args
        elif isinstance(self.model_args, Dict):
            model_args = {
                'batch_size': 100,
                'lr': 2e-4,
                'betas': (0.5, 0.9),
                'noise_dim': 64
            }
            model_args.update(self.model_args)
            self.model_args = ModelParameters(**model_args)
        else:
            self.model_args = ModelParameters(
                batch_size=100,
                lr=2e-4,
                betas=(0.5, 0.9),
                noise_dim=64,
            )

        # Validate training arguments
        if isinstance(self.train_args, TrainParameters):
            self.train_args = self.train_args
        elif isinstance(self.train_args, Dict):
            self.train_args = TrainParameters(**self.train_args)
        else:
            self.train_args = TrainParameters(epochs=self.epochs)

        # Validate additional arguments
        if isinstance(self.addition_args, Dict):
            self.addition_args = self.addition_args
        elif self.addition_args is None:
            if self.model_name == 'wgan':
                self.addition_args = {'n_critic': 10}
            elif self.model_name == 'dragan':
                self.addition_args = {'n_discriminator': 3}
            elif self.model_name == 'fast':
                self.addition_args = {'random_state': self.random_seed}
            else:
                self.addition_args = {}
        else:
            self.addition_args = {}

    def fit(self) -> RegularSynthesizer:
        """
        Fit the specified model to the data.

        Returns
        -------
        RegularSynthesizer
            The fitted synthesizer.
        """
        self.validate_arg()

        # Identify numerical and categorical columns based on metadata
        num_cols = [
            col for col, sdtype in self.metadata['columns'].items()
            if sdtype['sdtype'] in ['numerical', 'datetime']
        ]
        cat_cols = [
            col for col, sdtype in self.metadata['columns'].items()
            if sdtype['sdtype'] == 'categorical'
        ]

        # Prepare keyword arguments for fitting the synthesizer
        fit_kwargs = {
            'data': self.data,
            'num_cols': num_cols,
            'cat_cols': cat_cols,
        }

        if self.model_name != 'fast':
            fit_kwargs['train_arguments'] = self.train_args
            synth = RegularSynthesizer(
                modelname=self.model_name,
                model_parameters=self.model_args,
                **self.addition_args,
            )
        else:
            synth = RegularSynthesizer(
                modelname=self.model_name,
                **self.addition_args,
            )

        synth.fit(**fit_kwargs)
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
            return self.fit().sample(self.num_rows)[:self.num_rows]
        elif isinstance(self.model_name, List):
            model_name = self.model_name.copy()
            model_args = self.model_args.copy() if self.model_args else None
            train_args = self.train_args.copy() if self.train_args else None
            addition_args = self.addition_args.copy() if self.addition_args else None

            results = {}

            for _model_name in model_name:
                self.model_name = _model_name

                if isinstance(model_args, Dict):
                    self.model_args = model_args.get(_model_name, None)
                else:
                    self.model_args = None

                if isinstance(train_args, Dict):
                    self.train_args = train_args.get(_model_name, None)
                else:
                    self.train_args = None

                if isinstance(addition_args, Dict):
                    self.addition_args = addition_args.get(_model_name, None)
                else:
                    self.addition_args = None

                result = self.fit().sample(self.num_rows)[:self.num_rows]
                results[_model_name] = result
            return results
