from typing import Dict, Any, Optional, Callable, List, Union

import numpy as np
from eml.net.reader import keras_reader
from keras.layers import Dense
from keras.models import Sequential, clone_model

from models.model import EmbedModel


class NeuralNetwork(EmbedModel):
    def __init__(self,
                 units: List[int] = (16, 16),
                 std: Callable = EmbedModel.gp_std,
                 loss: Any = 'mse',
                 optimizer: Any = 'adam',
                 metrics: Optional[List] = None,
                 loss_weights: Union[None, List, Dict] = None,
                 weighted_metrics: Optional[List] = None,
                 run_eagerly: bool = False,
                 epochs: int = 100,
                 shuffle: bool = True,
                 validation_split: float = 0.,
                 batch_size: Optional[int] = None,
                 class_weight: Optional[Dict] = None,
                 callbacks: Optional[List] = None,
                 verbose: Union[bool, str] = False,
                 warm_start: Union[bool, int] = False,
                 **layer_kwargs: Any):
        """A Multi-layer Perceptron model, with 'relu' activation in the hidden layers and a single output unit with
        linear activation function. The warm start policy works as follow:
            - for level 0 (False), the weights are reinitialized and the model is trained from scratch;
            - for level 1 (True), the optimizer is reinitialized while the weights are kept from the previous iteration;
            - for level 2, neither the weights nor the optimizer are reinitialized.
        """
        assert warm_start in [0, 1, 2, True, False], "'warm_start' must be either a boolean or an integer in {0, 1, 2}"
        super(NeuralNetwork, self).__init__()
        layers = [Dense(hu, activation='relu', **layer_kwargs) for hu in units] + [Dense(1, **layer_kwargs)]

        self.model: Sequential = Sequential(layers)
        """The neural model."""

        self.warm_start: int = warm_start if isinstance(warm_start, int) else int(warm_start)
        """The warm start level."""

        self.fit_kwargs: Dict[str, Any] = dict(
            epochs=epochs,
            shuffle=shuffle,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=callbacks,
            class_weight=class_weight,
            verbose=verbose
        )
        """Custom arguments to be passed to the '.fit()' method."""

        self.compile_kwargs: Dict[str, Any] = dict(
            loss=loss,
            optimizer=optimizer,
            metrics=metrics,
            loss_weights=loss_weights,
            weighted_metrics=weighted_metrics,
            run_eagerly=run_eagerly
        )
        """Custom arguments to be passed to the '.compile()' method."""

        self.std_fn: Callable = std
        """The function f(samples, references) -> std computing the standard deviation of sample points."""

        self.points: Optional[np.ndarray] = None
        """The reference points used for fitting."""

        # if the warm start involves the optimizer as well, pre-compile the model
        if self.warm_start == 2:
            self.model.compile(**self.compile_kwargs)

    def fit(self, x: np.ndarray, y: np.ndarray, sample_weight: Optional[np.ndarray] = None):
        # store the training data which will be used to compute the standard deviation
        self.points = x
        # depending on the warm start level, re-initialize the weights and/or the optimizer
        # (leverage the 'clone_model' utility to create a copy of the model structure with uninitialized weights)
        if self.warm_start == 0:
            self.model = clone_model(self.model)
            self.model.compile(**self.compile_kwargs)
        elif self.warm_start == 1:
            self.model.compile(**self.compile_kwargs)
        # eventually fit the model
        self.model.fit(x, y, sample_weight=sample_weight, **self.fit_kwargs)

    def mean(self, x: np.ndarray) -> np.ndarray:
        return self.model.predict(x, verbose=False).flatten()

    def std(self, x: np.ndarray) -> np.ndarray:
        return self.std_fn(samples=x, references=self.points)

    def embed(self) -> Any:
        return keras_reader.read_keras_sequential(self.model)
