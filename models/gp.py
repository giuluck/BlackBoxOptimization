from typing import Any

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor

from models.model import Model


class GaussianProcess(Model):
    def __init__(self, kernel: Any = None, alpha: float = 1e-10, **gp_kwargs):
        """Wraps a scikit-learn GaussianProcessRegressor model."""
        self.model = GaussianProcessRegressor(kernel=kernel, alpha=alpha, **gp_kwargs)

    def fit(self, x: np.ndarray, y: np.ndarray):
        self.model.fit(x, y)

    def mean(self, x: np.ndarray) -> np.ndarray:
        return self.model.predict(x).flatten()

    def std(self, x: np.ndarray) -> np.ndarray:
        return self.model.predict(x, return_std=True)[1]
