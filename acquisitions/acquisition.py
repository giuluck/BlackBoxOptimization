import numpy as np


class Acquisition:
    def value(self, mu: np.ndarray, sigma: np.ndarray, best: float) -> np.ndarray:
        """Computes the acquisition function value based on the mean and the standard deviation."""
        raise NotImplementedError()
