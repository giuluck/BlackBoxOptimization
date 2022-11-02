import numpy as np

from acquisitions.acquisition import Acquisition


class UpperConfidenceBound(Acquisition):
    def __init__(self, eps: float = 1.0):
        """Builds an upper confidence bound acquisition function."""
        super(UpperConfidenceBound, self).__init__()

        self.eps: float = eps
        """The hyper-parameter used to balance between exploration and exploitation"""

    def value(self, mu: np.ndarray, sigma: np.ndarray, best: float) -> np.ndarray:
        return mu + self.eps * sigma
