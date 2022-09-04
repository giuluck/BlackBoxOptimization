import numpy as np

from acquisitions.acquisition import Acquisition


class LowerConfidenceBound(Acquisition):
    def __init__(self, theta: float = 1.0):
        """Builds an acquisition function f(mu, sigma) -> mu - theta * sigma."""
        super(LowerConfidenceBound, self).__init__()

        self.theta: float = theta
        """The hyper-parameter used to balance between exploration and exploitation"""

    def value(self, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
        return mu - self.theta * sigma
