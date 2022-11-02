import numpy as np
from scipy.stats.distributions import norm

from acquisitions.acquisition import Acquisition


class ExpectedImprovement(Acquisition):
    def __init__(self, eps: float = 0.0):
        """Builds a probability of improvement acquisition function."""
        super(ExpectedImprovement, self).__init__()

        self.eps: float = eps
        """The hyper-parameter used to balance between exploration and exploitation"""

    def value(self, mu: np.ndarray, sigma: np.ndarray, best: float) -> np.ndarray:
        val = mu - best - self.eps
        return val * norm.cdf(val / sigma) + sigma * norm.pdf(val / sigma)
