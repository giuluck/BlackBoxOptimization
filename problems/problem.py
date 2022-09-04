from typing import Tuple, Optional

import numpy as np
from pyDOE import lhs
from scipy.stats.distributions import uniform


class Problem:
    def __init__(self, *bounds: Tuple[Optional[float], Optional[float]]):
        """Builds an unconstrained problem with the defined bounds and infers its input dimension."""
        self.input_shape: int = len(bounds)
        self.upper_bounds: np.ndarray = np.array([float('inf') if ub is None else ub for _, ub in bounds])
        self.lower_bounds: np.ndarray = np.array([-float('inf') if lb is None else lb for lb, _ in bounds])

    def sample(self, n: int) -> np.ndarray:
        """Samples data points within the bounded region using a latin hypercube sampling strategy."""
        x = lhs(self.input_shape, samples=n)
        return uniform(loc=self.lower_bounds, scale=self.upper_bounds - self.lower_bounds).ppf(x)

    def evaluate(self, x: np.ndarray) -> np.ndarray:
        """Evaluates the black-box function."""
        raise NotImplementedError()
