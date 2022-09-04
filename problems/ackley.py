from typing import Tuple, Optional

import numpy as np

from problems.problem import Problem


class AckleyProblem(Problem):
    def __init__(self,
                 *bounds: Tuple[Optional[float], Optional[float]],
                 a: float = 20,
                 b: float = 0.2,
                 c: float = 2 * np.pi):
        """Builds a problem instance using the Ackley function as black-box function."""
        assert len(bounds) > 0, "Each tuple represents a dimension, therefore there should be at least one."
        super(AckleyProblem, self).__init__(*bounds)
        self.a: float = a
        self.b: float = b
        self.c: float = c

    def evaluate(self, x: np.ndarray) -> np.ndarray:
        d = self.input_shape
        term1 = -self.a * np.exp(-self.b * np.sqrt(np.sum(x ** 2, axis=-1) / d))
        term2 = -np.exp(np.sum(np.cos(self.c * x), axis=-1) / d)
        return term1 + term2 + self.a + np.e
