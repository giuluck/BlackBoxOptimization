from abc import ABC
from typing import Any

import numpy as np
from sklearn.gaussian_process.kernels import Kernel, RBF, ConstantKernel as Const


class Model:
    @staticmethod
    def abs_std(samples: np.ndarray, references: np.ndarray) -> np.ndarray:
        """Computes the standard deviation as the l1 distance from the nearest reference point."""
        samples = np.repeat([samples], len(references), axis=0).transpose((1, 0, 2))
        distances = np.abs(samples - references).sum(axis=2)
        return distances.min(axis=1)

    @staticmethod
    def sqr_std(samples: np.ndarray, references: np.ndarray) -> np.ndarray:
        """Computes the standard deviation as the l2 distance from the nearest reference point."""
        samples = np.repeat([samples], len(references), axis=0).transpose((1, 0, 2))
        distances = np.square(samples - references).sum(axis=2)
        return distances.min(axis=1)

    @staticmethod
    def log_std(samples: np.ndarray, references: np.ndarray) -> np.ndarray:
        """Computes the standard deviation as the log absolute distance from the nearest reference point."""
        samples = np.repeat([samples], len(references), axis=0).transpose((1, 0, 2))
        distances = np.log(np.abs(samples - references) + 1).sum(axis=2)
        return distances.min(axis=1)

    @staticmethod
    def gp_std(samples: np.ndarray,
               references: np.ndarray,
               kernel: Kernel = Const(1.0, constant_value_bounds="fixed") * RBF(1.0, length_scale_bounds="fixed"),
               alpha: float = 1e-10) -> np.ndarray:
        """Computes the standard deviation using the gaussian processes formulation. The default kernel is the same
        default kernel as in scikit-learn, while the code to compute the standard deviation is taken from:
        https://stats.stackexchange.com/questions/330185/how-to-calculate-the-standard-deviation-for-a-gaussian-process
        """
        kss = kernel(samples, samples)
        krs = kernel(references, samples)
        krr = kernel(references, references)
        chl = np.linalg.cholesky(krr + alpha * np.eye(len(references)))
        lk = np.linalg.solve(chl, krs)
        var = np.diag(kss) - np.sum(lk ** 2, axis=0)
        return np.sqrt(var)

    def fit(self, x: np.ndarray, y: np.ndarray):
        """Fits the machine learning model."""
        raise NotImplementedError()

    def mean(self, x: np.ndarray) -> np.ndarray:
        """Predicts the expected target based on the given input data."""
        raise NotImplementedError()

    def std(self, x: np.ndarray) -> np.ndarray:
        """Predicts the target standard deviation based on the given input data."""
        raise NotImplementedError()


class EmbedModel(Model, ABC):
    def embed(self) -> Any:
        """Embeds the model into the Empirical Model Learning framework."""
        return NotImplementedError()
