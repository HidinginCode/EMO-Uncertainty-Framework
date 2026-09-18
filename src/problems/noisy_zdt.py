"""This file contains the implementation of noisy ZDT."""
from __future__ import annotations

# Make other project modules accessible
import sys
import os
sys.path.insert(1, os.path.join(os.path.dirname(__file__), ".."))

# Logger import
from config.logging_mixin import LoggingMixin

# Pymoo imports
from pymoo.problems.multi import ZDT1, ZDT2, ZDT3, ZDT4, ZDT5, ZDT6

# Package imports
import numpy as np


def _sample_noise(noise_type: str, noise_std: float, shape: tuple) -> np.ndarray:
    """Return zero-mean noise of the requested type and shape.

    Args:
        noise_type: "gaussian", "poisson", "uniform", "rademacher"
        noise_std:  Standard deviation (Gaussian) / rate parameter (Poisson).
        shape:      Shape of the output array.

    Returns:
        Zero-mean noise array.
    """
    match noise_type:
        case "gaussian":
            return np.random.normal(0.0, noise_std, shape)
        case "poisson":
            # Poisson has mean = lam; subtract lam to centre around zero.
            return np.random.poisson(lam=noise_std, size=shape).astype(float) - noise_std
        case "uniform":
            # Uniform on [-a, a] with a = sqrt(3)*noise_std so that Var = noise_std².
            # Sub-Gaussian parameter: a² = 3*noise_std².
            a = np.sqrt(3) * noise_std
            return np.random.uniform(-a, a, shape)
        case "rademacher":
            # ±noise_std with equal probability. Var = noise_std².
            # Sub-Gaussian parameter: noise_std².
            return np.where(np.random.random(shape) < 0.5, -noise_std, noise_std)
        case _:
            raise ValueError(
                f"'{noise_type}' is either an unknown or a not yet implemented noise type!"
            )

class NZDT1(ZDT1, LoggingMixin):
    """Noisy ZDT1 with additive zero-mean noise on the objectives."""

    def __init__(self, n_var=30, noise_std: float = 0.1,
                 noise_type: str = "gaussian", **kwargs):
        self.noise_std = noise_std
        self.noise_type = noise_type
        super().__init__(n_var, **kwargs)
        self.logger.info(f"Instance of {self.__class__.__name__} with noise of type {self.noise_type} and std {noise_std} created.")

    def _evaluate(self, x, out, *args, **kwargs):
        super()._evaluate(x, out, *args, **kwargs)
        out["F"] += _sample_noise(self.noise_type, self.noise_std, out["F"].shape)

    def evaluate_noiseless(self, X: np.ndarray) -> np.ndarray:
        out = {}
        super()._evaluate(X, out)
        return out["F"]

class NZDT2(ZDT2, LoggingMixin):
    """Noisy ZDT2 with additive zero-mean noise on the objectives."""

    def __init__(self, n_var=30, noise_std: float = 0.1,
                 noise_type: str = "gaussian", **kwargs):
        self.noise_std = noise_std
        self.noise_type = noise_type
        super().__init__(n_var, **kwargs)
        self.logger.info(f"Instance of {self.__class__.__name__} with noise of type {self.noise_type} and std {noise_std} created.")

    def _evaluate(self, x, out, *args, **kwargs):
        super()._evaluate(x, out, *args, **kwargs)
        out["F"] += _sample_noise(self.noise_type, self.noise_std, out["F"].shape)

    def evaluate_noiseless(self, X: np.ndarray) -> np.ndarray:
        out = {}
        super()._evaluate(X, out)
        return out["F"]

class NZDT3(ZDT3, LoggingMixin):
    """Noisy ZDT3 with additive zero-mean noise on the objectives."""

    def __init__(self, n_var=30, noise_std: float = 0.1,
                 noise_type: str = "gaussian", **kwargs):
        self.noise_std = noise_std
        self.noise_type = noise_type
        super().__init__(n_var, **kwargs)
        self.logger.info(f"Instance of {self.__class__.__name__} with noise of type {self.noise_type} and std {noise_std} created.")

    def _evaluate(self, x, out, *args, **kwargs):
        super()._evaluate(x, out, *args, **kwargs)
        out["F"] += _sample_noise(self.noise_type, self.noise_std, out["F"].shape)

    def evaluate_noiseless(self, X: np.ndarray) -> np.ndarray:
        out = {}
        super()._evaluate(X, out)
        return out["F"]

class NZDT4(ZDT4, LoggingMixin):
    """Noisy ZDT4 with additive zero-mean noise on the objectives."""

    def __init__(self, n_var=30, noise_std: float = 0.1,
                 noise_type: str = "gaussian", **kwargs):
        self.noise_std = noise_std
        self.noise_type = noise_type
        super().__init__(n_var, **kwargs)
        self.logger.info(f"Instance of {self.__class__.__name__} with noise of type {self.noise_type} and std {noise_std} created.")

    def _evaluate(self, x, out, *args, **kwargs):
        super()._evaluate(x, out, *args, **kwargs)
        out["F"] += _sample_noise(self.noise_type, self.noise_std, out["F"].shape)

    def evaluate_noiseless(self, X: np.ndarray) -> np.ndarray:
        out = {}
        super()._evaluate(X, out)
        return out["F"]

class NZDT5(ZDT5, LoggingMixin):
    """Noisy ZDT5 with additive zero-mean noise on the objectives."""

    def __init__(self, m: int = 11, n: int = 5, noise_std: float = 0.1,
                 noise_type: str = "gaussian", **kwargs):
        # Unlike the other ZDT problems, ZDT5's decision-space size is derived
        # from (m, n) as 30 + n*(m-1), not passed directly as n_var.
        self.noise_std = noise_std
        self.noise_type = noise_type
        super().__init__(m, n, **kwargs)
        self.logger.info(f"Instance of {self.__class__.__name__} with noise of type {self.noise_type} and std {noise_std} created.")

    def _evaluate(self, x, out, *args, **kwargs):
        super()._evaluate(x, out, *args, **kwargs)
        out["F"] += _sample_noise(self.noise_type, self.noise_std, out["F"].shape)

    def evaluate_noiseless(self, X: np.ndarray) -> np.ndarray:
        out = {}
        super()._evaluate(X, out)
        return out["F"]

class NZDT6(ZDT6, LoggingMixin):
    """Noisy ZDT6 with additive zero-mean noise on the objectives."""

    def __init__(self, n_var=30, noise_std: float = 0.1,
                 noise_type: str = "gaussian", **kwargs):
        self.noise_std = noise_std
        self.noise_type = noise_type
        super().__init__(n_var, **kwargs)
        self.logger.info(f"Instance of {self.__class__.__name__} with noise of type {self.noise_type} and std {noise_std} created.")

    def _evaluate(self, x, out, *args, **kwargs):
        super()._evaluate(x, out, *args, **kwargs)
        out["F"] += _sample_noise(self.noise_type, self.noise_std, out["F"].shape)

    def evaluate_noiseless(self, X: np.ndarray) -> np.ndarray:
        out = {}
        super()._evaluate(X, out)
        return out["F"]
