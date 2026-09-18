"""This file contains all noisy DTLZ implementations"""
from __future__ import annotations

# Make other project modules accessible
import sys
import os
sys.path.insert(1, os.path.join(os.path.dirname(__file__), ".."))

# Logger import
from config.logging_mixin import LoggingMixin

# Pymoo imports
from pymoo.problems.many import DTLZ1, DTLZ2, DTLZ3, DTLZ7
from pymoo.util.ref_dirs import get_reference_directions
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

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


class NDTLZ1(DTLZ1, LoggingMixin):
    """Adaptation of DTLZ1 with additive zero-mean noise on the objectives."""

    def __init__(self, n_var=7, n_obj=3, noise_std: float = 0.1,
                 noise_type: str = "gaussian", **kwargs):
        super().__init__(n_var, n_obj, **kwargs)
        self.noise_std = noise_std
        self.noise_type = noise_type
        (f"Instance of {self.__class__.__name__} with noise of type {self.noise_type} and std {noise_std} created.")

    def _evaluate(self, x, out, *args, **kwargs):
        super()._evaluate(x, out, *args, **kwargs)
        out["F"] += _sample_noise(self.noise_type, self.noise_std, out["F"].shape)

    def evaluate_noiseless(self, X: np.ndarray) -> np.ndarray:
        out = {}
        super()._evaluate(X, out)
        return out["F"]


class NDTLZ2(DTLZ2, LoggingMixin):
    """Adaptation of DTLZ2 with additive zero-mean noise on the objectives."""

    def __init__(self, n_var=10, n_obj=3, noise_std: float = 0.1,
                 noise_type: str = "gaussian", **kwargs):
        super().__init__(n_var, n_obj, **kwargs)
        self.noise_std = noise_std
        self.noise_type = noise_type
        (f"Instance of {self.__class__.__name__} with noise of type {self.noise_type} and std {noise_std} created.")

    def _evaluate(self, x, out, *args, **kwargs):
        super()._evaluate(x, out, *args, **kwargs)
        out["F"] += _sample_noise(self.noise_type, self.noise_std, out["F"].shape)

    def evaluate_noiseless(self, X: np.ndarray) -> np.ndarray:
        out = {}
        super()._evaluate(X, out)
        return out["F"]


class NDTLZ3(DTLZ3, LoggingMixin):
    """Adaptation of DTLZ3 with additive zero-mean noise on the objectives."""

    def __init__(self, n_var=10, n_obj=3, noise_std: float = 0.1,
                 noise_type: str = "gaussian", **kwargs):
        super().__init__(n_var, n_obj, **kwargs)
        self.noise_std = noise_std
        self.noise_type = noise_type
        (f"Instance of {self.__class__.__name__} with noise of type {self.noise_type} and std {noise_std} created.")

    def _evaluate(self, x, out, *args, **kwargs):
        super()._evaluate(x, out, *args, **kwargs)
        out["F"] += _sample_noise(self.noise_type, self.noise_std, out["F"].shape)

    def evaluate_noiseless(self, X: np.ndarray) -> np.ndarray:
        out = {}
        super()._evaluate(X, out)
        return out["F"]


class NDTLZ7(DTLZ7, LoggingMixin):
    """Adaptation of DTLZ7 with additive zero-mean noise on the objectives."""

    def __init__(self, n_var=10, n_obj=3, noise_std: float = 0.1,
                 noise_type: str = "gaussian", **kwargs):
        super().__init__(n_var, n_obj, **kwargs)
        self.noise_std = noise_std
        self.noise_type = noise_type
        (f"Instance of {self.__class__.__name__} with noise of type {self.noise_type} and std {noise_std} created.")

    def _evaluate(self, x, out, *args, **kwargs):
        super()._evaluate(x, out, *args, **kwargs)
        out["F"] += _sample_noise(self.noise_type, self.noise_std, out["F"].shape)

    def evaluate_noiseless(self, X: np.ndarray) -> np.ndarray:
        out = {}
        super()._evaluate(X, out)
        return out["F"]

    def _calc_pareto_front(self, *args, **kwargs):
        n_f = self.n_obj - 1
        n_grid = 100 if n_f == 1 else 50
        grids = np.meshgrid(*[np.linspace(0.0, 1.0, n_grid)] * n_f)
        Fi = [g.ravel() for g in grids]
        h = float(self.n_obj) - sum(
            (Fi[i] / 2.0) * (1.0 + np.sin(3.0 * np.pi * Fi[i])) for i in range(n_f)
        )
        F_last = 2.0 * h  # (1 + g*) = 2
        valid = F_last >= 0.0
        pf = np.column_stack([*[Fi[i][valid] for i in range(n_f)], F_last[valid]])
        idx = NonDominatedSorting().do(pf, only_non_dominated_front=True)
        return pf[idx]
