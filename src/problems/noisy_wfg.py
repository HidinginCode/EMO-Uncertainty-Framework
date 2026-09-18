"""This file contains all noisy WFG implementations"""
from __future__ import annotations

# Make other project modules accessible
import sys
import os
sys.path.insert(1, os.path.join(os.path.dirname(__file__), ".."))

# Logger import
from config.logging_mixin import LoggingMixin

# Pymoo imports
from pymoo.problems.many import WFG1, WFG2, WFG3, WFG4, WFG5, WFG6, WFG7, WFG8, WFG9
from pymoo.util.ref_dirs import get_reference_directions

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

class NWFG1(WFG1, LoggingMixin):
    """This class represents the implementation of noisy WFG1."""
    
    def __init__(self, n_var=24, n_obj=3, k=None, l=None, noise_std =0.1, noise_type: str = "gaussian", **kwargs):
        self.noise_type = noise_type
        self.noise_std = noise_std
        super().__init__(n_var, n_obj, k, l, **kwargs)
        self.logger.info(f"Instance of {self.__class__.__name__} with noise of type {self.noise_type} and std {noise_std} created.")
    
    def _evaluate(self, x, out, *args, **kwargs):
        super()._evaluate(x, out, *args, **kwargs)
        out["F"]+=_sample_noise(noise_type=self.noise_type, noise_std=self.noise_std, shape=out["F"].shape)
        
    def evaluate_noiseless(self, X: np.ndarray) -> np.ndarray:
        """Evaluates a population without noise."""
        out = {}
        super()._evaluate(X, out)
        return out["F"]

class NWFG2(WFG2, LoggingMixin):
    """This class represents the implementation of noisy WFG2."""
    
    def __init__(self, n_var=24, n_obj=3, k=None, l=None, noise_std =0.1, noise_type: str = "gaussian", **kwargs):
        self.noise_type = noise_type
        self.noise_std = noise_std
        super().__init__(n_var, n_obj, k, l, **kwargs)
        self.logger.info(f"Instance of {self.__class__.__name__} with noise of type {self.noise_type} and std {noise_std} created.")
    
    def _evaluate(self, x, out, *args, **kwargs):
        super()._evaluate(x, out, *args, **kwargs)
        out["F"]+=_sample_noise(noise_type=self.noise_type, noise_std=self.noise_std, shape=out["F"].shape)
        
    def evaluate_noiseless(self, X: np.ndarray) -> np.ndarray:
        """Evaluates a population without noise."""
        out = {}
        super()._evaluate(X, out)
        return out["F"]

class NWFG3(WFG3, LoggingMixin):
    """This class represents the implementation of noisy WFG3."""
    
    def __init__(self, n_var=24, n_obj=3, noise_std =0.1, noise_type: str = "gaussian", **kwargs):
        self.noise_type = noise_type
        self.noise_std = noise_std
        super().__init__(n_var, n_obj,**kwargs)
        self.logger.info(f"Instance of {self.__class__.__name__} with noise of type {self.noise_type} and std {noise_std} created.")
    
    def _evaluate(self, x, out, *args, **kwargs):
        super()._evaluate(x, out, *args, **kwargs)
        out["F"]+=_sample_noise(noise_type=self.noise_type, noise_std=self.noise_std, shape=out["F"].shape)
        
    def evaluate_noiseless(self, X: np.ndarray) -> np.ndarray:
        """Evaluates a population without noise."""
        out = {}
        super()._evaluate(X, out)
        return out["F"]

class NWFG4(WFG4, LoggingMixin):
    """This class represents the implementation of noisy WFG4."""
    
    def __init__(self, n_var=24, n_obj=3, k=None, l=None, noise_std =0.1, noise_type: str = "gaussian", **kwargs):
        self.noise_type = noise_type
        self.noise_std = noise_std
        super().__init__(n_var, n_obj, k, l, **kwargs)
        self.logger.info(f"Instance of {self.__class__.__name__} with noise of type {self.noise_type} and std {noise_std} created.")
    
    def _evaluate(self, x, out, *args, **kwargs):
        super()._evaluate(x, out, *args, **kwargs)
        out["F"]+=_sample_noise(noise_type=self.noise_type, noise_std=self.noise_std, shape=out["F"].shape)
        
    def evaluate_noiseless(self, X: np.ndarray) -> np.ndarray:
        """Evaluates a population without noise."""
        out = {}
        super()._evaluate(X, out)
        return out["F"]

class NWFG5(WFG5, LoggingMixin):
    """This class represents the implementation of noisy WFG5."""
    
    def __init__(self, n_var=24, n_obj=3, k=None, l=None, noise_std =0.1, noise_type: str = "gaussian", **kwargs):
        self.noise_type = noise_type
        self.noise_std = noise_std
        super().__init__(n_var, n_obj, k, l, **kwargs)
        self.logger.info(f"Instance of {self.__class__.__name__} with noise of type {self.noise_type} and std {noise_std} created.")
    
    def _evaluate(self, x, out, *args, **kwargs):
        super()._evaluate(x, out, *args, **kwargs)
        out["F"]+=_sample_noise(noise_type=self.noise_type, noise_std=self.noise_std, shape=out["F"].shape)
        
    def evaluate_noiseless(self, X: np.ndarray) -> np.ndarray:
        """Evaluates a population without noise."""
        out = {}
        super()._evaluate(X, out)
        return out["F"]

class NWFG6(WFG6, LoggingMixin):
    """This class represents the implementation of noisy WFG6."""
    
    def __init__(self, n_var=24, n_obj=3, k=None, l=None, noise_std =0.1, noise_type: str = "gaussian", **kwargs):
        self.noise_type = noise_type
        self.noise_std = noise_std
        super().__init__(n_var, n_obj, k, l, **kwargs)
        self.logger.info(f"Instance of {self.__class__.__name__} with noise of type {self.noise_type} and std {noise_std} created.")
    
    def _evaluate(self, x, out, *args, **kwargs):
        super()._evaluate(x, out, *args, **kwargs)
        out["F"]+=_sample_noise(noise_type=self.noise_type, noise_std=self.noise_std, shape=out["F"].shape)
        
    def evaluate_noiseless(self, X: np.ndarray) -> np.ndarray:
        """Evaluates a population without noise."""
        out = {}
        super()._evaluate(X, out)
        return out["F"]

class NWFG7(WFG7, LoggingMixin):
    """This class represents the implementation of noisy WFG7."""
    
    def __init__(self, n_var=24, n_obj=3, k=None, l=None, noise_std =0.1, noise_type: str = "gaussian", **kwargs):
        self.noise_type = noise_type
        self.noise_std = noise_std
        super().__init__(n_var, n_obj, k, l, **kwargs)
        self.logger.info(f"Instance of {self.__class__.__name__} with noise of type {self.noise_type} and std {noise_std} created.")
    
    def _evaluate(self, x, out, *args, **kwargs):
        super()._evaluate(x, out, *args, **kwargs)
        out["F"]+=_sample_noise(noise_type=self.noise_type, noise_std=self.noise_std, shape=out["F"].shape)
        
    def evaluate_noiseless(self, X: np.ndarray) -> np.ndarray:
        """Evaluates a population without noise."""
        out = {}
        super()._evaluate(X, out)
        return out["F"]

class NWFG8(WFG8, LoggingMixin):
    """This class represents the implementation of noisy WFG8."""
    
    def __init__(self, n_var=24, n_obj=3, k=None, l=None, noise_std =0.1, noise_type: str = "gaussian", **kwargs):
        self.noise_type = noise_type
        self.noise_std = noise_std
        super().__init__(n_var, n_obj, k, l, **kwargs)
        self.logger.info(f"Instance of {self.__class__.__name__} with noise of type {self.noise_type} and std {noise_std} created.")
    
    def _evaluate(self, x, out, *args, **kwargs):
        super()._evaluate(x, out, *args, **kwargs)
        out["F"]+=_sample_noise(noise_type=self.noise_type, noise_std=self.noise_std, shape=out["F"].shape)
        
    def evaluate_noiseless(self, X: np.ndarray) -> np.ndarray:
        """Evaluates a population without noise."""
        out = {}
        super()._evaluate(X, out)
        return out["F"]

class NWFG9(WFG9, LoggingMixin):
    """This class represents the implementation of noisy WFG9."""
    
    def __init__(self, n_var=24, n_obj=3, k=None, l=None, noise_std =0.1, noise_type: str = "gaussian", **kwargs):
        self.noise_type = noise_type
        self.noise_std = noise_std
        super().__init__(n_var, n_obj, k, l, **kwargs)
        self.logger.info(f"Instance of {self.__class__.__name__} with noise of type {self.noise_type} and std {noise_std} created.")
    
    def _evaluate(self, x, out, *args, **kwargs):
        super()._evaluate(x, out, *args, **kwargs)
        out["F"]+=_sample_noise(noise_type=self.noise_type, noise_std=self.noise_std, shape=out["F"].shape)
        
    def evaluate_noiseless(self, X: np.ndarray) -> np.ndarray:
        """Evaluates a population without noise."""
        out = {}
        super()._evaluate(X, out)
        return out["F"]