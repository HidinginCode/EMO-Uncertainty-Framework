"""This module contains a static resampling version of NSGA-II."""

# Make other project modules accessible
import sys
import os

sys.path.insert(1, os.path.join(os.path.dirname(__file__), ".."))

# Logger import
from config.logging_mixin import LoggingMixin

# Package import
import numpy as np

# Pymoo imports
from pymoo.algorithms.moo.nsga2 import NSGA2

# CB_Resampling import
from src.resampling.cb_resampling import CB_Resampling

class SR_NSGA2(NSGA2, LoggingMixin):
    """This class implements NSGA-II for noisy problems with static resampling."""

    def __init__(self, pop_size=100, k = 10):
        """Init function for static resampling NSGA-II.

        Args:
            pop_size (int, optional): Population size used by NSGA-II. Defaults to 100.
            k (int, optional): Number of re_evaluations per individual. Defaults to 10."""

        self.k = k
        super().__init__(pop_size)
        self.logger.info(f"Instance of {self.__class__.__name__} created. Chosen k = {self.k}.")

    def _initialize_advance(self, infills=None, **kwargs) -> None:
        # CB resampling can be used here to prepare infills
        # Things to keep in mind: It just sets up the variables for infills
        # The new individuals will still need to be sampled with k-1 times.

        # Re-use CBR resampling preparation of infills
        CB_Resampling._prepare_infills(infills = infills)

        # Resampling step (k-1) times
        self.logger.debug("Starting resampling of infills.")
        self._static_resampling(infills = infills)

        return super()._initialize_advance(infills, **kwargs)

    def _advance(self, infills=None, **kwargs) -> None:

        # Prepare new infills with necessary arguments
        CB_Resampling._prepare_infills(infills = infills)

        # Resample
        self._static_resampling(infills = infills)

        return super()._advance(infills, **kwargs)

    def _static_resampling(self, infills) -> None:
        """Method that implements static resampling for individuals."""
        if infills is not None:
            for ind in infills:
                while ind.n_evals < self.k:
                    self.evaluator.eval(problem = self.problem, pop = ind, algorithm = self, skip_already_evaluated=False)
                    ind.old_mean = ind.mean
                    ind.n_evals += 1
                    ind.f_sum += ind.F
                    ind.mean = ind.f_sum / ind.n_evals
                    ind.F = ind.mean
