"""This module contains the imlementation for CB-SPEA2."""
from __future__ import annotations

# Make other project modules accessible
import sys
import os
sys.path.insert(1, os.path.join(os.path.dirname(__file__), ".."))

# Logger import
from config.logging_mixin import LoggingMixin

# Pymoo imports
from pymoo.algorithms.moo.spea2 import SPEA2
from pymoo.algorithms.moo.spea2 import SPEA2Survival

# Package imports
import numpy as np

class CB_SPEA2(SPEA2, LoggingMixin):
    """This class implements confidence bound SPEA-2"""
    
    def __init__(self, pop_size=100, c=0.1, **kwargs):
        super().__init__(pop_size=pop_size, survival=CB_Survival(), **kwargs)
        self.c = c
        self.T = 0
        self.logger.info(f"Instance of {self.__class__.__name__} created.")
    
    def _initialize_advance(self, infills=None, **kwargs):
        self.logger.debug("Initializing advance ...")
        if infills is not None:
            for ind in infills:
                if not hasattr(ind, 'n_evals'):
                    ind.n_evals = 0
                    ind.f_sum = np.zeros_like(ind.F)
                ind.n_evals += 1
                ind.f_sum += ind.F
                self.T += 1
        super()._initialize_advance(infills=infills, **kwargs)

    def _advance(self, infills=None, **kwargs):
        self.logger.debug("Advance ...")
        if infills is not None:
            for ind in infills:
                if not hasattr(ind, 'n_evals'):
                    ind.n_evals = 0
                    ind.f_sum = np.zeros_like(ind.F)
                ind.n_evals += 1
                ind.f_sum += ind.F
                self.T += 1
        super()._advance(infills=infills, **kwargs)

        self.logger.debug("Re-evaluating rank 0 individuals ...")
        rank0 = self.pop[self.pop.get("SPEA_R") == 0]
        self.evaluator.eval(self.problem, rank0, algorithm=self, skip_already_evaluated=False)
        for ind in rank0:
            ind.f_sum += ind.F
            ind.n_evals += 1
            self.T += 1

class CB_Survival(SPEA2Survival):
    def do(self, problem, pop, *args, algorithm=None, **kwargs):
        c = algorithm.c
        if algorithm.pop is not None:
            rank0_mask = algorithm.pop.get("SPEA_R") == 0 # Specialty for SPEA -> Non dominated R value and ignore density estimation
            T_rank0 = sum(ind.n_evals for ind in algorithm.pop[rank0_mask]) if rank0_mask.any() else algorithm.T
        else:
            T_rank0 = algorithm.T

        for ind in pop:
            if hasattr(ind, 'n_evals') and ind.n_evals > 0:
                mean = ind.f_sum / ind.n_evals
                ind.F = mean + c * np.sqrt(2 * np.log(T_rank0) / ind.n_evals)

        return super().do(problem, pop, *args, algorithm=algorithm, **kwargs)