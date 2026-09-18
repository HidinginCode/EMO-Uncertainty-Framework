"""This module implements CB-NSGA-III"""
from __future__ import annotations

# Make other project modules accessible
import sys
import os
sys.path.insert(1, os.path.join(os.path.dirname(__file__), ".."))

# Logger import
from config.logging_mixin import LoggingMixin

# Python imports
import math

# Pymoo imports
from pymoo.algorithms.moo.nsga3 import NSGA3, ReferenceDirectionSurvival
from pymoo.util.ref_dirs import get_reference_directions

# Package imports
import numpy as np

def _das_dennis_ref_dirs(n_obj: int, target_points: int = 99):
    """Return das-dennis reference directions with at least target_points vectors."""
    if n_obj == 2:
        return get_reference_directions("das-dennis", n_obj, n_points=target_points)
    p = 1
    while math.comb(p + n_obj - 1, n_obj - 1) < target_points:
        p += 1
    return get_reference_directions("das-dennis", n_obj, n_partitions=p)

class CB_Survival(ReferenceDirectionSurvival):
    def __init__(self, ref_dirs):
        super().__init__(ref_dirs)

    def do(self, problem, pop, *args, algorithm=None, **kwargs):
        c = algorithm.c
        if algorithm.pop is not None:
            rank0_mask = algorithm.pop.get("rank") == 0
            T_rank0 = sum(ind.n_evals for ind in algorithm.pop[rank0_mask]) if rank0_mask.any() else algorithm.T
        else:
            T_rank0 = algorithm.T

        for ind in pop:
            if hasattr(ind, 'n_evals') and ind.n_evals > 0:
                mean = ind.f_sum / ind.n_evals
                ind.F = mean + c * np.sqrt(2 * np.log(T_rank0) / ind.n_evals)

        return super().do(problem, pop, *args, algorithm=algorithm, **kwargs)

class cb_nsga3(NSGA3, LoggingMixin):
    """CB implementation of NSGA-III"""

    def __init__(self, n_obj=2, c=0.2, **kwargs):
        self.T = 0
        self.c = c
        ref_dirs = _das_dennis_ref_dirs(n_obj)
        super().__init__(ref_dirs=ref_dirs, survival=CB_Survival(ref_dirs), **kwargs)
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

        self.logger.debug("Re-evaluating rank 0 indiviuals ...")
        rank0 = self.pop[self.pop.get("rank") == 0]
        self.evaluator.eval(self.problem, rank0, algorithm=self, skip_already_evaluated=False)
        for ind in rank0:
            ind.f_sum += ind.F
            ind.n_evals += 1
            self.T += 1
