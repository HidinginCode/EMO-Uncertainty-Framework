"""This module contains the confidence bound implementation of SMS-EMOA"""
from __future__ import annotations

# Make other project modules accessible
import sys
import os
sys.path.insert(1, os.path.join(os.path.dirname(__file__), ".."))

# Logger import
from config.logging_mixin import LoggingMixin

# Pymoo import
from pymoo.algorithms.moo.sms import SMSEMOA, LeastHypervolumeContributionSurvival

# Package imports
import numpy as np

class CB_Survival(LeastHypervolumeContributionSurvival):
    """Replaces F with LCB values before SMS-EMOA's HV-contribution elimination."""

    # Implements UCB function value calc and updates the mean for rank-0 individuals
    def _do(self, problem, pop, *args, algorithm=None, **kwargs):
        c = algorithm.c
        if algorithm.pop is not None:
            rank0_mask = algorithm.pop.get("rank") == 0
            if rank0_mask.any():
                T_rank0 = sum(ind.n_evals for ind in algorithm.pop[rank0_mask])
            else:
                T_rank0 = sum(ind.n_evals for ind in pop if hasattr(ind, 'n_evals') and ind.n_evals > 0) or 1
        else:
            T_rank0 = sum(ind.n_evals for ind in pop if hasattr(ind, 'n_evals') and ind.n_evals > 0) or 1

        for ind in pop:
            if hasattr(ind, 'n_evals') and ind.n_evals > 0:
                mean = ind.f_sum / ind.n_evals
                ind.F = mean + c * np.sqrt(2 * np.log(T_rank0) / ind.n_evals)
        return super()._do(problem, pop, *args, algorithm=algorithm, **kwargs)


class CB_SMSEMOA(SMSEMOA, LoggingMixin):
    """Confidence-bound SMS-EMOA."""

    def __init__(self, pop_size=100, c=-0.1, **kwargs):
        self.c = c
        self.T = 0
        super().__init__(pop_size=pop_size, survival=CB_Survival(), **kwargs)
        self.logger.info(f"Instance of {self.__class__.__name__} was created.")

    # Override initialize_advance and advance to give necessary attributes to all individuals
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

        # Re-evaluate rank-0 survivors
        self.logger.debug("Re-evaluating rank 0 individuals ...")
        rank0 = self.pop[self.pop.get("rank") == 0]
        self.evaluator.eval(self.problem, rank0, algorithm=self, skip_already_evaluated=False)
        for ind in rank0:
            ind.f_sum += ind.F
            ind.n_evals += 1
            self.T += 1
