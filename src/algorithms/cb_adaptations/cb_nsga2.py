"""This file contains the implementation of a NSGA-II algorithm that uses UCB1 to deal with noise."""
from __future__ import annotations

# Make other project modules accessible
import sys
import os
sys.path.insert(1, os.path.join(os.path.dirname(__file__), ".."))

# Logger import
from config.logging_mixin import LoggingMixin

# Pymoo imports
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.moo.nsga2 import RankAndCrowding

# Package imports
import numpy as np

class CB_NSGA2(NSGA2, LoggingMixin):
    """Implementation of CB NSGA-II based on normal NSGA-II"""
    
    def __init__(self, pop_size=100, c=-0.2, **kwargs):
        # Initialize rank-0 eval counter (T) and the exploration/exploitation constant (c)
        self.T = 0
        self.c = c
        super().__init__(pop_size=pop_size, survival=CB_Survival(), **kwargs)
        self.logger.info(f"Instance of {self.__class__.__name__} created.")


    def _initialize_advance(self, infills=None, **kwargs):
        # Initialize advance happens after first individuals have been generated and evaluated
        # Here we set up all individuals with its own eval counter (n_evals) and averaged fitness (f_sum)
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
        # Advance handles what happens after evaluation of new individuals and sets them up how initialize advance does
        # Furthermore, it re-evaluates the rank-0 individuals and updates the counters
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
    
class CB_Survival(RankAndCrowding):
    """Class that implements ranking using the UCB algorithm."""
    def do(self, problem, pop, *args, algorithm=None, **kwargs):
        #T = algorithm.T -> This is accumulating over all reevaluations. Using this high value will lead to a major confidence boost for individuals in later generations
        c = algorithm.c
        if algorithm.pop is not None:
            rank0_mask = algorithm.pop.get("rank") == 0  # all False if ranks not yet set (None != 0)
            T_rank0 = sum(ind.n_evals for ind in algorithm.pop[rank0_mask]) if rank0_mask.any() else algorithm.T
        else:
            T_rank0 = algorithm.T

        for ind in pop:
            if hasattr(ind, 'n_evals') and ind.n_evals > 0:
                mean = ind.f_sum / ind.n_evals
                ind.F = mean + c * np.sqrt(2 * np.log(T_rank0) / ind.n_evals)
        return super().do(problem, pop, *args, algorithm=algorithm, **kwargs)