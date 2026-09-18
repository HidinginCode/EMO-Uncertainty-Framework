from __future__ import annotations

import numpy as np
from pymoo.algorithms.moo.age import AGEMOEA, AGEMOEASurvival

class CB_AGEMOEASurvival(AGEMOEASurvival):
    def _do(self, problem, pop, *args, algorithm=None, **kwargs):
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

        return super()._do(problem, pop, *args, algorithm=algorithm, **kwargs)


class cb_agemoea(AGEMOEA):
    """CB implementation of AGE-MOEA.

    Applies LCB to F before AGEMOEA-Survival's geometry-adaptive ranking.
    Re-evaluates rank-0 survivors each generation to reduce uncertainty on the
    current Pareto front approximation.
    """

    def __init__(self, pop_size=100, c=-0.2, **kwargs):
        self.c = c
        self.T = 0
        super().__init__(pop_size=pop_size, **kwargs)
        # AGEMOEASurvival is hardcoded in AGEMOEA.__init__; replace it here
        self.survival = CB_AGEMOEASurvival()

    def _initialize_advance(self, infills=None, **kwargs):
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
        if infills is not None:
            for ind in infills:
                if not hasattr(ind, 'n_evals'):
                    ind.n_evals = 0
                    ind.f_sum = np.zeros_like(ind.F)
                ind.n_evals += 1
                ind.f_sum += ind.F
                self.T += 1
        super()._advance(infills=infills, **kwargs)

        rank0 = self.pop[self.pop.get("rank") == 0]
        self.evaluator.eval(self.problem, rank0, algorithm=self, skip_already_evaluated=False)
        for ind in rank0:
            ind.f_sum += ind.F
            ind.n_evals += 1
            self.T += 1
