"""Confidence-Based Dominance NSGA-II"""
from __future__ import annotations

# Make other project modules accessible
import sys
import os
sys.path.insert(1, os.path.join(os.path.dirname(__file__), ".."))

# Logger import
from config.logging_mixin import LoggingMixin

# Pymoo imports
from pymoo.algorithms.moo.nsga2 import NSGA2, RankAndCrowding

# Package imports
import numpy as np

# Misc imports
from algorithms.pd_adaptations.pd_nsga2 import _fronts_from_dom_matrix


def _cbd_domination_matrix(means: np.ndarray, n_evals: np.ndarray,
                            noise_std: float, z: float, alpha: float = 1.0) -> np.ndarray:
    """Return dom[i,j]=1 iff i confidence-dominates j with degree >= alpha."""
    se  = noise_std / np.sqrt(n_evals)                      # (N,)
    eps = z * se                                             # (N,)  ε_p

    mean_diff = means[None, :, :] - means[:, None, :]        # (N,N,M): μ_j − μ_i
    eps_diff  = (eps[None, :] - eps[:, None])[:, :, None]     # (N,N,1): ε_j − ε_i
    denom     = 2.0 * eps[:, None, None]                      # (N,1,1): 2·ε_i

    deg = np.clip((mean_diff - eps_diff) / (denom + 1e-300), 0.0, 1.0)  # (N,N,M)

    dom = np.all(deg >= alpha, axis=2).astype(np.int8)
    np.fill_diagonal(dom, 0)
    return dom


class CBD_Survival(RankAndCrowding):
    """Survival selection using confidence-interval-based dominance."""

    def _do(self, problem, pop, *args, n_survive=None, algorithm=None, **kwargs):
        z         = getattr(algorithm, 'z', 1.0)
        alpha     = getattr(algorithm, 'alpha', 1.0)
        noise_std = getattr(algorithm, 'noise_std', 1.0)

        means = np.array([
            ind.f_sum / ind.n_evals
            if (hasattr(ind, 'f_sum') and ind.n_evals > 0) else ind.F
            for ind in pop
        ])
        n_evals = np.array([
            ind.n_evals if (hasattr(ind, 'n_evals') and ind.n_evals > 0) else 1
            for ind in pop
        ], dtype=float)

        dom    = _cbd_domination_matrix(means, n_evals, noise_std, z, alpha)
        fronts = _fronts_from_dom_matrix(dom, n_stop=n_survive)

        for rank, front in enumerate(fronts):
            pop[front].set("rank", rank)

        for front in fronts:
            cd = self.crowding_func.do(means[front])
            pop[front].set("crowding", cd)

        survivors = []
        for front in fronts:
            if n_survive is None or len(survivors) + len(front) <= n_survive:
                survivors.extend(front.tolist())
            else:
                needed   = n_survive - len(survivors)
                crowding = pop[front].get("crowding")
                survivors.extend(front[np.argsort(-crowding)[:needed]].tolist())
                break

        return pop[survivors]


class CBD_NSGA2(NSGA2, LoggingMixin):
    """NSGA-II with confidence-based dominance (Boonma & Suzuki 2009 / Karshenas et al. 2015).

    Rank-0 individuals are re-evaluated each generation to accumulate tighter
    confidence intervals.

    Args:
        z:         Half-width of the confidence interval in units of σ/√n.
                   z=1.0  → ~68% CI,  z=1.645 → 90% CI,  z=1.96 → 95% CI.
        alpha:     Dominance degree threshold in (0, 1]
        noise_std: Known noise standard deviation.
    """

    def __init__(self, pop_size=100, z=1.0, alpha=1.0, noise_std=1.0, **kwargs):
        self.z         = z
        self.alpha     = alpha
        self.noise_std = noise_std
        super().__init__(pop_size=pop_size, survival=CBD_Survival(), **kwargs)
        self.logger.info(f"Instance of {self.__class__.__name__} created.")

    def _initialize_advance(self, infills=None, **kwargs):
        self.logger.debug("Initializing advance ...")
        if infills is not None:
            for ind in infills:
                if not hasattr(ind, 'n_evals'):
                    ind.n_evals = 0
                    ind.f_sum   = np.zeros_like(ind.F)
                ind.n_evals += 1
                ind.f_sum   += ind.F
        super()._initialize_advance(infills=infills, **kwargs)

    def _advance(self, infills=None, **kwargs):
        self.logger.debug("Advance ...")
        if infills is not None:
            for ind in infills:
                if not hasattr(ind, 'n_evals'):
                    ind.n_evals = 0
                    ind.f_sum   = np.zeros_like(ind.F)
                ind.n_evals += 1
                ind.f_sum   += ind.F
        super()._advance(infills=infills, **kwargs)

        self.logger.debug("Re-evaluating rank 0 individuals ...")
        rank0 = self.pop[self.pop.get("rank") == 0]
        self.evaluator.eval(self.problem, rank0, algorithm=self, skip_already_evaluated=False)
        for ind in rank0:
            ind.f_sum   += ind.F
            ind.n_evals += 1
