"""Probabilistic Dominance NSGA-II."""
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
from scipy.stats import norm

def _prob_domination_matrix(means: np.ndarray, n_evals: np.ndarray,
                             noise_std: float, alpha: float) -> np.ndarray:
    """Return dom[i,j]=1 iff i α-probabilistically dominates j."""
    inv_n = 1.0 / n_evals                                           # (N,)
    se    = noise_std * np.sqrt(inv_n[:, None] + inv_n[None, :])    # (N, N)
    # diff[i, j, k] = μ_j_k - μ_i_k  (positive means i is better on obj k)
    diff  = means[None, :, :] - means[:, None, :]                   # (N, N, M)
    probs = norm.cdf(diff / (se[:, :, None] + 1e-300))              # (N, N, M)
    dom   = np.all(probs >= alpha, axis=2).astype(np.int8)
    np.fill_diagonal(dom, 0)
    return dom

def _fronts_from_dom_matrix(dom: np.ndarray, n_stop: int | None = None) -> list:
    """Non-dominated sorting from a precomputed domination matrix.
    Returns a list of fronts (each front is an int array of indices).
    """
    N          = dom.shape[0]
    dom_count  = dom.sum(axis=0).copy()                   # how many dominate i
    dom_set    = [list(np.where(dom[i])[0]) for i in range(N)]  # whom i dominates

    fronts, ranked = [], 0
    current = list(np.where(dom_count == 0)[0])

    while current:
        fronts.append(np.array(current, dtype=int))
        ranked += len(current)
        if n_stop is not None and ranked >= n_stop:
            break
        nxt = []
        for i in current:
            for j in dom_set[i]:
                dom_count[j] -= 1
                if dom_count[j] == 0:
                    nxt.append(j)
        current = nxt

    # Cycle fallback: any solution still with dom_count > 0
    unranked = np.where(dom_count > 0)[0]
    if len(unranked):
        fronts.append(unranked.astype(int))

    return fronts

class PD_Survival(RankAndCrowding):
    """Survival selection using probabilistic dominance."""

    def _do(self, problem, pop, *args, n_survive=None, algorithm=None, **kwargs):
        alpha     = getattr(algorithm, 'alpha', 0.75)
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

        dom    = _prob_domination_matrix(means, n_evals, noise_std, alpha)
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

class PD_NSGA2(NSGA2, LoggingMixin):
    """NSGA-II with probabilistic dominance.

    Rank-0 individuals are re-evaluated each generation to accumulate tighter
    mean estimates.

    Args:
        alpha:     Confidence level for dominance comparison (default 0.75).
        noise_std: Known noise standard deviation, passed from the problem.
    """

    def __init__(self, pop_size=100, alpha=0.75, noise_std=1.0, **kwargs):
        self.alpha     = alpha
        self.noise_std = noise_std
        super().__init__(pop_size=pop_size, survival=PD_Survival(), **kwargs)
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
