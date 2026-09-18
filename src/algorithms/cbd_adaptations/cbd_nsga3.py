"""Confidence-Based Dominance NSGA-III"""
from __future__ import annotations

# Make other project modules accessible
import sys
import os
sys.path.insert(1, os.path.join(os.path.dirname(__file__), ".."))

# Logger import
from config.logging_mixin import LoggingMixin

# Pymoo imports
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.algorithms.moo.nsga3 import ReferenceDirectionSurvival
from pymoo.algorithms.moo.nsga3 import associate_to_niches
from pymoo.algorithms.moo.nsga3 import calc_niche_count
from pymoo.algorithms.moo.nsga3 import niching
from pymoo.util.misc import intersect

# Package imports
import numpy as np

# Module imports
from algorithms.cb_adaptations.cb_nsga3 import _das_dennis_ref_dirs
from algorithms.pd_adaptations.pd_nsga2 import _fronts_from_dom_matrix
from algorithms.cbd_adaptations.cbd_nsga2 import _cbd_domination_matrix


class CBD_Survival_NSGA3(ReferenceDirectionSurvival):
    def _do(self, problem, pop, n_survive, D=None, random_state=None, **kwargs):
        algorithm = kwargs.get('algorithm')
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

        rank = np.empty(len(pop), dtype=int)
        for r, front in enumerate(fronts):
            rank[front] = r

        # Normalise using mean values so noise does not shift the hyperplane estimate
        self.norm.update(means, nds=fronts[0])
        ideal, nadir = self.norm.ideal_point, self.norm.nadir_point

        # Restrict to individuals that were assigned a front
        I = np.concatenate(fronts)
        pop_s, rank_s, means_s = pop[I], rank[I], means[I]

        # Remap front indices into the sub-population
        counter, fronts_s = 0, []
        for front in fronts:
            fronts_s.append(np.arange(counter, counter + len(front), dtype=int))
            counter += len(front)
        last_front_s = fronts_s[-1]

        # Reference-direction association on mean values
        niche_of_ind, dist_to_niche, dist_matrix = \
            associate_to_niches(means_s, self.ref_dirs, ideal, nadir)

        pop_s.set('rank', rank_s, 'niche', niche_of_ind, 'dist_to_niche', dist_to_niche)

        closest  = np.unique(dist_matrix[:, np.unique(niche_of_ind)].argmin(axis=0))
        self.opt = pop_s[intersect(fronts_s[0], closest)]
        if len(self.opt) == 0:
            self.opt = pop_s[fronts_s[0]]

        if len(pop_s) > n_survive:
            if len(fronts_s) == 1:
                n_rem, until_last, niche_count = (
                    n_survive, np.array([], dtype=int),
                    np.zeros(len(self.ref_dirs), dtype=int),
                )
            else:
                until_last  = np.concatenate(fronts_s[:-1])
                niche_count = calc_niche_count(len(self.ref_dirs), niche_of_ind[until_last])
                n_rem       = n_survive - len(until_last)

            S = niching(pop_s[last_front_s], n_rem, niche_count,
                        niche_of_ind[last_front_s], dist_to_niche[last_front_s],
                        random_state=random_state)
            sel       = last_front_s[np.array(S, dtype=int)]
            survivors = np.concatenate((until_last, sel)).astype(int)
            pop_s     = pop_s[survivors]

        return pop_s

class CBD_NSGA3(NSGA3, LoggingMixin):
    """NSGA-III with confidence-based dominance and reference-direction niching.

    Rank-0 survivors are re-evaluated each generation.

    Args:
        n_obj:     Number of objectives.
        z:         CI half-width in units of σ/√n (default 1.0).
        alpha:     Dominance degree threshold in (0, 1]
        noise_std: Known noise standard deviation.
    """

    def __init__(self, n_obj=3, z=1.0, alpha=1.0, noise_std=1.0, **kwargs):
        self.z         = z
        self.alpha     = alpha
        self.noise_std = noise_std
        ref_dirs = _das_dennis_ref_dirs(n_obj)
        super().__init__(ref_dirs=ref_dirs, survival=CBD_Survival_NSGA3(ref_dirs), **kwargs)
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
