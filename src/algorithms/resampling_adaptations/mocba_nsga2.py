"""This module contains the implementation of MOCBA NSGA-II."""

# Make other project modules accessible
import sys
import os

sys.path.insert(1, os.path.join(os.path.dirname(__file__), ".."))

# Logger import
from config.logging_mixin import LoggingMixin

# Package import
import numpy as np
from scipy.stats import norm

# Pymoo imports
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.individual import Individual
from pymoo.core.population import Population
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

# CB_Resampling import
from src.resampling.cb_resampling import CB_Resampling

# Python imports
import math

class MOCBA_NSGA2(NSGA2, LoggingMixin):
    """This class implements NSGA-II using the MOCBA resampling scheme."""

    def __init__(self, pop_size:int = 100, init_replications:int = 5, resampling_budget: int = 50):
        """Init method for MOCBA-NSGA2.

        Args:
            pop_size(int, optional): Population size of EA. Defaults to 100.
            init_replications(int, optional): Initial ammount individual is sampled. Defaults to 5.
            resampling_budget(int, optional): Budget used for resampling per gen. Defaults to 50.
        """

        self.init_replications = init_replications
        self.resampling_budget = resampling_budget
        self.logger.info(f"Created instance of {self.__class__.__name__}. Inital replications set to {init_replications}.")
        super().__init__(pop_size)

    def _initialize_advance(self, infills=None, **kwargs):
        # Need to resample initial individuals
        # First prepare them using CBR Resampling
        if infills is not None:
            CB_Resampling._prepare_infills(infills = infills) # Adds most necessary attributes but missing variance.
            for ind in infills:
                ind.variance = np.zeros_like(ind.mean)
                ind.M2 = np.zeros_like(ind.mean)   # accumulator for sum of squared deviations

                # Sample rest of initial replications
                self._resample(ind = ind, res_amount = self.init_replications-1)


        return super()._initialize_advance(infills, **kwargs)

    def _advance(self, infills=None, **kwargs):

        # Prepare infills again
        if infills is not None:
            CB_Resampling._prepare_infills(infills = infills) # Adds most necessary attributes but missing variance.
            for ind in infills:
                ind.variance = np.zeros_like(ind.mean)
                ind.M2 = np.zeros_like(ind.mean)

                # Sample rest of initial replications
                self._resample(ind = ind, res_amount = self.init_replications-1)

        # Get dominated and non_dom individuals
        # Needs infills and normal pop as one population object
        ind_collection =  Population.merge(self.pop, infills)
        nds = NonDominatedSorting().do(F = ind_collection.get("F"))

        # Nds retuns indices, so we have to extract from merged population again
        non_dom = [ind_collection[x] for x in nds[0]]
        dom = [ind_collection[x] for y in nds[1:] for x in y]

        # Compute re-allocation weight for every dominated individual
        weights = {}
        for ind in dom:
            # Identify most probable misclassification
            closest_non_dom, misclass_prob, q_star = self._closest_non_dom_by_risk(ind, non_dom)

            weights[ind] = misclass_prob * np.sqrt(ind.variance[q_star]) / max(ind.n_evals, 1)  # pyright: ignore[reportAttributeAccessIssue]

        total_weight = np.sum([weights[ind] for ind in dom])

        if total_weight == 0:
            return super()._advance(infills, **kwargs)

        resampling_alloc = {}
        for ind in dom:
            resampling_alloc[ind] = (weights[ind] / total_weight) * self.resampling_budget

        # Round the re-allocations so we hit the budget perfectly
        rounded_allocs = self.largest_remainder_round(resampling_alloc, self.resampling_budget)

        for entry in rounded_allocs:
            self._resample(ind = entry, res_amount = rounded_allocs[entry])

        return super()._advance(infills, **kwargs)

    def _resample(self, ind, res_amount: int) -> None:
        """Method that resamples individual specified amount of times.

        Args:
            ind (Individual): Individual that is to be resampled.
            res_amount (int): Amount of times ind is resampled.
            """

        # Safety checks
        assert hasattr(ind, "n_evals"), f"Mocba resampling individual did not have n_evals attribute."
        assert hasattr(ind, "f_sum"), f"Mocba resampling individual did not have f_sum attribute."
        assert hasattr(ind, "mean"), f"Mocba resampling individual did not have mean attribute."

        for _ in range(res_amount):
            # Re-evaluate
            self.evaluator.eval(problem = self.problem, pop = ind, skip_already_evaluated=False)

            # Re-calculate mean and variance
            ind.n_evals += 1
            delta = ind.F - ind.mean
            ind.mean = ind.mean + delta / ind.n_evals
            delta2 = ind.F - ind.mean
            ind.M2 = ind.M2 + delta * delta2
            ind.variance = ind.M2 / (ind.n_evals - 1) if ind.n_evals > 1 else np.zeros_like(ind.mean)
            ind.F = ind.mean

    def _standardized_objective_gap(self, dom_ind, non_dom_ind) -> np.ndarray:
        """Calculates standardized distance in objectives between supplied individuals.

        Args:
            dom_ind(Individual): Dominated individual used for the comparison.
            non_dom_ind(Individual): Non-dominated individual used for the comparison.

        Returns:
            np.ndarray: Normalized objective difference vector."""

        # Safety check
        assert np.shape(dom_ind.mean) == np.shape(non_dom_ind.mean), f"Shapes of individual means for objective gap calculation did not match. Non dom: {np.shape(non_dom_ind.mean)}, Dom: {np.shape(dom_ind.mean)}."
        assert np.shape(dom_ind.variance) == np.shape(non_dom_ind.variance), f"Shapes of individual variance for objective gap calculation did not match. Non dom: {np.shape(non_dom_ind.variance)}, Dom: {np.shape(dom_ind.variance)}."

        normalized_diff = []
        # Calculate in the loop for every objective
        for i, _ in enumerate(dom_ind.mean):
            normalized_diff.append((dom_ind.mean[i] - non_dom_ind.mean[i])/np.sqrt(((dom_ind.variance[i]/dom_ind.n_evals)+(non_dom_ind.variance[i]/non_dom_ind.n_evals))))

        return np.asarray(normalized_diff)

    def _closest_non_dom_by_risk(self, dom_ind, non_dom):
        """Select the non-dominated competitor that gives the HIGHEST misclassification probability for dom_ind.

        Args:
            dom_ind: Dominated individudal for comparison.
            non_dom: Non dominated individuals.
        """
        best_prob = -1.0
        best_ind = None
        best_q_star = None

        for non_dom_ind in non_dom:
            gaps = self._standardized_objective_gap(dom_ind=dom_ind, non_dom_ind=non_dom_ind)
            q_star = np.argmin(gaps)
            prob = norm.cdf(-gaps[q_star])

            if prob > best_prob:
                best_prob = prob
                best_ind = non_dom_ind
                best_q_star = q_star

        return best_ind, best_prob, best_q_star

    def largest_remainder_round(self, alloc_dict, target_total):
        """Round raw allocations to ints that sum exactly to target_total.

        Args:
            alloc_dict: Raw (fractional) allocation.
            target_total: Total the rounded allocations must sum to.

        Returns:
            Dict integer allocations summing to target_total."""

        inds = list(alloc_dict)
        floored = {ind: math.floor(alloc_dict[ind]) for ind in inds}

        # Units still to distribute
        difference = int(round(target_total - sum(floored.values())))
        assert abs(difference) <= len(inds), f"Rounding gap {difference} exceeds number of allocations {len(inds)}."

        # Individuals ordered by fractional remainder, largest first
        order = sorted(inds, key=lambda ind: alloc_dict[ind] - math.floor(alloc_dict[ind]), reverse=True)

        if difference > 0:
            for ind in order[:difference]:
                floored[ind] += 1
        elif difference < 0:
            for ind in order[difference:]:
                floored[ind] -= 1

        return floored
