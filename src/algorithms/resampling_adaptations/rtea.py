"""This module contains the implementation of Fieldsend's Rolling Tide Evoluionary algorithm."""

# Make other project modules accessible
import sys
import os

from pymoo.operators.sampling.rnd import FloatRandomSampling
sys.path.insert(1, os.path.join(os.path.dirname(__file__), ".."))

# Logger import
from config.logging_mixin import LoggingMixin

# Package imports
import numpy as np

# Pymoo imports
from pymoo.core.algorithm import Algorithm
from pymoo.core.population import Population
from pymoo.core.individual import Individual
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting as NDS
from pymoo.util.dominator import Dominator
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.gauss import GaussianMutation
from pymoo.operators.repair.bounds_repair import repair_random_init
from pymoo.core.variable import get

# Python imports
import typing

class SeededGaussianMutation(GaussianMutation):
    """GaussianMutation whose out of bounds repair uses the supplied random state.

    Pymoos version calls its repair without a random state, so that part draws from an unseeded generator and makes runs irreproducible."""

    def _do(self, problem, X, random_state = None, **kwargs):
        X = X.astype(float)

        sigma = get(self.sigma, size = len(X))
        prob_var = self.get_prob_var(problem, size = len(X))

        # Same procedure as pymoos mut_gauss, every chosen variable gets additive noise with a width relative to its range
        mut = random_state.random(X.shape) < prob_var[:, None]
        width = sigma[:, None] * (problem.xu - problem.xl)[None, :]
        Xp = X.copy()
        Xp[mut] = random_state.normal(X[mut], width[mut])

        return repair_random_init(Xp, X, problem.xl, problem.xu, random_state = random_state)

# Main thing for this implementation is that this algortihm does not fit existing patterns.
# Consequently I will build all the steps myself using pymoo but maybe sidestepping some mechanisms.
class RTEA(LoggingMixin, Algorithm):
    """This class implements Fieldsend's Rolling Tide Evolutionary Algorithm."""

    def __init__(self, pop_size: int = 100, sampling = FloatRandomSampling(), cross_prob: float = 0.8, archive_resamples: int = 1, archive_refinement: float = 0.05) -> None:
        """Init method for RTEA.

        Args:
            pop_size (int, optinal): Number of individuals in a given population. Defaults to 100.
            cross_prob (float, optional): Crossover probability. Defaults to 0.8.
            archive_resamples (int, optional): Number of resamples spent on the archive per generation. Defaults to 1.
            archive_refinement (float, optional): Portion of the run that is solely used to refine archive estimations. Defaults to 0.05 or 5%."""

        # Safety checks
        assert pop_size > 2 and isinstance(pop_size, int), f"Population size musst be bigger than 2 and an integer. Supplied pop_size: {pop_size}."
        assert 0 <= cross_prob <= 1 and isinstance(cross_prob, float), f"Crossover probability must be within [0, 1] and be a float. Supplied cross_prob {cross_prob}."
        assert archive_resamples >= 0 and isinstance(archive_resamples, int), f"Archive resamples must be a positive integer. Supplied archive_resamples: {archive_resamples}."
        assert archive_refinement >= 0 and isinstance(archive_refinement, float), f"Archive refinement portion of the run must be a positive float value. Supplied archive_refinement: {archive_refinement}."

        super().__init__()

        self.sampling = sampling
        self.pop_size = pop_size

        self._cross_prob = cross_prob
        self._archive_resamples = archive_resamples
        self._archive_refinement = archive_refinement

        # Create archive
        self._archive: Population = Population()

        # Reverse lookup from a dominator to the pop members that currently track it, saves scanning the whole pop on every resample
        self._dependents: dict[Individual, list[Individual]] = {}

        self.logger.info(f"Instance of {self.__class__.__name__} has been created.")
        self.logger.debug(f"Pop size: {self.pop_size}\nCrossover probabilitiy: {self._cross_prob}\nArchive resamples: {self._archive_resamples}\nArchive refinement: {self._archive_refinement}")

    def _initialize_infill(self) -> Population:
        # Runs once during the first next() call
        # Is invoked inside infill()
        # Produces initial batch of candidate solutions to be evaluated
        return self.sampling.do(self.problem, self.pop_size, random_state=self.random_state)

    def _infill(self) -> Population:
        # RTEA requires only a single offspring to be created using simulated binary crossover
        # Probability for this is given on algorithm level as self._cross_prob
        # Individual has to wrapped into a Population object (pymoo demands that)

        # Safety check
        assert self.random_state is not None, f"Tried to utilize random state during _infill but it was None."

        # During the refinement phase no new candidates are generated, only existing archive members get resampled
        if not self._in_search_phase():
            return None  # pyright: ignore[reportReturnType]

        # Sample two archive members at random, only allowing a duplicate pair if the archive has a single member
        if len(self._archive) > 1:
            chosen_indices = self.random_state.choice(len(self._archive), size = 2, replace = False)
        else:
            chosen_indices = np.array([0, 0])

        if self.random_state.random() < self._cross_prob:
            # eta = 20, prob = 1.0 and n_offsprings = 1 match Fieldsend's reference implementation
            new_ind = SBX(eta = 20, prob = 1.0, n_offsprings = 1).do(problem = self.problem, pop = self._archive, parents = chosen_indices.reshape(1, -1), random_state = self.random_state)
        else: # If we do not create a new child we copy the parent's decision variables into a fresh individual (a deepcopy would keep the parent's evaluated flag and pymoo would skip evaluating it) and mutate it
            new_ind = Population.new("X", self._archive[[chosen_indices[0]]].get("X"))

        # Mutate the new_ind
        new_ind = SeededGaussianMutation(sigma = 0.2).do(problem = self.problem, pop = new_ind, random_state = self.random_state)

        return new_ind

    def _initialize_advance(self, infills=None, **kwargs) -> None:
        # Populate the archive since first eval has been executed by now
        self._populate_archive(population=infills) # After this population is split into _archive with non_dom members and pop with dom members
        self._track_dominators()

        return super()._initialize_advance(infills, **kwargs)

    def _advance(self, infills=None, **kwargs):
        # Individual that comes out of infill is processed here, the archive placement logic lives in _update_front
        if infills is not None:
            infill = infills[0]
            infill.set("n_evals", 1)
            self._update_front(infill)

        # Resampling refines the archive's accuracy every iteration, regardless of search or refinement phase
        self._resample_archive()

        return super()._advance(infills, **kwargs)

    def _update_front(self, individual, in_pop: bool = False):
        """Method that checks an individual against the current archive and places it in the archive or the search population.

        Args:
            individual (Individual): Individual to check against the archive.
            in_pop (bool, optional): Whether the individual already is part of the search population (recheck of a tracked member). Defaults to False."""

        # If individual is not domianted by any archive member it is added
        # All archive members that are dominated get returned to pop with the individual as their tracked dominator
        # If it is dominated by the archive then one archive member is chosen to be its domiantor and it is added to the search population

        removed_archive_indices = []
        # Dominance check against archive
        for i, a_member in enumerate(self._archive):
            dom_realation = Dominator().get_relation(a = a_member.F , b = individual.F)

            match dom_realation:
                case 1: # Archive member dominates individual
                    self._set_dominator(individual, a_member)
                    if not in_pop:
                        self.pop = Population.merge(a = self.pop, b = individual)
                    break
                case -1: # Individual dominates archive member
                    self._set_dominator(a_member, individual)
                    removed_archive_indices.append(i)
        else: # Loop completed without breaking, so individual was not dominated by the archive
            if in_pop: # Promoted out of the search population
                self.pop = self.pop[np.array([p_member is not individual for p_member in self.pop], dtype = bool)]  # pyright: ignore[reportAttributeAccessIssue]
            self._archive = Population.merge(a = self._archive, b = individual)

        # Take archive member indices and transfer them to pop
        transfer_pop = self._archive[removed_archive_indices]
        self.pop = Population.merge(a = self.pop, b = transfer_pop)
        self._archive = self._archive[np.setdiff1d(np.arange(len(self._archive)), removed_archive_indices)]  # pyright: ignore[reportAttributeAccessIssue]

    def _set_dominator(self, individual, dominator):
        """Method that records the tracked dominator of an individual, both on the individual itself and in the reverse lookup.

        Args:
            individual (Individual): Individual that is dominated.
            dominator (Individual): Individual that dominates it."""

        individual.set("dominator", dominator)
        self._dependents.setdefault(dominator, []).append(individual)

    def _resample_archive(self):
        """Method that reevaluates the archive members with the fewest samples to refine their objective estimates."""

        self.logger.debug("Resampling archive.")

        for _ in range(min(self._archive_resamples, len(self._archive))):
            n_evals = self._archive.get("n_evals")
            chosen_index = np.argmin(n_evals)
            chosen = self._archive[chosen_index]

            assert self.pop is not None, f"Population was None while trying to resample the archive."
            # Pop members currently tracking chosen as their dominator need to be rechecked once its estimate changes
            rechecked = self._dependents.pop(chosen, [])

            # Remove chosen from the archive while its estimate is refined
            self._archive = self._archive[np.setdiff1d(np.arange(len(self._archive)), [chosen_index])]  # pyright: ignore[reportAttributeAccessIssue]

            # Reevaluate chosen once and fold the new sample into its running mean estimate
            resample = Population.new("X", chosen.X.reshape(1, -1))
            self.evaluator.eval(self.problem, resample, algorithm = self)

            new_n_evals = chosen.get("n_evals") + 1
            chosen.set("F", chosen.F + (resample[0].F - chosen.F) / new_n_evals)
            chosen.set("n_evals", new_n_evals)

            self._update_front(chosen)

            for p_member in rechecked:
                self._update_front(p_member, in_pop = True)

    def _in_search_phase(self) -> bool:
        """Method that checks whether the run is still in its search phase, as opposed to the pure archive refinement phase."""

        # Safety check
        assert self.termination is not None, f"Termination criterion was none in _in_search_phase method."
        assert hasattr(self.termination, "n_max_evals"), f"Termination had no attribute n_max_evals"

        return self.evaluator.n_eval < (1 - self._archive_refinement) * self.termination.n_max_evals  # pyright: ignore[reportAttributeAccessIssue]

    def _set_optimum(self) -> None:
        # The archive is the estimated Pareto set, pymoos default would rescan the entire (ever growing) pop after every iteration
        self.opt = self._archive

    def _populate_archive(self, population: Population = None):
        """Method that populates the archive using a given population.

        Args:
            population(Population): Population used to populate the archive. Defaults to None."""

        self.logger.debug("Populating the archive.")

        if population is not None:
            population.set("n_evals", 1)
            non_dom = NDS().do(F = population.get("F"), only_non_dominated_front = True)
            self._archive = typing.cast(Population, population[non_dom])
            self.pop = typing.cast(Population, population[np.setdiff1d(np.arange(len(population)), non_dom)])

        self.logger.debug(f"Individuals in archive: {len(self._archive)}, Individuals in pop {len(self.pop)}")

    def _track_dominators(self):
        "Method that tracks which archive members dominate population members."

        self.logger.debug(f"Tracking domiantors.")

        # Safety check
        assert self._archive is not None, f"Tried to track dominators but archive was None."
        assert self.pop is not None, f"Tried to track dominators but population was None."

        for p_member in self.pop:
            for a_member in self._archive:
                if Dominator().get_relation(a = a_member.F , b = p_member.F) == 1: # a_member dominates p_member
                    self._set_dominator(p_member, a_member)
                    break
