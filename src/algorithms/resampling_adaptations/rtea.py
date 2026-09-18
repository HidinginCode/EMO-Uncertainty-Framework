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
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting as NDS
from pymoo.util.dominator import Dominator
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.gauss import GaussianMutation

# Python imports
import typing
import copy

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
        else: # If we do not create a new child we copy the parent and mutate it
            new_ind = copy.deepcopy(self._archive[[chosen_indices[0]]])

        # Mutate the new_ind
        new_ind = GaussianMutation(sigma = 0.2).do(problem = self.problem, pop = new_ind, random_state = self.random_state)

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

    def _update_front(self, individual):
        """Method that checks an individual against the current archive and places it in the archive or the search population.

        Args:
            individual (Individual): Individual to check against the archive."""

        # If individual is not domianted by any archive member it is added
        # All archive members that are dominated get returned to pop with the individual as their tracked dominator
        # If it is dominated by the archive then one archive member is chosen to be its domiantor and it is added to the search population

        removed_archive_indices = []
        # Dominance check against archive
        for i, a_member in enumerate(self._archive):
            dom_realation = Dominator().get_relation(a = a_member.F , b = individual.F)

            match dom_realation:
                case 1: # Archive member dominates individual
                    individual.set("dominator", a_member)
                    self.pop = Population.merge(a = self.pop, b = individual)
                    break
                case -1: # Individual dominates archive member
                    a_member.set("dominator", individual)
                    removed_archive_indices.append(i)
        else: # Loop completed without breaking, so individual was not dominated by the archive
            self._archive = Population.merge(a = self._archive, b = individual)

        # Take archive member indices and transfer them to pop
        transfer_pop = self._archive[removed_archive_indices]
        self.pop = Population.merge(a = self.pop, b = transfer_pop)
        self._archive = self._archive[np.setdiff1d(np.arange(len(self._archive)), removed_archive_indices)]  # pyright: ignore[reportAttributeAccessIssue]

    def _resample_archive(self):
        """Method that reevaluates the archive members with the fewest samples to refine their objective estimates."""

        self.logger.debug("Resampling archive.")

        for _ in range(min(self._archive_resamples, len(self._archive))):
            n_evals = self._archive.get("n_evals")
            chosen_index = np.argmin(n_evals)
            chosen = self._archive[chosen_index]

            assert self.pop is not None, f"Population was None while trying to resample the archive."
            # Pop members currently tracking chosen as their dominator need to be rechecked once its estimate changes
            tracking_mask = np.array([p_member.get("dominator") is chosen for p_member in self.pop], dtype = bool)
            rechecked = self.pop[tracking_mask]
            self.pop = self.pop[~tracking_mask]  # pyright: ignore[reportAttributeAccessIssue]

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
                self._update_front(p_member)

    def _in_search_phase(self) -> bool:
        """Method that checks whether the run is still in its search phase, as opposed to the pure archive refinement phase."""

        # Safety check
        assert self.termination is not None, f"Termination criterion was none in _in_search_phase method."
        assert hasattr(self.termination, "n_max_evals"), f"Termination had no attribute n_max_evals"

        return self.evaluator.n_eval < (1 - self._archive_refinement) * self.termination.n_max_evals  # pyright: ignore[reportAttributeAccessIssue]

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
                    p_member.set("dominator", a_member)
                    break
