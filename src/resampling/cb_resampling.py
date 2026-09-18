"""This module contains the implementation of confidence bound resampling."""

# Make other project modules accessible
import sys
import os
sys.path.insert(1, os.path.join(os.path.dirname(__file__), ".."))

# Logger import
from config.logging_mixin import LoggingMixin

# Package imports
import numpy as np

# Pymoo imports
from pymoo.core.algorithm import Algorithm
from pymoo.core.population import Population
from pymoo.core.individual import Individual

# Python imports
from random import randrange

class CB_Resampling(LoggingMixin):
    """Class that contains the resampling methods."""

    @staticmethod
    def resample(algo_instance: Algorithm, dynamic_threshold: bool = True, threshold: np.ndarray = None, **kwargs):
        """Takes an algorithm instance and applies confidence bound resampling to its individuals.

        Args:
            algo_instance (Algorithm): Instance of the pymoo genetic algorithm.
            dynamic_threshold (bool, optional): Decides if fixed or dynamic threshold is used. Defaults to True.
            threshold (np.ndarray, optional): Threshold for which to re-evaluate (ind.uncertainty >= threshold -> re-eval). Defaults to None.
        """
        CB_Resampling.logger.debug("Starting population resampling.")

        if dynamic_threshold:
            # Calculate threshold
            dyn_threshold = CB_Resampling._calc_threshold(algo_instance = algo_instance, method = "population")
            CB_Resampling._threshold_resampling(algo_instance=algo_instance, threshold = dyn_threshold)
        else:
            # Use fixed threshold
            CB_Resampling._threshold_resampling(algo_instance = algo_instance, threshold = threshold)


    @staticmethod
    def _prepare_infills(infills=None):
        """Adds re_evaluation counter and running mean to the new infills.

        Args:
            infills (Pymoo infills): New individuals added after recombination and mutation. Defaults to None.
        """
        # Preparing new infills
        if infills is not None:
            CB_Resampling.logger.debug("Adding evaluation counter, f_sum, mean and last threshold to new infills.")
            for ind in infills:
                if not hasattr(ind, "n_evals"):
                    ind.n_evals = 1
                    ind.f_sum = np.zeros_like(ind.F)
                    ind.mean = 0
                    ind.old_mean = np.zeros_like(ind.F)
                    ind.suggested_threshold = np.random.random(len(ind.f_sum))
                ind.f_sum += ind.F
                ind.mean = ind.f_sum / ind.n_evals
                ind.F = ind.mean

    @staticmethod
    def _threshold_resampling(algo_instance: Algorithm, threshold: np.ndarray):
        """Re-evaluates individuals that are over the specified threshold.

        Args:
            algo_instance (Algorithm): Instance of the pymoo genetic algorithm.
            threshold (np.ndarray): Threshold for which to re-evaluate (ind.uncertainty >= threshold -> re-eval)
        """

        CB_Resampling.logger.debug("Applying CB resampling.")
        # Here we need to decide what T is for the CB formula
        # Preliminary T will be made up off the re-eval counts of k-nearest neighbors
        # Subsequently k is also a tunable parameter

        assert algo_instance.pop is not None, f"Population of {algo_instance.__class__.__name__} was None but should not have been."

        pop_knn = CB_Resampling._knn(pop = algo_instance.pop, k = 5)

        # Calculate T values and the indicator value --> Decide if resampling is necessary
        for knn in pop_knn:
            T = 0
            ind = knn[0]
            for neighbor in knn[1]:
                T += neighbor.n_evals
            CB_Resampling.logger.debug(f"T value: {T}")
            uncertainty_indicator = np.sqrt(2*np.log(T)/knn[0].n_evals)

            CB_Resampling.logger.debug(f"Calculated uncertainty indicator: {uncertainty_indicator}")

            # If uncertainty is higher than threshold resample
            if uncertainty_indicator > threshold:
                algo_instance.evaluator.eval(problem=algo_instance.problem, pop = ind, algorithm = algo_instance, skip_already_evaluated=False)

                # Refresh mean and update counter
                # F here is the value that was obtained from the evaluator
                ind.old_mean = ind.mean
                ind.n_evals += 1
                ind.f_sum += ind.F
                ind.mean = ind.f_sum / ind.n_evals
                ind.F = ind.mean
                ind.last_threshold = ind.suggested_threshold

                # Recalculate individual threshold suggestion
                ind.suggested_threshold = (((ind.old_mean - ind.mean)/ind.old_mean)+1)*ind.last_threshold
                CB_Resampling.logger.debug(f"New suggested threshold: {ind.suggested_threshold}")
            else:
                ind.suggested_threshold = ind.suggested_threshold * 0.9 # Lower the threshold by 10%

    @staticmethod
    def _minkowski_dist(x: list, y: list, p: float = 2) -> float:
        """Calculates the Minkowski dist between two points x and y.

        Args:
            x (list): Point x as list
            y (list): Point y as list
            p (float, optional): P value for formula. When p=2 euclidean distance, when p=1 manhattan distance. Defaults to 2.

        Returns:
            float: Minkowski distance between x and y
        """

        #CB_Resampling.logger.debug("Calculating Minkowski distance.")

        # Safety checks
        assert isinstance(x, list) or isinstance(x, np.ndarray), f"Expected x to be list or np.ndarray but got {type(x)}"
        assert isinstance(y, list) or isinstance(y, np.ndarray), f"Expected y to be list or np.ndarray but got {type(y)}"

        assert np.shape(x) == np.shape(y), f"x and y should be of same shape but x={np.shape(x)}, y={np.shape(y)}"

        # If x or y not np.ndarray convert
        x_tmp = np.array(object=x)
        y_tmp = np.array(object=y)

        minkowski = np.sum(np.abs(x_tmp-y_tmp)**p)**(1/p)

        return minkowski

    @staticmethod
    def _knn(pop: Population, k: int = 5) -> list:
        """Returns a list of k nearest neighbors for every individual in the population.

        Args:
            pop (Population): Pymoo population for which to compute KNN.
            k (int, optional): Number of neighbors. Defaults to 5.

        Returns:
            list: List of individuals and their KNN.
        """

        CB_Resampling.logger.debug(f"Calculating KNN for population of size {len(pop)}.")
        neighbors = []
        for ind in pop:
            distances = []
            for ind2 in pop:
                if ind is ind2: continue

                distances.append([ind2, CB_Resampling._minkowski_dist(ind.mean, ind2.mean)])

            # Sort based on distance to ind
            distances.sort(key=lambda x: x[1])

            # Add to neighbors
            neighbors.append([ind, [d[0] for d in distances[:k]]]) # Only append individual not the distance
        return neighbors

    @staticmethod
    def _calc_threshold(algo_instance: Algorithm, method: str = "population") -> np.ndarray:
        """Calculates the threshold value using the chosen method.

        Args:
            algo_instance (Algorithm): Instance of the current algorithm.
            method (str, optional): Method that decides how threshold is calculated. Available methods population, ... . Defaults to "population".

        Return:
            float: Calculated threshold.
        """

        CB_Resampling.logger.debug(f"Calculating Threshold")

        # Safety checks
        assert method in ["population"], f"The chosen method for threshold calculation is invalid, options are [population, ...]. Chosen was {method}"
        assert algo_instance.pop is not None and len(algo_instance.pop) > 0, f"The supplied algorithm instance did not have a population (was None or had length 0)."
        assert algo_instance.n_gen is not None, f"Generation counter in algorithm instance was None. It should be a positive integer."
        assert algo_instance.problem is not None, f"Problem of the supplied algorithm instance was None."

        # Case for first generation of algorithm where all individuals have no prior threshold
        # In this case threshold is set to 1 globally TODO: Figure out good value for this case
        CB_Resampling.logger.debug(f"Algorithm Generation {algo_instance.n_gen}")
        if algo_instance.n_gen <= 1:
            CB_Resampling.logger.debug(f"First generation so threshold 1 was used.")
            return np.ones(algo_instance.problem.n_obj)*3

        # Additional attribute check
        assert all(hasattr(ind, "suggested_threshold") for ind in algo_instance.pop), f"Not all individuals in the algorithm instance population had the attribute 'last_threshold'."

        CB_Resampling.logger.debug(f"Calculating threshold using method {method}.")

        mean_threshold = np.zeros_like(algo_instance.problem.n_obj)

        if method == "population":
            # Aggregate all old threshold values
            CB_Resampling.logger.debug(f"Number of individuals that contribute to threshold {len(algo_instance.pop)}")
            mean_threshold = np.mean([ind.suggested_threshold for ind in algo_instance.pop], axis = 0)

            CB_Resampling.logger.debug(f"Calculated mean was {mean_threshold} ...")
        return mean_threshold
