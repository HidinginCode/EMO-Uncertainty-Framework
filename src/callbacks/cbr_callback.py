"""This module implements a standard callback that collects most useful metrics."""
# Make other project modules accessible
import sys
import os

sys.path.insert(1, os.path.join(os.path.dirname(__file__), ".."))

# Logger import
from config.logging_mixin import LoggingMixin

# Package import
import numpy as np
import pandas as pd

# Pymoo import
from pymoo.core.callback import Callback
from pymoo.core.algorithm import Algorithm
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from pymoo.indicators.igd_plus import IGDPlus
from pymoo.indicators.gd_plus import GDPlus

# Python imports
from pathlib import Path
from shutil import rmtree
import pickle

class CBR_Callback(Callback, LoggingMixin):
    """This class implements the standard callback used in most experiments."""

    def __init__(self) -> None:
        self.means = []
        self.suggested_thresholds = []
        self.re_evaluations = [] # Re-evaluations (summed) per generation
        self.evaluations_per_gen = [] # Stores the evaluation counter after every generation
        self.clean_pops = [] # Stores populations evaluated without noise
        self.clean_fronts = []
        self.igd_plus = []
        self.gd_plus = []
        self.logger.info(f"Instance of {self.__class__.__name__} created.")
        super().__init__()

    def notify(self, algorithm: Algorithm) -> None:
        """Handler function that runs at end of every generation to log results.

        Args:
            algorithm (Algorithm): Instance of the algorithm for which to collect data."""

        # Safety checks
        assert algorithm.pop is not None, f"Tried to execute notify function, but algorithm.pop was None."
        assert algorithm.problem is not None, f"Problem instance inside the supplied algorithm was None."
        assert all(hasattr(ind, "mean") for ind in algorithm.pop), f"Tried to execute notify function, but not all individuals of population have attribute 'mean'."
        assert all(hasattr(ind, "suggested_threshold") for ind in algorithm.pop), f"Tried to execute notify function, but not all individuals of population have attribute 'suggested_threshold'."
        assert all(hasattr(ind, "n_evals") for ind in algorithm.pop), f"Tried to execute notify function, but not all individuals of population have attribute 'n_evals'."
        assert callable(getattr(algorithm.problem, "evaluate_noiseless")), f"{algorithm.problem.__class__.__name__} does not have an 'evaluate_noiseless' funciton."
        assert callable(getattr(algorithm.problem, "pareto_front")), f"{algorithm.problem.__class__.__name__} does not have an 'pareto_front' function."

        # Compute true F values
        clean_pop = algorithm.problem.evaluate_noiseless(algorithm.pop.get("X"))    # pyright: ignore[reportAttributeAccessIssue]
        clean_front = clean_pop[NonDominatedSorting().do(F = clean_pop, only_non_dominated_front = True)]

        # Collect metrics
        self.means.append([ind.mean for ind in algorithm.pop])
        self.suggested_thresholds.append([ind.suggested_threshold for ind in algorithm.pop])
        self.re_evaluations.append(np.sum([ind.n_evals for ind in algorithm.pop]))
        self.evaluations_per_gen.append(algorithm.evaluator.n_eval)
        self.clean_pops.append(clean_pop)
        self.clean_fronts.append(clean_front)
        self.igd_plus.append(self.calculate_igd_plus(true_pf = algorithm.problem.pareto_front(), approx_pf = clean_front))
        self.gd_plus.append(self.calculate_gd_plus(true_pf = algorithm.problem.pareto_front(), approx_pf = clean_front))

        self.logger.debug(f"Callback collected metrics, Ammount --> Means:{len(self.means)}, Suggested Thresholds: {len(self.suggested_thresholds)}, Re-Evaluations: {len(self.re_evaluations)}.")
        return super().notify(algorithm)

    def save_data(self, data_path: str= f"{Path(__file__).resolve().parents[2]}/data/results.pkl", wipe_old_data = True):
        """Function that saves the current data of the callback to a predefined path.

        Args:
            data_path (str, Path, optional): Path that leads to the data directory (where it should be created). Defaults to project_root/data.
            wipe_old_data (bool, optional): Decides if existing data gets wiped and new directory is created instead. Defaults to true.
        """

        data_dir = Path(data_path).parent
        # Wipes data if path exists and argument is set correctly
        if wipe_old_data and Path.exists(Path(data_dir)):
            rmtree(data_dir)
            self.logger.debug(f"Removed existing {data_dir}")

        # Create data path
        self.logger.debug(f"Creating data directory at {data_dir}")
        os.makedirs(data_dir, exist_ok = True)

        # Create data dict
        data_dict = {
            # "clean_pops": self.clean_pops,
            "clean_fronts": self.clean_fronts,
            "means": self.means,
            "suggested_thresholds": self.suggested_thresholds,
            "re_evaluations": self.re_evaluations,
            "evaluations_per_gen": self.evaluations_per_gen,
            "igd_plus": self.igd_plus,
            "gd_plus": self.gd_plus
        }

        # Make dataframe
        frame = pd.DataFrame(data_dict)

        # Save to pickle (and maybe CSV)
        with open(data_path, "wb") as f:
            pickle.dump(obj = frame, file = f)

    def calculate_igd_plus(self, true_pf: np.ndarray, approx_pf) -> float:
        """Calculates the IGD+ between the true Pareto front and the approximated one.

        Args:
            true_pf(np.ndarray): Actual pareto front of the problem.
            approx_pf(): Current approximated pareto front."""

        igd_plus_indicator = IGDPlus(pf = true_pf)
        igd_plus_value = igd_plus_indicator.do(F = approx_pf)
        assert igd_plus_value is not None, f"Tried to calculate IGD+ value but was None."
        return igd_plus_value

    def calculate_gd_plus(self, true_pf: np.ndarray, approx_pf):
        """Calculates the GD+ between the true Pareto front and the approximated one.

        Args:
            true_pf(np.ndarray): Actual pareto front of the problem.
            approx_pf(): Current approximated pareto front."""

        gd_plus_indiator = GDPlus(pf = true_pf)
        gd_plus_value = gd_plus_indiator.do(F = approx_pf)
        assert gd_plus_value is not None, f"Tried to calculate GD+ value but was None."
        return gd_plus_value
