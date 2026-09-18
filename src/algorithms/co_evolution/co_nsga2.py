"""This module contains the implementation of a co_evolving algorithm."""
from __future__ import annotations

# Make other project modules accessible
import sys
import os
sys.path.insert(1, os.path.join(os.path.dirname(__file__), ".."))

# Logger import
from config.logging_mixin import LoggingMixin

# Python imports
import random
import copy

# Pymoo imports
from pymoo.indicators.hv import HV
from pymoo.core.population import Population

# Package Imports
import numpy as np

# Module imports
from utils.callbacks import DualMetricsCallback
import utils.helper as hp

class DualEvolution(LoggingMixin):
    """This class creates a co-evolving evolutionary algorithm."""

    def __init__(self, algo1, algo2, problem, pop_size: int, n_gen: int = 0, n_eval: int = 0):
        """Init function for both algorithms.

        Args:
            algo1: First algorithm that has to be initialized.
            algo2: Second algorithm that has to be initialized.
            problem: Problem that should be treated.
            pop_size (int): Popsize that will be split between both algorithms.
            n_gen (int): Number of generations that will be split between the algorithms. Defaults to 0.
            n_eval (int): Number of function evaluations that will be split between both algorithms. Defaults to 0.
        """
        # Sanity checks
        if pop_size < 2:
            raise ValueError(f"Popsize must be at least 2, was {pop_size}!")
        
        if n_gen == 0 and n_eval == 0:
            raise ValueError("n_gen and n_eval can not both be 0!")
        
        # Setting variables
        if n_gen != 0 and n_eval != 0:
            self.logger.warning(msg = "Both n_gen and n_eval were set to not 0. N_gen will be prefered!", exc_info=True)
            self.term_criterion = ("n_gen", int(n_gen/2))
        elif n_gen != 0:
            self.term_criterion = ("n_gen", int(n_gen/2))
        else:
            self.term_criterion = ("n_eval", int(n_eval/2))
        self.logger.info(msg=f"Set termination criterion to {self.term_criterion}.")
        
        self.pop_size = int(pop_size/2)
        self.logger.info(msg=f"Set pop_size to {self.pop_size}")
        
        # Inititalize algorithms
        self.algo1 = algo1(pop_size = self.pop_size)
        self.algo2 = algo2(pop_size = self.pop_size)
        self.logger.info(msg = f"Initialized both algorithms succesfully. Algo 1: {self.algo1.__class__.__name__}, Algo 2: {self.algo1.__class__.__name__}")
        
        # Initialize problem
        if type(problem) == type: # Unitialized -> initialize with standard values
            self.problem = problem()
            self.logger.info(f"Initialized problem {self.problem.__class__.__name__}")
        else: 
            self.problem = problem
            self.logger.info(f"Problem was already initialized and is of class {self.problem.__class__.__name__}")
        
        # Create empty res object
        self.res: dict | None = None
        
        self.logger.info(f"Instance of {self.__class__.__name__} created.")

    def evolution(self):
        """Starts co evolution of the initialized algorithms."""
        
        # Since we can not use the normal minimize, we have to create the loop ourselfs
        self.logger.info(f"Creating algorithm setup for algo1...")
        self.algo1.setup(self.problem, termination = self.term_criterion, callback = DualMetricsCallback(problem=self.problem), verbose=False) # can include seed argument
        
        self.logger.info(f"Creating algorithm setup for algo2...")
        self.algo2.setup(self.problem, termination = self.term_criterion, callback = DualMetricsCallback(problem=self.problem), verbose=False)
        
        # Loop through generations of algorithms
        self.logger.info(f"Starting the loop through both algorithms")
        while self.algo1.has_next() or self.algo2.has_next():
            
            # Invoke next gen
            self.algo1.next()
            self.algo2.next()
            
            # HV Computation
            assert self.algo1.pop is not None and self.algo2.pop is not None
            
            # Compute ref point
            ref_point = hp.compute_hv_ref_point([self.algo1.opt, self.algo2.opt])
            hv_indicator = HV(ref_point=ref_point)

            # Compute hypervolumes
            alg1_hv = hv_indicator.do(self.algo1.opt.get("F"))
            alg2_hv = hv_indicator.do(self.algo2.opt.get("F"))
            
            # Sync fronts
            assert alg1_hv is not None and alg2_hv is not None
            
            if alg1_hv != alg2_hv:
                if alg1_hv > alg2_hv:
                    self.logger.debug(f"Algo 1 HV ({alg1_hv}) > Algo 2 HV ({alg2_hv})")
                    self._swap_individuals_hv(algo1=self.algo1, algo2=self.algo2, ref_point=ref_point)
                        
                else:
                    self.logger.debug(f"Algo 2 HV ({alg2_hv}) > Algo 1 HV ({alg1_hv})")
                    self._swap_individuals_hv(algo1=self.algo2, algo2=self.algo1, ref_point=ref_point)
            else:
                self.logger.debug(f"Hypervolume was equal {alg1_hv} == {alg2_hv}")

            self.logger.debug(f"Full HV: {alg1_hv} Hypervolume contributions: {hp.compute_hv_contribution(self.algo1.opt, ref_point=ref_point)}")       
        
        # Create resolution object
        self._create_res()

    def _create_res(self):
        """Creates the resolution based on algo1 and algo2 as a dictionary."""
        self.logger.info("Creating res dictionary ...")
        
        # Get results of algorithms
        res1 = copy.deepcopy(self.algo1.result())
        res2 = copy.deepcopy(self.algo2.result())
        
        # Merge algorithms together
        self.res = {
            "X": np.concatenate((res1.X, res2.X)),
            "F": np.concatenate((res1.F, res2.F)),
            "G": np.concatenate((res1.G, res2.G)), # G = constraints
            "CV": np.concatenate((res1.CV, res2.CV)), # CV = aggregated constraint violations
            "algorithms": (res1.algorithm, res2.algorithm), # Keep both algorithm objects (maybe important for callback)
            "opt": Population.merge(res1.opt, res2.opt),
            "pop": Population.merge(res1.pop, res2.pop),
            "exec_times": (res1.exec_time, res2.exec_time)
        }

    @staticmethod
    def _swap_individuals(algo1, algo1_index_list: list, algo2, algo2_index_list: list):
        """Swaps individuals of 2 algorithms, that can be specified via an index list.

        Args:
            algo1: Algorithm that individuals are taken from to insert into other.
            algo1_index_list (list): Index list for individuals in algo1.
            algo2: Algorithm that individuals are swapped out from
            algo2_index_list (list): Index list for individuals in algo2.
        """
        
        for idx in algo1_index_list:
            selected_algo2_id = random.choice(algo2_index_list)
            algo2.pop[selected_algo2_id] = copy.deepcopy(algo1.pop[idx])

    @staticmethod
    def _swap_individuals_hv(algo1, algo2, ref_point: np.ndarray):
        """Swaps the individuals using their hypervolume contribution.

        Args:
            algo1: Algorithm that individuals are taken from to insert into other.
            algo2: Algorithm that individuals are swapped out from
            ref_point (np.ndarray): Reference point for hypervolume
        """
        
        # Hypervolume contribution
        algo1_hv = hp.compute_hv_contribution(algo1.opt, ref_point = ref_point)
        algo2_hv = hp.compute_hv_contribution(algo2.opt, ref_point = ref_point)
        
        # opt_ids = {id(ind) for ind in self.algo1.opt}
        # alg1_opt_idx = [i for i, ind in enumerate(self.algo1.pop) if id(ind) in opt_ids]
        
        
        # Order indices
        algo1_hv_ordered = np.argsort(algo1_hv["hv_contribution"], kind="stable")[::-1] # Descending order
        algo2_hv_ordered = np.argsort(algo2_hv["hv_contribution"], kind="stable") # Ascending order
        
        # Get shorter list
        shorter_list = np.min([len(algo1_hv_ordered), len(algo2_hv_ordered)])
        
        # Swap em
        for i in range(int(shorter_list)):
            # Get individuals from contribution list
            ind1 = algo1_hv[algo1_hv_ordered[i]]["individual"]
            ind2 = algo2_hv[algo2_hv_ordered[i]]["individual"]
            
            idx2 = next(j for j, ind in enumerate(algo2.pop) if id(ind) == id(ind2))
            algo2.pop[idx2] = copy.deepcopy(ind1)