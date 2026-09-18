"""This module contains the implementation of NSGA-II using CB_Resampling."""

# Make other project modules accessible
import sys
import os
sys.path.insert(1, os.path.join(os.path.dirname(__file__), ".."))

# Logger import
from config.logging_mixin import LoggingMixin

# CB_Resampling import
from src.resampling.cb_resampling import CB_Resampling

# Package imports
import numpy as np

# Pymoo imports
from pymoo.algorithms.moo.nsga2 import NSGA2

class CBR_NSGA2(NSGA2, LoggingMixin):
    """This class implements NSGA-II with CB_Resampling."""

    def __init__(self, pop_size=100, resampling_threshold:float = 1.0, **kwargs):
        super().__init__(pop_size=pop_size, **kwargs)
        self.resampling_threshold  = resampling_threshold
        self.logger.info(f"Instance of {self.__class__.__name__} created.")

    def _initialize_advance(self, infills=None, **kwargs):
        self.logger.debug("Initializing advance")

        # Prepare infills based on CB_Resampling requirements
        CB_Resampling._prepare_infills(infills = infills)

        return super()._initialize_advance(infills, **kwargs)

    def _advance(self, infills=None, **kwargs):
        self.logger.debug("Advancing ...")

        # Prepare infills based on CB_Resampling requirements
        CB_Resampling._prepare_infills(infills = infills)

        self.logger.debug("Calling threshold resampling")
        CB_Resampling.resample(algo_instance=self, dynamic_threshold=False, threshold=self.resampling_threshold)

        return super()._advance(infills, **kwargs) # Survival happens in this step
