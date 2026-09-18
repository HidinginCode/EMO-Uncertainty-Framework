"""This module contains the tests for the Rolling Tide Evolutionary Algorithm."""
# Make other project modules accessible
import sys
import os
sys.path.insert(1, os.path.join(os.path.dirname(__file__), ".."))

# Logging config
from config.logging_config import setup_logging

# RTEA import
from src.algorithms.resampling_adaptations.rtea import RTEA

# CBR Callback import
from src.callbacks.rtea_callback import RTEA_Callback

# Problem import
from src.problems.noisy_zdt import NZDT1

# Pymoo imports
from pymoo.optimize import minimize

# Set up logging
setup_logging()

# Create algorithm instance
algo_instance = RTEA()

print("Creating RTEA: [X]")

# Create problem and try to minimize
problem = NZDT1(noise_std=0.6)

# Define callback to keep acess later
callback = RTEA_Callback()

minimize(
    problem=problem,
    algorithm = algo_instance,
    termination=("n_eval",10000),
    callback = callback
)

callback.save_data()
