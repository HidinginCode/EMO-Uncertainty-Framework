"""This module contains tests for the CB_Resampling NSGA-II class."""
# Make other project modules accessible
import sys
import os
sys.path.insert(1, os.path.join(os.path.dirname(__file__), ".."))

# Logging config
from config.logging_config import setup_logging

# CBR-NSGA2 import
from src.algorithms.resampling_adaptations.cbr_nsga2 import CBR_NSGA2

# CBR Callback import
from callbacks.cbr_callback import CBR_Callback

# Problem import
from src.problems.noisy_zdt import NZDT1

# Pymoo imports
from pymoo.optimize import minimize

# Pickle import
import pickle

# Set up logging
setup_logging()

# Create algorithm instance
algo_instance = CBR_NSGA2(resampling_threshold=0.5)

print("Creating CBR_NSGA-II: [X]")

# Create problem and try to minimize
problem = NZDT1(noise_std=0.6)

# Define callback to keep acess later
callback = CBR_Callback()

minimize(
    problem=problem,
    algorithm = algo_instance,
    termination=("n_eval",10000),
    callback = callback
)

callback.save_data()

with open("/home/malte/Documents/Work/Uncertainty-Framework-WIP/data/results.pkl", "rb") as f:
    df = pickle.load(file = f)
    print(df)
