"This module containts the main experiments for confidence based resampling."

# Make other project modules accessible
import sys
import os

sys.path.insert(1, os.path.join(os.path.dirname(__file__), ".."))

# Algorithm and problem imports
from src.algorithms.resampling_adaptations.cbr_nsga2 import CBR_NSGA2
from src.algorithms.resampling_adaptations.mocba_nsga2 import MOCBA_NSGA2
from src.algorithms.resampling_adaptations.static_res_nsga2 import SR_NSGA2
from src.callbacks.cbr_callback import CBR_Callback
from src.problems.noisy_zdt import NZDT1, NZDT2, NZDT3, NZDT4, NZDT6

# Import Config
from config import parameters

# Package imports
import numpy as np

# Pymoo imports
from pymoo.optimize import minimize
from argparse import ArgumentParser

#Python imports
import random
from pathlib import Path
import multiprocessing as mp

# Results path as global variable
RESULTS_ROOT = Path(__file__).resolve().parents[1] / "results"

def start_experiment(args: tuple):
    """Method to start experiments using multiprocessing."""

    NOISE_LEVELS = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3]

    problem, noise_level, algorithm, parameters, seed = args # Extract values from tupled args

    # Set seeds
    random.seed(seed)
    np.random.seed(seed)

    problem_instance = None

    match problem:
        case "NZDT1": problem_instance = NZDT1(noise_std = NOISE_LEVELS[noise_level])
        case "NZDT2": problem_instance = NZDT2(noise_std = NOISE_LEVELS[noise_level])
        case "NZDT3": problem_instance = NZDT3(noise_std = NOISE_LEVELS[noise_level])
        case "NZDT4": problem_instance = NZDT4(noise_std = NOISE_LEVELS[noise_level])
        case "NZDT6": problem_instance = NZDT6(noise_std = NOISE_LEVELS[noise_level])
        case _: raise ValueError(f"Supplied problem instance name did not match implemented problems.")

    assert problem_instance is not None, f"Problem instance was None after problem should have been selected. Supplied name '{problem}'."

    algorithm_instance = None

    match algorithm:
        case "cbr":
            assert type(parameters) == float, f"Selected algorithm was CBR_NSGA2 but parameter was not of type float. Supplied parameter {parameters}, type: {type(parameters)}."
            algorithm_instance = CBR_NSGA2(resampling_threshold = parameters)
        case "mocba":
            assert type(parameters) == tuple, f"Selected algorithm was MOCBA NSGA2 but the supplied parameters were not a tuple. Supplied parameter {parameters}, type: {type(parameters)}."
            budget, init_repls = parameters
            algorithm_instance = MOCBA_NSGA2(init_replications=init_repls, resampling_budget=budget)
        case "sr":
            assert type(parameters) == int, f"Selected algorithm was Static Resampling NSGA2 but supplied parameter was not of type int. Supplied parameter {parameters}, type: {type(parameters)}."
            algorithm_instance = SR_NSGA2(k = parameters)
        case _: raise ValueError(f"Supplied algorithm name did not match implemented algorithms. Supplied algorihm name '{algorithm}'.")

    assert algorithm_instance is not None, f"Algorithm instance was None after algorithm should have been selected. Supplied name '{algorithm}'."

    callback_instance = CBR_Callback()

    print(f"Starting optimization --> Problem: {problem_instance.__class__.__name__}, Noise type: {problem_instance.noise_type}, Noise std: {problem_instance.noise_std}, Algorithm: {algorithm_instance.__class__.__name__}, Parameters: {parameters}.")

    res = minimize(
        problem=problem_instance,
        algorithm=algorithm_instance,
        termination=("n_eval", 300000),
        callback=callback_instance
    )

    path = _result_dir(
        algorithm=algorithm,
        parameters=parameters,
        problem=problem,
        noise_type="gaussian",
        std=NOISE_LEVELS[noise_level],
        seed=seed,
    )
    callback_instance.save_data(data_path=path, wipe_old_data=False)

def _result_dir(algorithm: str, parameters, problem: str, noise_type: str, std: float, seed: int) -> Path:
    """Method that builds a results path, covering all three algorithms.

    Derives the results folder and swept-parameter folder from `algorithm` and
    `parameters` (the same pair `start_experiment` already switches on), so
    callers don't have to branch on the algorithm themselves.

    Args:
        algorithm (str): "cbr", "mocba" or "sr" -- same key used to select the algorithm.
        parameters: Swept parameter(s) for `algorithm`: a float threshold (cbr), a
            (budget, init_replications) int tuple (mocba), or an int k (sr).
        problem (str): Problem name as a string.
        noise_type (str): Selected noise type.
        std (float): Standard deviation.
        seed (int): Chosen seed for the run.

    Returns:
        Path: results/main_eval/<algo>/<problem>/<noise_type>/std_X.XXX/<param_dir>/seed_YYYY.pkl
    """
    match algorithm:
        case "cbr":
            assert isinstance(parameters, float), f"cbr expects a float threshold, got {parameters!r}."
            algo, param_dir = "cbr_nsga2", f"tau_{parameters:.3f}"
        case "mocba":
            assert isinstance(parameters, tuple), f"mocba expects a (budget, init_replications) tuple, got {parameters!r}."
            budget, init_repls = parameters
            algo, param_dir = "mocba_nsga2", f"budget_{budget:04d}_r{init_repls:02d}"
        case "sr":
            assert isinstance(parameters, int), f"sr expects an int k, got {parameters!r}."
            algo, param_dir = "sr_nsga2", f"k_{parameters:03d}"
        case _:
            raise ValueError(f"Supplied algorithm name did not match implemented algorithms. Supplied algorithm name '{algorithm}'.")

    return (
            RESULTS_ROOT
            / "main_eval"
            / algo
            / problem
            / noise_type
            / f"std_{std:.3f}"
            / param_dir
            / f"seed_{seed:04d}.pkl"
        )

def main():
    parser = ArgumentParser()
    parser.add_argument("--algo", type=str, required=True, choices=["cbr", "sr", "mocba"])
    parser.add_argument("--noise_level", type=int, required=True, choices=[0, 1, 2, 3, 4, 5])
    parser.add_argument("--problem", type=str, required=True, choices=['NZDT1', 'NZDT2', 'NZDT3', 'NZDT4', 'NZDT6', 'NDTLZ1', 'NDTLZ2', 'NDTLZ3', 'NDTLZ7'])

    args = parser.parse_args()

    parameter_values = parameters.params[args.problem][args.algo][args.noise_level]

    seeds = list(range(31))

    run_configs = [(args.problem, args.noise_level, args.algo, parameter_values, seed) for seed in seeds]
    print(run_configs)
    with mp.Pool(processes=len(seeds)) as p:
        p.map(func=start_experiment, iterable=run_configs)

if __name__ == "__main__":
    main()
