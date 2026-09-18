"""This module contains the code for the confidence based resampling (CBR) experiments."""

# Make other project modules accessible
import sys
import os

from numpy._core import minimum

sys.path.insert(1, os.path.join(os.path.dirname(__file__), ".."))

# Logging config import
from config.logging_config import setup_logging

# Import Problems, Algorithms and Callback
from src.problems.noisy_dtlz import *
from src.problems.noisy_wfg import *
from src.problems.noisy_zdt import *
from src.algorithms.resampling_adaptations.cbr_nsga2 import CBR_NSGA2
from src.algorithms.resampling_adaptations.static_res_nsga2 import SR_NSGA2
from src.algorithms.resampling_adaptations.mocba_nsga2 import MOCBA_NSGA2
from src.algorithms.resampling_adaptations.rtea import RTEA
from src.callbacks.cbr_callback import CBR_Callback
from src.callbacks.rtea_callback import RTEA_Callback

# Package imports
import numpy as np

# Pymoo imports
from pymoo.core.problem import Problem
from pymoo.optimize import minimize

# Python imports
import random
from pathlib import Path
import argparse
import multiprocessing as mp

# Results path as global variable
RESULTS_ROOT = Path(__file__).resolve().parents[1] / "results"

def tau_sweep(seed: int, threshold: float, problem: str, std: float, noise_type: str):
    """Method that starts the experiments.

    Args:
        seed (int): Experiment seed for RNG.
        threshold (float): Threshold that should be tested.
        problem (Problem): Pymoo problem that is to be optimized.
    """

    # Set seeds
    random.seed(seed)
    np.random.seed(seed)

    # Initialize Algorithm, Callback and Problem
    algo_instance = CBR_NSGA2(resampling_threshold=threshold)
    problem_instance = None

    match problem:
        case "NZDT1": problem_instance = NZDT1(noise_std=std, noise_type=noise_type)
        case "NZDT2": problem_instance = NZDT2(noise_std=std, noise_type=noise_type)
        case "NZDT3": problem_instance = NZDT3(noise_std=std, noise_type=noise_type)
        case "NZDT4": problem_instance = NZDT4(noise_std=std, noise_type=noise_type)
        case "NZDT5": problem_instance = NZDT5(noise_std=std, noise_type=noise_type)
        case "NZDT6": problem_instance = NZDT6(noise_std=std, noise_type=noise_type)
        case "NDTLZ1": problem_instance = NDTLZ1(noise_std=std, noise_type=noise_type)
        case "NDTLZ2": problem_instance = NDTLZ2(noise_std=std, noise_type=noise_type)
        case "NDTLZ3": problem_instance = NDTLZ3(noise_std=std, noise_type=noise_type)
        case "NDTLZ7": problem_instance = NDTLZ7(noise_std=std, noise_type=noise_type)

    callback_instance = CBR_Callback()

    minimize(
        problem=problem_instance,
        algorithm=algo_instance,
        callback = callback_instance,
        termination=("n_eval", 200000)
    )

    # Save data to pickle files
    path = _result_dir(
        sweep = "tau_sweep",
        algo = type(algo_instance).__name__.lower(),
        problem = problem,
        noise_type = noise_type,
        std = std,
        param_dir = f"tau_{threshold:.3f}",
        seed = seed,
    )
    callback_instance.save_data(data_path = path, wipe_old_data=False)

def resampling_sweep(seed: int, problem: str, std: float, noise_type: str, n_samples: int):
    """Method used to launch experiments for static resampling NSGA-II.

    Args:
        seed (int): Seed used for RNG.
        problem (str): Problem name as string.
        std (float): Standard deviation for noise.
        n_samples (int): Number of static samples."""

    # Set seeds
    random.seed(seed)
    np.random.seed(seed)

    # Initialize Algorithm, Callback and Problem
    algo_instance = SR_NSGA2(k = n_samples)
    problem_instance = None

    match problem:
        case "NZDT1": problem_instance = NZDT1(noise_std=std, noise_type=noise_type)
        case "NZDT2": problem_instance = NZDT2(noise_std=std, noise_type=noise_type)
        case "NZDT3": problem_instance = NZDT3(noise_std=std, noise_type=noise_type)
        case "NZDT4": problem_instance = NZDT4(noise_std=std, noise_type=noise_type)
        case "NZDT5": problem_instance = NZDT5(noise_std=std, noise_type=noise_type)
        case "NZDT6": problem_instance = NZDT6(noise_std=std, noise_type=noise_type)
        case "NDTLZ1": problem_instance = NDTLZ1(noise_std=std, noise_type=noise_type)
        case "NDTLZ2": problem_instance = NDTLZ2(noise_std=std, noise_type=noise_type)
        case "NDTLZ3": problem_instance = NDTLZ3(noise_std=std, noise_type=noise_type)
        case "NDTLZ7": problem_instance = NDTLZ7(noise_std=std, noise_type=noise_type)

    callback_instance = CBR_Callback()

    minimize(
        problem=problem_instance,
        algorithm=algo_instance,
        callback = callback_instance,
        termination=("n_eval", 200000)
    )

    # Save data to pickle files
    path = _result_dir(
        sweep = "static_sweep",
        algo = type(algo_instance).__name__.lower(),
        problem = problem,
        noise_type = noise_type,
        std = std,
        param_dir = f"k_{n_samples:03d}",
        seed = seed,
    )
    callback_instance.save_data(data_path = path, wipe_old_data=False)

def mocba_sweep(seed: int, problem: str, std: float, noise_type: str, budget: int, init_replications: int):
    """Method used to launch experiments for MOCBA NSGA-II.

    Args:
        seed (int): Seed used for RNG.
        problem (str): Problem name as string.
        std (float): Standard deviation for noise.
        noise_type (str): Selected noise type.
        budget (int): Resampling budget distributed per generation.
        init_replications (int): Initial number of replications per individual."""

    # Set seeds
    random.seed(seed)
    np.random.seed(seed)

    # Initialize Algorithm, Callback and Problem
    algo_instance = MOCBA_NSGA2(init_replications = init_replications, resampling_budget = budget)
    problem_instance = None

    match problem:
        case "NZDT1": problem_instance = NZDT1(noise_std=std, noise_type=noise_type)
        case "NZDT2": problem_instance = NZDT2(noise_std=std, noise_type=noise_type)
        case "NZDT3": problem_instance = NZDT3(noise_std=std, noise_type=noise_type)
        case "NZDT4": problem_instance = NZDT4(noise_std=std, noise_type=noise_type)
        case "NZDT5": problem_instance = NZDT5(noise_std=std, noise_type=noise_type)
        case "NZDT6": problem_instance = NZDT6(noise_std=std, noise_type=noise_type)
        case "NDTLZ1": problem_instance = NDTLZ1(noise_std=std, noise_type=noise_type)
        case "NDTLZ2": problem_instance = NDTLZ2(noise_std=std, noise_type=noise_type)
        case "NDTLZ3": problem_instance = NDTLZ3(noise_std=std, noise_type=noise_type)
        case "NDTLZ7": problem_instance = NDTLZ7(noise_std=std, noise_type=noise_type)

    callback_instance = CBR_Callback()

    minimize(
        problem=problem_instance,
        algorithm=algo_instance,
        callback = callback_instance,
        termination=("n_eval", 200000)
    )

    # Save data to pickle files
    path = _result_dir(
        sweep = "mocba_sweep",
        algo = type(algo_instance).__name__.lower(),
        problem = problem,
        noise_type = noise_type,
        std = std,
        param_dir = f"budget_{budget:04d}_r{init_replications:02d}",
        seed = seed,
    )
    callback_instance.save_data(data_path = path, wipe_old_data=False)

def rtea_sweep(seed: int, problem: str, std: float, noise_type: str, archive_resamples: int):
    """Method used to launch experiments for RTEA.

    Args:
        seed (int): Seed used for RNG.
        problem (str): Problem name as string.
        std (float): Standard deviation for noise.
        noise_type (str): Selected noise type.
        archive_resamples (int): Number of archive resamples per iteration (k). Everything else stays at Fieldsend's reported defaults."""

    # Set seeds
    random.seed(seed)
    np.random.seed(seed)

    # Initialize Algorithm, Callback and Problem
    algo_instance = RTEA(archive_resamples = archive_resamples)
    problem_instance = None

    match problem:
        case "NZDT1": problem_instance = NZDT1(noise_std=std, noise_type=noise_type)
        case "NZDT2": problem_instance = NZDT2(noise_std=std, noise_type=noise_type)
        case "NZDT3": problem_instance = NZDT3(noise_std=std, noise_type=noise_type)
        case "NZDT4": problem_instance = NZDT4(noise_std=std, noise_type=noise_type)
        case "NZDT5": problem_instance = NZDT5(noise_std=std, noise_type=noise_type)
        case "NZDT6": problem_instance = NZDT6(noise_std=std, noise_type=noise_type)
        case "NDTLZ1": problem_instance = NDTLZ1(noise_std=std, noise_type=noise_type)
        case "NDTLZ2": problem_instance = NDTLZ2(noise_std=std, noise_type=noise_type)
        case "NDTLZ3": problem_instance = NDTLZ3(noise_std=std, noise_type=noise_type)
        case "NDTLZ7": problem_instance = NDTLZ7(noise_std=std, noise_type=noise_type)

    callback_instance = RTEA_Callback()

    minimize(
        problem=problem_instance,
        algorithm=algo_instance,
        callback = callback_instance,
        termination=("n_eval", 200000)
    )

    # Save data to pickle files
    path = _result_dir(
        sweep = "rtea_sweep",
        algo = type(algo_instance).__name__.lower(),
        problem = problem,
        noise_type = noise_type,
        std = std,
        param_dir = f"k_{archive_resamples:02d}",
        seed = seed,
    )
    callback_instance.save_data(data_path = path, wipe_old_data=False)

def _result_dir(sweep: str, algo: str, problem: str, noise_type: str, std: float, param_dir: str, seed: int):
    """Method that builds a path for results based on supplied arguments.

    Args:
        sweep (str): Sweep name / top-level results folder (e.g. "tau_sweep", "static_sweep").
        algo (str): Algorithm name as a string.
        problem (str): Prolem name as a string.
        noise_type (str): Selected noise type.
        std (float): Standard deviation.
        param_dir (str): Swept-parameter folder name (e.g. "tau_1.500", "k_010").
        seed (int): Chosen seed for the run.
    """
    return (
            RESULTS_ROOT
            / sweep
            / algo
            / problem
            / noise_type
            / f"std_{std:.3f}"
            / param_dir
            / f"seed_{seed:04d}.pkl"
        )

if __name__ == "__main__":
    #setup_logging()
    parser = argparse.ArgumentParser(
        prog = "Noisy NSGA-II resampling sweeps",
        description = "Runs all seeds of one (algo, problem, std, param) experiment, one process per seed."
    )
    parser.add_argument("--algo", type=str, required=True, choices=["cbr", "sr", "mocba", "rtea"],
                   help="cbr = CBR-NSGA-II (tau sweep), sr = static-resampling NSGA-II (k sweep), "
                        "mocba = MOCBA-NSGA-II (budget sweep), rtea = RTEA (archive resamples k sweep)")
    parser.add_argument("--problem", type=str, required=True,
                   choices=['NZDT1', 'NZDT2', 'NZDT3', 'NZDT4', 'NZDT6', 'NDTLZ1', 'NDTLZ2', 'NDTLZ3', 'NDTLZ7'])
    parser.add_argument("--std", type=float, required=True)
    parser.add_argument("--noise-type", type=str, required=True,
                   choices=["gaussian", "..."])
    parser.add_argument("--n-seeds", type=int, default=31, help="Seeds 0 .. n-seeds-1. Defaults to 31.")

    # Algorithm-specific swept parameters (validated against --algo below).
    parser.add_argument("--threshold", type=float,
                   help="Resampling threshold tau. Required for --algo cbr.")
    parser.add_argument("--n-samples", type=int,
                   help="Static number of samples k. Required for --algo sr.")
    parser.add_argument("--budget", type=int,
                   help="MOCBA resampling budget distributed per generation. Required for --algo mocba.")
    parser.add_argument("--init-replications", type=int, default=5,
                   help="MOCBA initial replications per individual. Defaults to 5. Only used for --algo mocba.")
    parser.add_argument("--archive-resamples", type=int,
                   help="RTEA archive resamples per iteration k. Required for --algo rtea.")

    args = parser.parse_args()

    seeds = list(range(args.n_seeds))

    if args.algo == "cbr":
        if args.threshold is None:
            parser.error("--threshold is required when --algo cbr")
        func = tau_sweep
        tasks = [(s, args.threshold, args.problem, args.std, args.noise_type) for s in seeds]
    elif args.algo == "sr":
        if args.n_samples is None:
            parser.error("--n-samples is required when --algo sr")
        func = resampling_sweep
        tasks = [(s, args.problem, args.std, args.noise_type, args.n_samples) for s in seeds]
    elif args.algo == "mocba":
        if args.budget is None:
            parser.error("--budget is required when --algo mocba")
        func = mocba_sweep
        tasks = [(s, args.problem, args.std, args.noise_type, args.budget, args.init_replications) for s in seeds]
    else:  # args.algo == "rtea"
        if args.archive_resamples is None:
            parser.error("--archive-resamples is required when --algo rtea")
        func = rtea_sweep
        tasks = [(s, args.problem, args.std, args.noise_type, args.archive_resamples) for s in seeds]

    with mp.Pool(processes=len(seeds)) as pool:
        pool.starmap(func, tasks)
