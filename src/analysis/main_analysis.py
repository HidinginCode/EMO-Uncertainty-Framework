"""This module is used for the evaluation of the main experiments.
While some methods might be similar to the tau sweep analysis, the main focus here is on comparing methods amongst each other and not with themselves."""

# Make other project modules accessible
import sys
import os
sys.path.insert(1, os.path.join(os.path.dirname(__file__), ".."))

# Package imports
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import streamlit as st

# Python imports
from pathlib import Path
import time
from concurrent.futures import ProcessPoolExecutor

# Pymoo imports
from pymoo.indicators.hv import HV

RESULTS_ROOT = "/home/malte/Documents/Work/Uncertainty-Framework-WIP/results/main_eval"

# The pickles hold every column CBR_Callback.save_data writes: "clean_fronts",
# "means", "suggested_thresholds", "re_evaluations", "evaluations_per_gen",
# "igd_plus", "gd_plus" (src/callbacks/cbr_callback.py:93-102). "means" and
# "suggested_thresholds" are per-individual, full-population columns -- they
# dwarf everything else and none of the plots below use them, so they're
# dropped at load time. "clean_fronts" is kept because the HV plot needs the
# actual objective vectors, not just the two scalar indicators.
NEEDED_COLUMNS = ["clean_fronts", "evaluations_per_gen", "igd_plus", "gd_plus"]


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #

def _collect_tasks() -> list[tuple[str, str, str, str, str, str, str]]:
    """Walks RESULTS_ROOT and lists every seed file to load.

    Returns:
        list: One (algo, problem, noise_type, std, parameter, seed_name, file_path)
        tuple per seed file.
    """
    tasks = []
    for algo in sorted(os.listdir(RESULTS_ROOT)):
        algo_path = os.path.join(RESULTS_ROOT, algo)

        for problem in sorted(os.listdir(algo_path)):
            problem_path = os.path.join(algo_path, problem)

            for noise_type in sorted(os.listdir(problem_path)): # This could theoretically be hardcoded to be gaussian, but maybe later other noise types
                noise_type_path = os.path.join(problem_path, noise_type)

                for std in sorted(os.listdir(noise_type_path)):
                    std_path = os.path.join(noise_type_path, std)

                    for parameter in sorted(os.listdir(std_path)):
                        parameter_path = os.path.join(std_path, parameter)

                        for file in sorted(os.listdir(parameter_path)):
                            if not file.endswith(".pkl"):
                                continue

                            seed_name = os.path.splitext(file)[0]  # e.g. "seed_0000"
                            file_path = os.path.join(parameter_path, file)
                            tasks.append((algo, problem, noise_type, std, parameter, seed_name, file_path))

    return tasks


def _load_leaf_file(task: tuple[str, str, str, str, str, str, str]):
    """Worker: unpickles one seed file and keeps only NEEDED_COLUMNS.

    Runs in a worker process. Unpickling itself is CPU-bound (reconstructing
    the object-dtype columns holds the GIL), so this is dispatched through a
    ProcessPoolExecutor rather than threads. Trimming to NEEDED_COLUMNS
    *before* returning matters too: it's what actually crosses the process
    boundary, and "means"/"suggested_thresholds" made up >90% of a file's
    pickled size in testing -- dropping them here cuts the return trip to
    near zero instead of paying the unpickle cost twice.

    Args:
        task: One entry from `_collect_tasks`.

    Returns:
        tuple: `task` with the file path replaced by `(df, error)`, where
        exactly one of the two is None -- a failed file is reported rather
        than raised, so it doesn't kill the whole pool.
    """
    algo, problem, noise_type, std, parameter, seed_name, file_path = task

    try:
        with open(file_path, "rb") as f:
            df = pickle.load(f)
        df = df[NEEDED_COLUMNS]
        error = None
    except Exception as exc:
        df = None
        error = f"{file_path}: {exc}"

    return algo, problem, noise_type, std, parameter, seed_name, df, error


def load_data(cache_path: str | None = None, refresh: bool = False, max_workers: int | None = None) -> dict:
    """Loads data from the results root.

    Structure: Algo -> Problem -> Noise Type -> Standard deviation -> Parameter -> "values" -> Seed -> DataFrame

    Unpickling ~2800 files serially took ~11 minutes (each file's dominant
    cost is CPU-bound object reconstruction, not disk I/O -- see the
    NEEDED_COLUMNS comment). This loads them across a process pool instead,
    and caches the assembled (already-trimmed) result to disk so repeat runs
    skip the raw files entirely.

    Args:
        cache_path: Where to cache the assembled result. Defaults to a
            `main_eval_cache.pkl` file next to RESULTS_ROOT.
        refresh: Reload from the raw .pkl files even if a cache exists.
        max_workers: Worker processes for unpickling; defaults to
            `os.cpu_count()` (via ProcessPoolExecutor).

    Returns:
        dict: The nested structure described above.
    """
    if cache_path is None:
        cache_path = os.path.join(os.path.dirname(RESULTS_ROOT), "main_eval_cache.pkl")

    if not refresh and os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    tasks = _collect_tasks()
    algo_results: dict = {}
    errors = []

    start = time.time()
    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        for algo, problem, noise_type, std, parameter, seed_name, df, error in pool.map(_load_leaf_file, tasks):
            if error is not None:
                errors.append(error)
                continue

            values = (
                algo_results
                .setdefault(algo, {})
                .setdefault(problem, {})
                .setdefault(noise_type, {})
                .setdefault(std, {})
                .setdefault(parameter, {"values": {}})
            )["values"]
            values[seed_name] = df
    print(f"Loaded {len(tasks) - len(errors)}/{len(tasks)} files in {time.time() - start:.1f}s")

    for error in errors:
        print(f"Skipped: {error}")

    with open(cache_path, "wb") as f:
        pickle.dump(algo_results, f, protocol=pickle.HIGHEST_PROTOCOL)

    return algo_results


def compile_config(values: dict) -> dict:
    """Reshapes one config's per-seed DataFrames into the per-metric,
    list-of-runs shape `homogenize_grid` / `hv_trajectories` expect (mirrors
    `tau_sweep_analysis.load_run_data`'s `combined_dict`).

    Args:
        values (dict): `{seed_name: DataFrame}`, as stored under
            `[...][parameter]["values"]` by `load_data`.

    Returns:
        dict: `{column: [per_run_array, ...]}` for each of NEEDED_COLUMNS.
    """
    compiled = {col: [] for col in NEEDED_COLUMNS}
    for df in values.values():
        for col in NEEDED_COLUMNS:
            compiled[col].append(df[col].to_numpy())
    return compiled


# --------------------------------------------------------------------------- #
# HV computation and grid homogenization -- ported from tau_sweep_analysis.py
# (that module runs its own multiprocessing sweep unconditionally on import,
# so these are copied rather than imported; kept behaviorally identical).
# --------------------------------------------------------------------------- #

def homogenize_grid(re_evaluations: list, metric: list):
    """Method used to create re_eval grid for multiple runs to plot metrics.
    Grid is needed because we plot with re_evaluations and runs collect metrics at different re_eval counts.

    Args:
        re_evaluations (list): #Function evaluations per generation per experiment.
        metric (list): Metric that is to be projected onto the grid.
    """

    # Create buckets to fit datapoints on same grid
    lo = max(e[0]  for e in re_evaluations) # Could also be set to 0 but get smallest possible number of evaluations from data
    hi = min(e[-1] for e in re_evaluations) # Get maximum number of function evals over all experiments
    grid = np.linspace(lo, hi, num=max(len(h) for h in metric)) # Create grid based on lowest, highest amount of function evals with #buckets = #most available values
    stacked = np.vstack([np.interp(grid, e, h)
                        for e, h in zip(re_evaluations, metric)]) # Interpolate data over the grid and stack results vertically

    mean_metric = stacked.mean(axis=0)
    std_metric  = stacked.std(axis=0, ddof=1) if stacked.shape[0] > 1 else np.zeros_like(mean_metric)

    return grid, mean_metric, std_metric


def hv_reference_point(compiled_data: list, scale: float = 1.1) -> np.ndarray:
    """Nadir-based hypervolume reference point over every individual in every run/gen.

    Args:
        compiled_data (list): Per run -> per generation -> iterable of objective vectors.
        scale (float): Factor applied to the component-wise maximum. Defaults to 1.1.

    Returns:
        np.ndarray: Reference point, `max(all objective vectors, axis=0) * scale`.
    """
    all_inds = [ind for data in compiled_data for gen in data for ind in gen]
    return np.asarray(all_inds).max(axis=0) * scale


def hv_trajectories(compiled_data: list, compiled_n_eval: list,
                    ref_point: np.ndarray | None = None, last_gen_only: bool = False):
    """Per-run hypervolume over generations.

    Args:
        compiled_data (list): Per run -> per generation -> iterable of objective vectors.
        compiled_n_eval (list): Per run -> per generation evaluation count.
        ref_point (np.ndarray, optional): Shared HV reference point. Derived from
            `compiled_data` via `hv_reference_point` when omitted.
        last_gen_only (bool): Evaluate HV only at each run's final generation.

    Returns:
        tuple: `(run_evals, run_hvs, ref_point)` with `run_evals` / `run_hvs` lists
        of 1D arrays aligned per run.
    """
    if ref_point is None:
        ref_point = hv_reference_point(compiled_data)
    hv_indicator = HV(ref_point=ref_point)

    run_hvs, run_evals = [], []
    for data, n_eval in zip(compiled_data, compiled_n_eval):
        gens = list(data)
        evals = np.asarray(list(n_eval), dtype=float)
        if last_gen_only:
            gens, evals = gens[-1:], evals[-1:]
        run_hvs.append(np.array([hv_indicator.do(F=np.asarray(gen)) for gen in gens])) # HV computation in fancy
        run_evals.append(evals)
    return run_evals, run_hvs, ref_point


def create_hv_plot(compiled_data: list, compiled_n_eval: list, title: str = "Cool Plot Bro"):
    """Creates a hypervolume plot for the combined data.
    The reference point is chosen as the nadir point of all submitted data multiplied by 1.1.

    Args:
        compiled_data (list): List of lists of datapoints [[data1], [data2], ...].
        compiled_n_eval (list): List of evaluation numbers per generation [[n_eval_per_gen experiment 1], ...].
        title (str, optional): Title for the created plot. Defaults to "Cool Plot Bro".
    """
    run_evals, run_hvs, _ = hv_trajectories(compiled_data, compiled_n_eval)
    grid, mean_hv, std_hv = homogenize_grid(re_evaluations=run_evals, metric=run_hvs)

    fig, ax = plt.subplots()
    ax.plot(grid, mean_hv, lw=2, label="mean HV")
    ax.fill_between(grid, mean_hv - std_hv, mean_hv + std_hv, alpha=0.2)
    ax.set_xlabel("n_eval"); ax.set_ylabel("hypervolume"); ax.legend()

    ax.set_title(title)
    return fig, grid, mean_hv, std_hv


def create_IGD_plus_plot(compiled_data: list, compiled_n_eval: list, title: str = "Cool Plot Bro"):
    """Creates a IGD+ plot for the combined data.

    Args:
        compiled_data (list): List of lists of datapoints [[data1], [data2], ...].
        compiled_n_eval (list): List of evaluation numbers per generation [[n_eval_per_gen experiment 1], ...].
        title (str, optional): Title for the created plot. Defaults to "Cool Plot Bro".
    """
    n_eval =[np.asarray(data, dtype=float) for data in compiled_n_eval]
    grid, mean_igd_plus, std_igd_plus = homogenize_grid(re_evaluations=n_eval, metric=compiled_data)
    fig, ax = plt.subplots()
    ax.plot(grid, mean_igd_plus, lw=2, label="Mean IGD+")
    ax.fill_between(grid, mean_igd_plus - std_igd_plus, mean_igd_plus + std_igd_plus, alpha = 0.2)
    ax.set_xlabel("Function Evaluations")
    ax.set_ylabel("IGD+")
    ax.set_title(title)

    return fig, grid, mean_igd_plus, std_igd_plus


def create_GD_plus_plot(compiled_data: list, compiled_n_eval: list, title: str = "Cool Plot Bro"):
    """Creates a GD+ plot for the combined data.

    Args:
        compiled_data (list): List of lists of datapoints [[data1], [data2], ...].
        compiled_n_eval (list): List of evaluation numbers per generation [[n_eval_per_gen experiment 1], ...].
        title (str, optional): Title for the created plot. Defaults to "Cool Plot Bro".
    """
    n_eval =[np.asarray(data, dtype=float) for data in compiled_n_eval]
    grid, mean_gd_plus, std_gd_plus = homogenize_grid(re_evaluations=n_eval, metric=compiled_data)
    fig, ax = plt.subplots()
    ax.plot(grid, mean_gd_plus, lw=2, label="Mean GD+")
    ax.fill_between(grid, mean_gd_plus - std_gd_plus, mean_gd_plus + std_gd_plus, alpha = 0.2)
    ax.set_xlabel("Function Evaluations")
    ax.set_ylabel("GD+")
    ax.set_title(title)

    return fig, grid, mean_gd_plus, std_gd_plus


# --------------------------------------------------------------------------- #
# Cross-algorithm comparison plots -- same underlying machinery as the
# single-series create_*_plot functions above, but overlaying one line per
# algorithm on shared axes so the three algorithms can be read off one plot.
# --------------------------------------------------------------------------- #

def create_hv_comparison_plot(series: dict[str, dict], title: str = "Cool Plot Bro"):
    """Overlays a mean-HV-over-evaluations line per algorithm on one plot.

    Uses a single HV reference point derived from every front across every
    algorithm in `series`, so hypervolumes stay comparable between algorithms
    (same idea as tau_sweep_analysis.get_best_runs, which shares one
    reference point across the configs it's ranking).

    Args:
        series (dict): `{algo_name: compiled_config_dict}`; each compiled
            dict as returned by `compile_config` (needs "clean_fronts" and
            "evaluations_per_gen").
        title (str, optional): Plot title. Defaults to "Cool Plot Bro".
    """
    all_fronts = [run for compiled in series.values() for run in compiled["clean_fronts"]]
    ref_point = hv_reference_point(all_fronts)

    fig, ax = plt.subplots()
    for algo in sorted(series):
        compiled = series[algo]
        run_evals, run_hvs, _ = hv_trajectories(compiled["clean_fronts"], compiled["evaluations_per_gen"], ref_point=ref_point)
        grid, mean_hv, std_hv = homogenize_grid(re_evaluations=run_evals, metric=run_hvs)
        ax.plot(grid, mean_hv, lw=2, label=algo)
        ax.fill_between(grid, mean_hv - std_hv, mean_hv + std_hv, alpha=0.15)

    ax.set_xlabel("n_eval"); ax.set_ylabel("hypervolume"); ax.legend()
    ax.set_title(title)
    return fig


def create_metric_comparison_plot(series: dict[str, dict], metric_col: str, ylabel: str, title: str = "Cool Plot Bro"):
    """Overlays a mean-metric-over-evaluations line per algorithm on one plot.
    Used for both IGD+ and GD+, which (unlike HV) are already comparable
    across algorithms since both are computed against the same true Pareto
    front in the callback -- no shared reference point needed here.

    Args:
        series (dict): `{algo_name: compiled_config_dict}`; each compiled
            dict as returned by `compile_config` (needs `metric_col` and
            "evaluations_per_gen").
        metric_col (str): "igd_plus" or "gd_plus".
        ylabel (str): Y-axis label.
        title (str, optional): Plot title. Defaults to "Cool Plot Bro".
    """
    fig, ax = plt.subplots()
    for algo in sorted(series):
        compiled = series[algo]
        n_eval = [np.asarray(data, dtype=float) for data in compiled["evaluations_per_gen"]]
        grid, mean_metric, std_metric = homogenize_grid(re_evaluations=n_eval, metric=compiled[metric_col])
        ax.plot(grid, mean_metric, lw=2, label=algo)
        ax.fill_between(grid, mean_metric - std_metric, mean_metric + std_metric, alpha=0.15)

    ax.set_xlabel("Function Evaluations"); ax.set_ylabel(ylabel); ax.legend()
    ax.set_title(title)
    return fig


# --------------------------------------------------------------------------- #
# Streamlit display
# --------------------------------------------------------------------------- #

@st.cache_data(show_spinner="Loading result data...")
def get_data(refresh: bool = False) -> dict:
    """Streamlit-cached wrapper around `load_data` (session-lifetime cache on
    top of `load_data`'s own disk cache)."""
    return load_data(refresh=refresh)


def _build_comparison_series(algo_results: dict, algos: list, problem: str, noise_type: str, std: str) -> tuple[dict, dict]:
    """Gathers, per algorithm, the compiled data for one (problem, noise_type, std)
    group -- i.e. what one comparison plot overlays. Normally there's exactly
    one parameter dir per (algo, problem, std) (e.g. one "tau_*" for cbr); if
    more than one is present their seeds are pooled under that algorithm.

    Returns:
        tuple: `(series, meta)` -- `series` is `{algo: compiled_config_dict}`
        for algorithms that have data for this group; `meta` is
        `{algo: (parameter_names, n_runs)}` for display captions.
    """
    series, meta = {}, {}
    for algo in algos:
        parameters = algo_results.get(algo, {}).get(problem, {}).get(noise_type, {}).get(std)
        if not parameters:
            continue

        values = {}
        for parameter, entry in parameters.items():
            values.update({f"{parameter}/{seed}": df for seed, df in entry["values"].items()})

        series[algo] = compile_config(values)
        meta[algo] = (",".join(sorted(parameters)), len(values))

    return series, meta


def main():
    st.set_page_config(page_title="Main Evaluation", layout="wide")
    st.title("Main Evaluation Results — Algorithm Comparison")

    refresh = st.sidebar.button("Reload from disk (ignore cache)")
    algo_results = get_data(refresh=refresh)

    all_algos = sorted(algo_results)
    if not all_algos:
        st.warning(f"No data found under {RESULTS_ROOT}.")
        return

    sel_algos = st.sidebar.multiselect("Algorithms to compare", all_algos, default=all_algos)
    if len(sel_algos) < 2:
        st.info("Select at least two algorithms to compare in the sidebar.")
        return

    problems = sorted({p for a in sel_algos for p in algo_results[a]})
    sel_problems = st.sidebar.multiselect("Problem", problems, default=problems[:1])
    if not sel_problems:
        st.info("Select at least one problem in the sidebar.")
        return

    for problem in sel_problems:
        st.header(problem)

        noise_types = sorted({n for a in sel_algos for n in algo_results[a].get(problem, {})})
        for noise_type in noise_types:
            stds = sorted(
                {s for a in sel_algos for s in algo_results[a].get(problem, {}).get(noise_type, {})},
                key=lambda s: float(s.removeprefix("std_")),
            )

            for std in stds:
                series, meta = _build_comparison_series(algo_results, sel_algos, problem, noise_type, std)
                if len(series) < 2:
                    continue  # Nothing to compare for this group.

                caption = ", ".join(f"{algo}={param} (n={n})" for algo, (param, n) in sorted(meta.items()))
                label = f"{problem} · {noise_type} · {std}"

                with st.expander(f"{noise_type} · {std}  —  {caption}"):
                    hv_fig = create_hv_comparison_plot(series, title=f"{label} — Hypervolume")
                    igd_fig = create_metric_comparison_plot(series, "igd_plus", "IGD+", title=f"{label} — IGD+")
                    gd_fig = create_metric_comparison_plot(series, "gd_plus", "GD+", title=f"{label} — GD+")

                    cols = st.columns(3)
                    for col, fig in zip(cols, (hv_fig, igd_fig, gd_fig)):
                        col.pyplot(fig)
                        plt.close(fig)


if __name__ == "__main__":
    main()
