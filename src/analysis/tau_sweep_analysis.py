# Make other project modules accessible
import sys
import os
sys.path.insert(1, os.path.join(os.path.dirname(__file__), ".."))

# Package imports
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# Python import
import pickle
from pathlib import Path
import multiprocessing as mp

# Pymoo imports
from pymoo.indicators.hv import HV
from pymoo.indicators.igd_plus import IGDPlus

def load_sub_directories(dir: str, files: bool = False):
    """Loads subdirectories of a given directory for data loading.

    Args:
        dir(str): Path to the directory you want to list.
        files(bool): Decide if files should be loaded.

    Returns:
        list: List of names of subdirs or files.
    """

    assert os.path.exists(dir), f"Path supplied to load subdirectories did not exist."

    if not files:
        return [subdir for subdir in os.listdir(dir) if os.path.isdir(os.path.join(dir, subdir))]
    else:
        return [subdir for subdir in os.listdir(dir) if os.path.isfile(os.path.join(dir, subdir))]

def _sorted(names):
    """Numeric sort when every name parses as a float (std, threshold), else lexical."""
    try:
        return sorted(names, key=float)
    except ValueError:
        return sorted(names)

def descend(base_path: str, selections: list) -> list:
    """Walk base_path following `selections`, one entry per already-resolved level.

    Args:
        base_path(str): Directory the hierarchy starts at.
        selections(list): List of per-level name lists. An empty list at a level
            means "keep every subdirectory at that level".

    Returns:
        list: Existing paths reachable under the given selections.
    """
    paths = [base_path]
    for choices in selections:
        nxt = []
        for parent in paths:
            for name in load_sub_directories(parent):
                if not choices or name in choices:
                    nxt.append(os.path.join(parent, name))
        paths = nxt
    st.write(f"Paths: {paths}")
    return paths

def add_multiselect(base_path: str, selected_options: list, multi_select_options: str) -> list:
    """Render one cascading multiselect for the next level.

    Args:
        base_path(str): Root directory the hierarchy starts at.
        selected_options(list): Selection lists for the levels already rendered.
        multi_select_options(str): Label for this level.

    Returns:
        list: `selected_options` with this level's picks appended.
    """
    options = set()
    for parent in descend(base_path, selected_options):
        options.update(load_sub_directories(parent))

    picked = st.multiselect(multi_select_options, _sorted(options), key=multi_select_options)
    return selected_options + [picked]

@st.cache_data
def load_run_data(run_path: str):
    """Loads and formats data of a given run.

    Args:
        run_path (str): Path to which you want to load results from."""

    # Synthesize file paths
    file_paths = [os.path.join(run_path, file_path) for file_path in load_sub_directories(dir = run_path, files = True)]
    #st.write(file_paths)
    # Load dataframes from files
    data_frames = []
    for path in file_paths:
        with open(path, "rb") as f:
            data_frames.append(pickle.load(file = f))

    # Each list in this dict holds 31 sublists, corresponding to the 31 runs

    combined_dict = {
        "clean_fronts": [],
        "re_evaluations": [],
        "evaluations_per_gen": [],
        "igd_plus": [],
        "gd_plus": [],
    }

    for df in data_frames:
        combined_dict["clean_fronts"].append(df["clean_fronts"])
        combined_dict["re_evaluations"].append(df["re_evaluations"])
        combined_dict["evaluations_per_gen"].append(df["evaluations_per_gen"])
        combined_dict["igd_plus"].append(df["igd_plus"])
        combined_dict["gd_plus"].append(df["gd_plus"])

    return combined_dict

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

    # TODO: Does not work as it does now for combined mean plot --> Buckets on n_eval or interpolation or 2 plots, 1 HV per generation, 2 function evals per generation
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

def _final_values(per_run_metric: list) -> np.ndarray:
    """Final-generation value of a per-generation metric, one per run.

    Args:
        per_run_metric (list): One entry per run; each entry a per-generation
            sequence (list, ndarray or Series) of the metric.

    Returns:
        np.ndarray: Final-generation value for every run that has data.
    """
    finals = []
    for run in per_run_metric:
        arr = np.asarray(run, dtype=float).ravel()
        if arr.size:
            finals.append(arr[-1])
    return np.asarray(finals, dtype=float)

def _stats(vals: np.ndarray) -> dict:
    """Mean / spread of a 1D array, ignoring non-finite entries."""
    vals = vals[np.isfinite(vals)]
    n = int(vals.size)
    if n == 0:
        return {"n": 0, "mean": np.nan, "std": np.nan, "sem": np.nan}
    std = float(vals.std(ddof=1)) if n > 1 else 0.0
    return {"n": n, "mean": float(vals.mean()), "std": std,
            "sem": std / np.sqrt(n) if n > 1 else 0.0}

def _summarize_config(data_dict: dict) -> dict:
    """Aggregate final-generation IGD+ and GD+ across all runs of one config."""
    igd = _stats(_final_values(data_dict["igd_plus"]))
    gd = _stats(_final_values(data_dict["gd_plus"]))
    return {
        "n_runs": igd["n"],
        "igd_plus_mean": igd["mean"], "igd_plus_std": igd["std"], "igd_plus_sem": igd["sem"],
        "gd_plus_mean": gd["mean"], "gd_plus_std": gd["std"], "gd_plus_sem": gd["sem"],
    }

# Metric columns that feed the combined ranking, with their "lower is better" flag.
_RANK_SPEC = (("igd_plus_mean", True), ("gd_plus_mean", True), ("hv_final_mean", False))

def _combined_rank(table: pd.DataFrame, weights=(1.0, 1.0, 1.0)) -> pd.DataFrame:
    """Rank configs on IGD+, GD+ and hypervolume jointly.

    Each metric is ranked across the configs of one group (1 = best): IGD+ and
    GD+ ascending, hypervolume descending. ``rank_score`` is the weighted mean of
    those three ranks and is what "best" is decided on -- lower is better. A
    config missing a metric takes the worst rank for it (``na_option="bottom"``).

    Args:
        table (pd.DataFrame): Per-config summary, indexed by config.
        weights: Relative weights for (IGD+, GD+, HV). Defaults to equal.

    Returns:
        pd.DataFrame: Copy of `table` with ``igd_plus_rank``, ``gd_plus_rank``,
            ``hv_final_rank`` and ``rank_score`` columns added.
    """
    out = table.copy()
    total_w = float(sum(weights))
    score = pd.Series(0.0, index=out.index)
    for (col, ascending), w in zip(_RANK_SPEC, weights):
        rank_col = col.replace("_mean", "_rank")
        out[rank_col] = out[col].rank(ascending=ascending, na_option="bottom")
        score = score + w * out[rank_col]
    out["rank_score"] = score / total_w
    return out

def get_best_runs(path_to_exp: str, rank_by: str = "combined", weights=(1.0, 1.0, 1.0)) -> dict:
    """Rank sweep configurations per (problem, noise setting) and report the best.

    For every problem / noise-std combination under `path_to_exp`, each config's
    runs are reduced to their final-generation IGD+, GD+ and hypervolume,
    averaged over the seeds. By default "best" is decided jointly on all three
    metrics via `_combined_rank` (weighted mean of per-metric ranks); pass a
    single summary column name to `rank_by` to rank on that column alone.
    Hypervolume uses a reference point shared by every config in the group so
    the values are comparable.

    Args:
        path_to_exp (str): Path to an algorithm-level collected directory
            (e.g. ``results/mocba_sweep/mocba_nsga2``).
        rank_by (str): ``"combined"`` (default) to rank on IGD+, GD+ and HV
            together, or a summary column name to rank on just that column
            (lower is better, except ``hv_final_*`` where higher is better).
        weights: Relative (IGD+, GD+, HV) weights for the combined ranking.
            Ignored unless ``rank_by == "combined"``.

    Returns:
        dict: Maps ``(problem, std_label)`` to
            ``{"best_config": str, "table": pd.DataFrame}`` where the table is
            the per-config summary sorted best-first.
    """

    # The path should lead directly to the collected dir for a certain setting
    # (preferably down to the algorithm level).
    real_path = Path(path_to_exp)
    algorithm_name = real_path.parts[-1]
    st.header(algorithm_name)

    results: dict = {}

    for problem in _sorted(load_sub_directories(real_path)):
        problem_path = os.path.join(real_path, problem)
        st.subheader(problem)

        noise_type_path = os.path.join(problem_path, "gaussian")
        if not os.path.isdir(noise_type_path):
            st.write("No 'gaussian' noise directory found.")
            continue

        for std_label in _sorted(load_sub_directories(noise_type_path)):
            noise_setting = os.path.join(noise_type_path, std_label)
            st.write(f"Gaussian {std_label}")

            # Load every config once, then fix a shared HV reference point over
            # all their runs so hypervolumes are comparable within this group.
            data_dicts = {config: load_run_data(os.path.join(noise_setting, config))
                          for config in _sorted(load_sub_directories(noise_setting))}
            if not data_dicts:
                st.write("No config directories found.")
                continue

            group_fronts = [run for d in data_dicts.values() for run in d["clean_fronts"]]
            hv_ref = hv_reference_point(group_fronts)

            # One summary row per config directory (tau_*/k_*/budget_*).
            rows = []
            for config, data_dict in data_dicts.items():
                summary = _summarize_config(data_dict)
                _, run_hvs, _ = hv_trajectories(
                    data_dict["clean_fronts"], data_dict["evaluations_per_gen"],
                    ref_point=hv_ref, last_gen_only=True,
                )
                hv = _stats(_final_values(run_hvs))
                summary.update(hv_final_mean=hv["mean"], hv_final_std=hv["std"], hv_final_sem=hv["sem"])
                summary["config"] = config
                rows.append(summary)

            table = pd.DataFrame(rows).set_index("config")

            metric_cols = [c for c, _ in _RANK_SPEC]
            if rank_by == "combined":
                if not set(metric_cols).issubset(table.columns) or table[metric_cols].isna().to_numpy().all():
                    st.write("Not enough IGD+/GD+/HV data to rank; showing unranked table.")
                    st.dataframe(table)
                    continue
                table = _combined_rank(table, weights).sort_values("rank_score")
            elif rank_by not in table.columns or table[rank_by].isna().all():  # pyright: ignore[reportGeneralTypeIssues]
                st.write(f"No usable '{rank_by}' values; skipping ranking.")
                st.dataframe(table)
                continue
            else:
                # Lower is better for IGD+/GD+, higher is better for hypervolume.
                table = table.sort_values(rank_by, ascending=not str(rank_by).startswith("hv"))

            best_config = str(table.index[0])
            best = table.iloc[0]

            # Rows are sorted best-first; the winning config is called out below.
            st.dataframe(table.round(6))
            st.caption(f"HV reference point: {np.round(hv_ref, 4).tolist()}")
            score_txt = f", combined rank {best['rank_score']:.2f}" if "rank_score" in table.columns else ""
            st.write(
                f"**Best: `{best_config}`** — "
                f"IGD+ {best['igd_plus_mean']:.4g} ± {best['igd_plus_sem']:.2g} (sem), "
                f"GD+ {best['gd_plus_mean']:.4g}, "
                f"HV {best['hv_final_mean']:.4g} ± {best['hv_final_sem']:.2g}, "
                f"n={int(best['n_runs'])}{score_txt}"
            )

            results[(problem, std_label)] = {"best_config": best_config, "table": table}

    # Tally which config wins most often across this algorithm's settings.
    if results:
        tally = pd.Series([v["best_config"] for v in results.values()]).value_counts()
        st.subheader(f"{algorithm_name}: best-config tally")
        st.bar_chart(tally)

    return results

def main():
    results_paths = ["/home/malte/Documents/Work/Uncertainty-Framework-WIP/results/tau_sweep/cbr_nsga2",
        Path("/home/malte/Documents/Work/Uncertainty-Framework-WIP/results/mocba_sweep/mocba_nsga2"),
        Path("/home/malte/Documents/Work/Uncertainty-Framework-WIP/results/static_sweep/sr_nsga2")]
    # LEVELS = ["problem", "noise_type", "std", "threshold"]

    # st.title("CBR-NSGA-II Threshold Sweep Results")
    # selections = []
    # for level in LEVELS:
    #     selections = add_multiselect(results_path, selections, level)

    # selected_paths = descend(results_path, selections)
    # st.caption(f"{len(selected_paths)} leaf directories selected")
    # st.write(selected_paths)

    # figures = []

    # # Very intrecate loop for creating the plot titles, plots and some metrics
    # for path in selected_paths:
    #     parts = Path(path).relative_to(results_path).parts
    #     label = " · ".join(f"{lvl}={val}" for lvl, val in zip(LEVELS, parts))
    #     section_header = f"Problem: {parts[0]}, Noise: {parts[2].removeprefix("std_")}, Threshold: {parts[3].removeprefix("tau_")}"
    #     dataframe =load_run_data(path)
    #     hv_plot, _, _, _ = create_hv_plot(compiled_data = dataframe["clean_fronts"], compiled_n_eval = dataframe["evaluations_per_gen"], title = label)
    #     igd_plus_plot, _, _, _ = create_IGD_plus_plot(compiled_data = dataframe["igd_plus"], compiled_n_eval = dataframe["evaluations_per_gen"], title = label)
    #     gd_plus_plot, _, _, _ = create_GD_plus_plot(compiled_data = dataframe["gd_plus"], compiled_n_eval = dataframe["evaluations_per_gen"], title = label)
    #     figures.append((section_header, (hv_plot, gd_plus_plot, igd_plus_plot)))

    # # Display figures with header
    # for comb in figures:
    #     st.header(comb[0])
    #     for fig in comb[1]:
    #         st.pyplot(fig)

    with mp.Pool(processes = len(results_paths)) as p:
        p.map(get_best_runs, results_paths)


main()
