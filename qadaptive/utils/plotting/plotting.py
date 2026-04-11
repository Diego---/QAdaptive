from __future__ import annotations

from dataclasses import dataclass
from collections import defaultdict
import re

import matplotlib.pyplot as plt
import numpy as np

from qadaptive import MutableAnsatzExperiment

@dataclass
class TrainingRunTrace:
    """
    Container for a single inner-loop training segment within an outer-loop run.

    This structure represents the data produced during one training phase
    (i.e., between two structural modifications of the ansatz). It provides
    a consistent interface for plotting and analysis across multiple outer steps.
    
    Attributes
    ----------
    run_index : int
        Sequential index of the training run (0-based).
    outer_iteration : int
        Index of the outer-loop iteration associated with this training segment.
    action : str
        Description of the structural modification applied before this training
        segment (e.g., "uniform_growth", "prune_sweep").
    accepted : bool
        Whether the structural modification associated with this segment was
        accepted according to the outer-loop acceptance criterion.
    start : int
        Global index (inclusive) of the first inner-loop iteration in this segment.
    stop : int
        Global index (exclusive) of the last inner-loop iteration in this segment.
    param_names : list[str]
        List of parameter names active during this training segment.
        The order matches the columns of `params`.
    params : np.ndarray
        Parameter values recorded during the training segment, with shape
        ``(num_iterations, num_parameters)``.
    values : np.ndarray
        Objective values recorded during the training segment, with shape
        ``(num_iterations,)``.
    stepsizes : np.ndarray
        Step sizes recorded during the training segment, with shape
        ``(num_iterations,)``.
    cost_after : float | None, optional
        Final objective value associated with this segment, typically taken
        from the corresponding outer-step result.
    note : str | None, optional
        Optional note associated with the segment, for example information
        about automatic acceptance or rollback.

    """
    run_index: int
    outer_iteration: int
    action: str
    accepted: bool
    start: int
    stop: int
    param_names: list[str]
    params: np.ndarray
    values: np.ndarray
    stepsizes: np.ndarray
    cost_after: float | None = None
    note: str | None = None


def _dedupe_parameter_memory_history(records):
    """
    Remove the duplicated initial-training record if present.
    """
    cleaned = []
    seen_initial = False

    for record in records:
        is_initial = (
            record.outer_iteration == 0
            and record.action == "Initial training before first plan"
        )

        if is_initial and seen_initial:
            continue

        cleaned.append(record)

        if is_initial:
            seen_initial = True

    return cleaned


def build_run_traces(
    experiment: MutableAnsatzExperiment,
    params_history: list[list[float]] | list[np.ndarray],
    values_history: list[float],
    stepsize_history: list[float],
    train_iterations: int,
) -> list[TrainingRunTrace]:
    """
    Reconstruct per-run traces from the flat callback history.

    Assumptions
    -----------
    - one callback entry per trainer iteration
    - parameter names active during a run are the names stored in the
      corresponding ParameterMemoryRecord at the end of that run
    """
    outer_results = list(experiment.outer_step_history)
    memory_records = _dedupe_parameter_memory_history(
        list(experiment.parameter_memory_history)
    )

    if len(outer_results) != len(memory_records):
        raise ValueError(
            f"Mismatch between outer_step_history ({len(outer_results)}) and "
            f"parameter_memory_history ({len(memory_records)} after deduplication)."
        )

    expected_points = len(outer_results) * train_iterations
    if len(params_history) < expected_points or len(values_history) < expected_points:
        raise ValueError(
            "Callback history is shorter than expected from "
            f"{len(outer_results)} runs x {train_iterations} iterations."
        )

    traces: list[TrainingRunTrace] = []

    for run_index, (outer_result, memory_record) in enumerate(
        zip(outer_results, memory_records)
    ):
        start = run_index * train_iterations
        stop = start + train_iterations

        param_names = list(memory_record.values.parameter_names)
        param_matrix = np.asarray(params_history[start:stop], dtype=float)
        value_vector = np.asarray(values_history[start:stop], dtype=float)
        step_vector = np.asarray(stepsize_history[start:stop], dtype=float)

        if param_matrix.ndim != 2:
            raise ValueError(
                f"Parameter history for run {run_index} is not 2D. "
                f"Got shape {param_matrix.shape}."
            )

        if param_matrix.shape[1] != len(param_names):
            raise ValueError(
                f"Run {run_index}: callback stored {param_matrix.shape[1]} parameters "
                f"but ParameterMemoryRecord has {len(param_names)} names."
            )

        traces.append(
            TrainingRunTrace(
                run_index=run_index,
                outer_iteration=outer_result.iteration,
                action=outer_result.action,
                accepted=outer_result.accepted,
                start=start,
                stop=stop,
                param_names=param_names,
                params=param_matrix,
                values=value_vector,
                stepsizes=step_vector,
                cost_after=outer_result.cost_after,
                note=outer_result.note,
            )
        )

    return traces


def plot_cost_with_outer_boundaries(
    traces: list[TrainingRunTrace],
    figsize: tuple[int, int] = (12, 5),
    annotate_actions: bool = True,
    label_rotation: float = 10,
) -> tuple[plt.Figure, plt.Axes]:
    """
    Plot the objective function trajectory across inner-loop iterations,
    with visual markers for outer-loop boundaries and actions.

    This function concatenates all inner-loop segments and displays the
    evolution of the objective as a function of global iteration index.
    Outer-loop structure is visualized via vertical boundary lines and
    shaded regions corresponding to individual training segments.

    Parameters
    ----------
    traces : list[TrainingRunTrace]
        Sequence of training segments produced by the outer-loop routine.
        Each trace defines a contiguous block in the global iteration axis.
    figsize : tuple[int, int], optional
        Size of the matplotlib figure. Default is (12, 5).
    annotate_actions : bool, optional
        If True, annotate each segment with its corresponding outer-loop
        action label. Default is True.
    label_rotation : float, optional
        Rotation angle (in degrees) for action labels. Useful to avoid
        overlap when labels are long. Default is 10.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The created matplotlib figure.
    ax : matplotlib.axes.Axes
        The axes containing the plot.

    Notes
    -----
    - Vertical dashed lines mark boundaries between consecutive training segments.
    - Shaded regions indicate accepted (green) or rejected (red) outer steps.
    - The x-axis represents the global inner-loop iteration index.
    """
    fig, ax = plt.subplots(figsize=figsize)

    x_all = []
    y_all = []

    for trace in traces:
        x = np.arange(trace.start, trace.stop)
        x_all.append(x)
        y_all.append(trace.values)

        shade_color = "green" if trace.accepted else "red"
        ax.axvspan(trace.start, trace.stop, alpha=0.06, color=shade_color)

    x_all = np.concatenate(x_all)
    y_all = np.concatenate(y_all)

    ax.plot(x_all, y_all, ".-")
    ax.set_xlabel("Global inner-loop iteration")
    ax.set_ylabel("Objective")
    ax.set_title("Objective trace with outer-step boundaries", pad=16)

    for trace in traces[:-1]:
        ax.axvline(trace.stop - 0.5, linestyle="--", linewidth=1)

    if annotate_actions:
        y_min = np.nanmin(y_all)
        y_max = np.nanmax(y_all)
        y_range = y_max - y_min
        y_text = y_max + 0.04 * y_range  # place labels just above data region

        for trace in traces:
            x_mid = 0.5 * (trace.start + trace.stop - 1)
            label = f"{trace.outer_iteration}: {trace.action}"
            ax.text(
                x_mid,
                y_text,
                label,
                rotation=label_rotation,
                ha="left",
                va="bottom",
                fontsize=8,
                alpha=0.9,
                clip_on=False,
            )

        ax.set_ylim(y_min, y_max + 0.14 * y_range)

    return fig, ax


def _parameter_sort_key(name: str):
    match = re.match(r"^θ_(\d+)$", name)
    if match is None:
        return (1, name)
    return (0, int(match.group(1)))


def build_parameter_series(
    traces: list[TrainingRunTrace],
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """
    Construct continuous parameter trajectories across multiple training segments.

    This function aggregates parameter values from a sequence of
    `TrainingRunTrace` objects into time series indexed by global iteration.
    Parameters that are absent in certain segments (due to pruning or late
    insertion) are represented with gaps in their trajectories.

    Parameters
    ----------
    traces : list[TrainingRunTrace]
        Sequence of training segments containing parameter dictionaries
        at each iteration.

    Returns
    -------
    series : dict[str, tuple[np.ndarray, np.ndarray]]
        Mapping from parameter name to a tuple (x, y), where:
        - x : np.ndarray
            Global iteration indices at which the parameter is defined.
        - y : np.ndarray
            Corresponding parameter values.

    Notes
    -----
    - Parameters may appear or disappear across traces due to structural
      modifications of the ansatz.
    - This function does not interpolate missing values; gaps are preserved.
    - Parameter identity is determined by name, typically from a
      `ParameterVector` or parameter memory mapping.
    """
    xs: dict[str, list[float]] = defaultdict(list)
    ys: dict[str, list[float]] = defaultdict(list)

    for trace in traces:
        x = np.arange(trace.start, trace.stop, dtype=float)

        for col, name in enumerate(trace.param_names):
            xs[name].extend(x.tolist())
            ys[name].extend(trace.params[:, col].tolist())

            # break the line after the run so disappear/reappear is visible
            xs[name].append(np.nan)
            ys[name].append(np.nan)

    return {
        name: (np.asarray(xs[name], dtype=float), np.asarray(ys[name], dtype=float))
        for name in sorted(xs, key=_parameter_sort_key)
    }


def plot_parameter_lifelines(
    traces: list[TrainingRunTrace],
    parameters: list[str] | None = None,
    figsize: tuple[int, int] = (12, 6),
    linewidth: float = 1.5,
    legend_outside: bool = True,
) -> tuple[plt.Figure, plt.Axes]:
    """
    Plot the evolution of named parameters across inner-loop iterations.

    Each parameter is displayed as a separate trajectory over the global
    iteration axis. The plot highlights how parameters evolve, persist,
    or disappear as the ansatz is modified during the outer loop.

    Parameters
    ----------
    traces : list[TrainingRunTrace]
        Sequence of training segments containing parameter histories.
    parameters : list[str] or None, optional
        Subset of parameter names to plot. If None, all parameters are shown.
    figsize : tuple[int, int], optional
        Size of the matplotlib figure. Default is (12, 6).
    linewidth : float, optional
        Line width for parameter trajectories. Default is 1.5.
    legend_outside : bool, optional
        If True, place the legend outside the plotting area to reduce clutter.
        Default is True.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The created matplotlib figure.
    ax : matplotlib.axes.Axes
        The axes containing the plot.

    Notes
    -----
    - Vertical dashed lines indicate boundaries between outer-loop segments.
    - Shaded regions represent accepted (green) or rejected (red) structural updates.
    - Parameters that are pruned or newly introduced will show discontinuities.
    - For large parameter counts, the legend may become dense; consider
      filtering `parameters` or grouping them.
    """
    series = build_parameter_series(traces)

    if parameters is None:
        parameters = list(series.keys())

    fig, ax = plt.subplots(figsize=figsize)

    for name in parameters:
        if name not in series:
            continue
        x, y = series[name]
        ax.plot(x, y, linewidth=linewidth, label=name)

    for trace in traces[:-1]:
        ax.axvline(trace.stop - 0.5, linestyle="--", linewidth=1)

    for trace in traces:
        shade_color = "green" if trace.accepted else "red"
        ax.axvspan(trace.start, trace.stop, alpha=0.04, color=shade_color)

    ax.set_xlabel("Global inner-loop iteration")
    ax.set_ylabel("Parameter value")
    ax.set_title("Named parameter trajectories")

    if legend_outside:
        ax.legend(
            loc="upper left",
            bbox_to_anchor=(1.02, 1.0),
            borderaxespad=0.0,
            fontsize=8,
            ncol=1,
        )
        fig.subplots_adjust(right=0.72)
    else:
        ax.legend(ncol=2, fontsize=8)

    return fig, ax


def plot_parameter_heatmap(
    traces: list[TrainingRunTrace],
    figsize: tuple[int, int] = (12, 6),
) -> tuple[plt.Figure, plt.Axes]:
    """
    Visualize parameter evolution as a heatmap over global iterations.

    This plot provides a compact representation of parameter dynamics,
    where each row corresponds to a parameter and each column to an
    inner-loop iteration. Color encodes the parameter value.

    Parameters
    ----------
    traces : list[TrainingRunTrace]
        Sequence of training segments containing parameter histories.
    parameters : list[str] or None, optional
        Subset of parameter names to include. If None, all parameters are used.
    figsize : tuple[int, int], optional
        Size of the matplotlib figure. Default is (12, 6).
    cmap : str, optional
        Matplotlib colormap used to encode parameter values.
        Default is "viridis".
    normalize : bool, optional
        If True, normalize each parameter independently (e.g., to zero mean
        and unit variance) before plotting. Default is False.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The created matplotlib figure.
    ax : matplotlib.axes.Axes
        The axes containing the heatmap.

    Notes
    -----
    - Missing parameter values (due to pruning or late insertion) are
      typically represented as NaNs and may appear as blank or masked regions.
    - This visualization is particularly useful for identifying:
        * inactive parameters,
        * sudden changes due to structural updates,
        * correlated parameter behavior.
    - For large parameter sets, consider sorting or clustering parameters
      to improve interpretability.
    """
    all_names = sorted(
        {name for trace in traces for name in trace.param_names},
        key=_parameter_sort_key,
    )

    total_steps = traces[-1].stop
    matrix = np.full((len(all_names), total_steps), np.nan, dtype=float)
    name_to_row = {name: i for i, name in enumerate(all_names)}

    for trace in traces:
        for col, name in enumerate(trace.param_names):
            row = name_to_row[name]
            matrix[row, trace.start:trace.stop] = trace.params[:, col]

    fig, ax = plt.subplots(figsize=figsize)
    masked = np.ma.masked_invalid(matrix)
    im = ax.imshow(masked, aspect="auto", interpolation="none")

    for trace in traces[:-1]:
        ax.axvline(trace.stop - 0.5, linestyle="--", linewidth=1, color="white")

    ax.set_xlabel("Global inner-loop iteration")
    ax.set_ylabel("Parameter")
    ax.set_yticks(np.arange(len(all_names)))
    ax.set_yticklabels(all_names)
    ax.set_title("Parameter activity heatmap")
    fig.colorbar(im, ax=ax, label="Parameter value")

    return fig, ax
