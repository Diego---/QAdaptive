
import re
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np

from qadaptive.utils.plotting.traces import TrainingRunTrace




def _parameter_sort_key(name: str):
    match = re.match(r"^θ_(\d+)$", name)
    if match is None:
        return (1, name)
    return (0, int(match.group(1)))

def _normalize_parameter_matrix_rows(matrix: np.ndarray) -> np.ndarray:
    """
    Return a row-wise normalized copy of a parameter matrix.

    Each row is normalized independently using only its finite entries.
    Rows with zero variance are centered but not rescaled.

    Parameters
    ----------
    matrix : np.ndarray
        Matrix of shape ``(num_parameters, num_steps)`` containing parameter
        values, possibly with NaNs for inactive regions.

    Returns
    -------
    np.ndarray
        Row-wise normalized matrix with NaNs preserved.
    """
    normalized = matrix.copy()

    for row in range(normalized.shape[0]):
        valid = np.isfinite(normalized[row])
        if not np.any(valid):
            continue

        values = normalized[row, valid]
        mean = np.mean(values)
        std = np.std(values)

        normalized[row, valid] = values - mean
        if std > 0:
            normalized[row, valid] /= std

    return normalized


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
    parameters: list[str] | None = None,
    figsize: tuple[int, int] = (12, 6),
    cmap: str = "viridis",
    normalize: bool = False,
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
    
    if parameters is None:
        selected_names = all_names
    else:
        selected_set = set(parameters)
        selected_names = [name for name in all_names if name in selected_set]

    if not selected_names:
        raise ValueError("No parameter names were selected for the heatmap.")

    total_steps = traces[-1].stop
    matrix = np.full((len(all_names), total_steps), np.nan, dtype=float)
    name_to_row = {name: i for i, name in enumerate(all_names)}

    for trace in traces:
        for col, name in enumerate(trace.param_names):
            if name not in name_to_row:
                continue
            row = name_to_row[name]
            matrix[row, trace.start:trace.stop] = trace.params[:, col]

    if normalize:
        matrix = _normalize_parameter_matrix_rows(matrix)

    fig, ax = plt.subplots(figsize=figsize)
    masked = np.ma.masked_invalid(matrix)
    im = ax.imshow(
        masked,
        aspect="auto",
        interpolation="none",
        cmap=cmap,
    )

    for trace in traces[:-1]:
        ax.axvline(trace.stop - 0.5, linestyle="--", linewidth=1, color="white")

    ax.set_xlabel("Global inner-loop iteration")
    ax.set_ylabel("Parameter")
    ax.set_yticks(np.arange(len(selected_names)))
    ax.set_yticklabels(selected_names)
    ax.set_title("Parameter activity heatmap")
    fig.colorbar(im, ax=ax, label="Normalized value" if normalize else "Parameter value")

    return fig, ax
