from __future__ import annotations

import re
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np

from qadaptive.utils.plotting.traces import TrainingRunTrace


def _parameter_sort_key(name: str) -> tuple[int, int | str]:
    """
    Return a sort key that keeps parameters named like ``θ_i`` in numeric order.

    Parameters
    ----------
    name : str
        Parameter name.

    Returns
    -------
    tuple[int, int | str]
        Sort key for consistent parameter ordering.
    """
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
    Construct continuous parameter trajectories across multiple training runs.

    Parameters
    ----------
    traces : list[TrainingRunTrace]
        Sequence of plot-ready training traces.

    Returns
    -------
    dict[str, tuple[np.ndarray, np.ndarray]]
        Mapping from parameter name to a tuple ``(x, y)``, where `x` contains
        global plotting indices and `y` contains parameter values.

    Notes
    -----
    A `NaN` separator is inserted after each run so that parameters which
    disappear and later reappear are shown as disconnected line segments.
    """
    xs: dict[str, list[float]] = defaultdict(list)
    ys: dict[str, list[float]] = defaultdict(list)

    for trace in traces:
        x = np.arange(trace.start, trace.stop, dtype=float)

        for col, name in enumerate(trace.param_names):
            xs[name].extend(x.tolist())
            ys[name].extend(trace.params[:, col].tolist())

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
    show_legend: bool = True,
) -> tuple[plt.Figure, plt.Axes]:
    """
    Plot the evolution of named parameters across global plotting indices.

    Parameters
    ----------
    traces : list[TrainingRunTrace]
        Sequence of plot-ready training traces.
    parameters : list[str] or None, optional
        Subset of parameter names to plot. If None, all parameters are shown.
    figsize : tuple[int, int], optional
        Figure size. Default is ``(12, 6)``.
    linewidth : float, optional
        Line width for parameter trajectories. Default is ``1.5``.
    legend_outside : bool, optional
        If True, place the legend outside the plotting area. Default is True.
    show_legend: bool, optional
        Whether to show legend.

    Returns
    -------
    tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]
        Created figure and axes.
    """
    if len(traces) == 0:
        raise ValueError("`traces` must contain at least one TrainingRunTrace.")

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
        ax.axvspan(trace.start - 0.5, trace.stop - 0.5, alpha=0.04, color=shade_color)

    ax.set_xlabel("Global inner-loop iteration")
    ax.set_ylabel("Parameter value")
    ax.set_title("Named parameter trajectories")

    if show_legend:
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
    Visualize parameter evolution as a heatmap over global plotting indices.

    Parameters
    ----------
    traces : list[TrainingRunTrace]
        Sequence of plot-ready training traces.
    parameters : list[str] or None, optional
        Subset of parameter names to include. If None, all parameters are used.
    figsize : tuple[int, int], optional
        Figure size. Default is ``(12, 6)``.
    cmap : str, optional
        Matplotlib colormap used for the heatmap. Default is ``"viridis"``.
    normalize : bool, optional
        If True, normalize each parameter row independently before plotting.
        Default is False.

    Returns
    -------
    tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]
        Created figure and axes.
    """
    if len(traces) == 0:
        raise ValueError("`traces` must contain at least one TrainingRunTrace.")

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
    matrix = np.full((len(selected_names), total_steps), np.nan, dtype=float)
    name_to_row = {name: i for i, name in enumerate(selected_names)}

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
