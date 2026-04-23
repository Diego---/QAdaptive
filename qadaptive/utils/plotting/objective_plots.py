from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from qadaptive.utils.plotting.traces import TrainingRunTrace


def plot_cost_with_outer_boundaries(
    traces: list[TrainingRunTrace],
    figsize: tuple[int, int] = (12, 5),
    annotate_actions: bool = True,
    label_rotation: float = 10,
    title: str | None = "Objective trace with outer-step boundaries",
    title_pad: int = 16,
    title_loc: str = "center",
) -> tuple[plt.Figure, plt.Axes]:
    """
    Plot the objective trajectory across global plotting indices.

    Parameters
    ----------
    traces : list[TrainingRunTrace]
        Sequence of plot-ready training traces.
    figsize : tuple[int, int], optional
        Figure size. Default is ``(12, 5)``.
    annotate_actions : bool, optional
        If True, annotate each shaded run region with its outer-loop action.
        Default is True.
    label_rotation : float, optional
        Rotation angle in degrees for action labels. Default is ``10``.

    Returns
    -------
    tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]
        Created figure and axes.
    """
    if len(traces) == 0:
        raise ValueError("`traces` must contain at least one TrainingRunTrace.")

    fig, ax = plt.subplots(figsize=figsize)

    x_all = []
    y_all = []

    for trace in traces:
        x = np.arange(trace.start, trace.stop, dtype=float)
        x_all.append(x)
        y_all.append(trace.values)

        shade_color = "green" if trace.accepted else "red"
        ax.axvspan(trace.start - 0.5, trace.stop - 0.5, alpha=0.06, color=shade_color)

    x_all = np.concatenate(x_all)
    y_all = np.concatenate(y_all)

    ax.plot(x_all, y_all, ".-")
    ax.set_xlabel("Global inner-loop iteration")
    ax.set_ylabel("Objective")
    if title is not None:
        ax.set_title(title, pad=title_pad, loc=title_loc)

    for trace in traces[:-1]:
        ax.axvline(trace.stop - 0.5, linestyle="--", linewidth=1)

    if annotate_actions:
        finite_values = y_all[np.isfinite(y_all)]
        if finite_values.size > 0:
            y_min = float(np.min(finite_values))
            y_max = float(np.max(finite_values))
        else:
            y_min, y_max = 0.0, 1.0

        y_range = y_max - y_min
        if y_range == 0.0:
            y_range = 1.0

        y_text = y_max + 0.04 * y_range

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
