import matplotlib.pyplot as plt
import numpy as np

from qadaptive.utils.plotting.traces import TrainingRunTrace

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
