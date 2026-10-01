from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict
from typing import Any, TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator

if TYPE_CHECKING:
    from qadaptive.outer.outer_loop import OuterStepResult


def _float_or_nan(value: float | None) -> float:
    return np.nan if value is None else float(value)


def plot_outer_history(
    outer_step_history: Sequence[OuterStepResult | Mapping[str, Any]],
    *,
    figsize: tuple[float, float] = (10, 4),
    ylabel: str = "Cost",
    show_rejected: bool = True,
    title: str | None = "Outer-loop history",
) -> tuple[plt.Figure, plt.Axes]:
    """
    Plot retained costs and rejected trial costs at their outer iterations.

    Initial training is included when present in the history.
    Missing costs appear as gaps.

    Parameters
    ----------
    outer_step_history : Sequence[OuterStepResult | Mapping[str, Any]]
        Sequence of outer-step execution records or dictionaries.
    figsize : tuple[float, float], optional
        Figure size. Default is ``(10, 4)``.
    ylabel : str, optional
        Label for the y-axis. Default is ``"Cost"``.
    show_rejected : bool, optional
        Whether to overlay rejected trial proposals. Default is True.
    title : str | None, optional
        Title for the plot. If None, no title is rendered.
        Default is ``"Outer-loop history"``.

    Returns
    -------
    tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]
        Created figure and axes objects.
    """
    if len(outer_step_history) == 0:
        raise ValueError("outer_step_history must contain at least one record.")

    records = [
        record if isinstance(record, Mapping) else asdict(record)
        for record in outer_step_history
    ]

    iterations = np.asarray(
        [record["iteration"] for record in records], dtype=int
    )
    accepted = np.asarray(
        [record["accepted"] for record in records], dtype=bool
    )
    before = np.asarray([
        _float_or_nan(record["cost_before"]) for record in records
    ])
    after = np.asarray([
        _float_or_nan(record["cost_after"]) for record in records
    ])
    retained = np.where(accepted, after, before)

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(
        iterations,
        retained,
        marker="o",
        color="tab:blue",
        label="Accepted state",
    )

    rejected = ~accepted & np.isfinite(after)
    if show_rejected and np.any(rejected):
        ax.scatter(
            iterations[rejected],
            after[rejected],
            marker="x",
            color="tab:red",
            label="Rejected proposal",
            zorder=3,
        )

    ax.set_xlabel("Outer iteration")
    ax.set_ylabel(ylabel)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    if title is not None:
        ax.set_title(title)
    ax.grid(alpha=0.2)
    ax.legend()
    fig.tight_layout()

    return fig, ax

def plot_complexity_evolution(
    outer_step_history: Sequence[OuterStepResult | Mapping[str, Any]],
    *,
    figsize: tuple[float, float] = (10, 6),
    show_rejected: bool = True,
    title: str | None = "Circuit complexity evolution",
) -> tuple[plt.Figure, np.ndarray]:
    """
    Plot retained parameter and two-qubit-instruction counts.

    Rejected proposals use their before counts in the retained trajectory;
    their trial counts may be shown separately.

    Parameters
    ----------
    outer_step_history : Sequence[OuterStepResult | Mapping[str, Any]]
        Sequence of outer-step execution records or mapping objects.
    figsize : tuple[float, float], optional
        Figure size as ``(width, height)``. Default is ``(10, 6)``.
    show_rejected : bool, optional
        Whether to display trial complexity values of rejected proposals as red 'x'
        markers. Default is True.
    title : str | None, optional
        Overall title for the figure. If None, no title is displayed.
        Default is ``"Circuit complexity evolution"``.

    Returns
    -------
    tuple[matplotlib.figure.Figure, numpy.ndarray]
        Created figure and a 1D NumPy array containing the two subplot axes 
        (top for parameters, bottom for two-qubit instructions).

    Raises
    ------
    ValueError
        If ``outer_step_history`` is empty.
    """
    if len(outer_step_history) == 0:
        raise ValueError("outer_step_history must contain at least one record.")

    records = [
        record if isinstance(record, Mapping) else asdict(record)
        for record in outer_step_history
    ]
    iterations = np.asarray(
        [record["iteration"] for record in records], dtype=int
    )
    accepted = np.asarray(
        [record["accepted"] for record in records], dtype=bool
    )
    rejected = ~accepted

    fig, axes = plt.subplots(2, 1, figsize=figsize, sharex=True)

    metrics = (
        ("num_parameters", "Parameters"),
        ("num_two_qubit", "Two-qubit instructions"),
    )
    for ax, (field, ylabel) in zip(axes, metrics):
        before = np.asarray(
            [record[f"{field}_before"] for record in records],
            dtype=int,
        )
        after = np.asarray(
            [record[f"{field}_after"] for record in records],
            dtype=int,
        )
        retained = np.where(accepted, after, before)

        ax.step(
            iterations,
            retained,
            where="post",
            marker="o",
            color="tab:blue",
            label="Accepted state",
        )
        if show_rejected and np.any(rejected):
            ax.scatter(
                iterations[rejected],
                after[rejected],
                marker="x",
                color="tab:red",
                label="Rejected proposal",
                zorder=3,
            )

        ax.set_ylabel(ylabel)
        ax.set_ylim(bottom=0)
        if ax.get_ylim()[1] < 1:
            ax.set_ylim(top=1)
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        ax.grid(alpha=0.2)

    axes[0].legend()
    axes[-1].set_xlabel("Outer iteration")
    axes[-1].xaxis.set_major_locator(MaxNLocator(integer=True))
    if title is not None:
        fig.suptitle(title)
    fig.tight_layout()

    return fig, axes
