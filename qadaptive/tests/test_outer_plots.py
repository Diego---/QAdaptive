from dataclasses import asdict

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from qadaptive.outer.outer_loop import OuterStepResult
from qadaptive.utils.plotting.outer_plots import plot_outer_history, plot_complexity_evolution


def _step(iteration, accepted, before, after):
    return OuterStepResult(
        iteration=iteration,
        action="test",
        accepted=accepted,
        cost_before=before,
        cost_after=after,
        delta_cost=None if before is None else after - before,
        num_parameters_before=1,
        num_parameters_after=1,
        num_two_qubit_before=0,
        num_two_qubit_after=0,
    )
    
def _complexity_step(
    iteration, accepted, p_before, p_after, q_before, q_after
):
    return OuterStepResult(
        iteration=iteration,
        action="test",
        accepted=accepted,
        cost_before=None,
        cost_after=None,
        delta_cost=None,
        num_parameters_before=p_before,
        num_parameters_after=p_after,
        num_two_qubit_before=q_before,
        num_two_qubit_after=q_after,
    )


@pytest.mark.parametrize("saved_records", [False, True])
@pytest.mark.parametrize("show_rejected", [False, True])
def test_complexity_plot_preserves_retained_counts(
    saved_records, show_rejected
):
    records = [
        _complexity_step(0, True, 1, 1, 0, 0),
        _complexity_step(2, True, 1, 3, 0, 2),
        _complexity_step(5, False, 3, 9, 2, 5),
        _complexity_step(7, True, 3, 2, 2, 1),
        _complexity_step(9, False, 2, 1, 1, 0),
    ]
    history = (
        [asdict(record) for record in records]
        if saved_records
        else records
    )
    expected = [
        ([1, 3, 3, 2, 2], [[5, 9], [9, 1]], "Parameters"),
        (
            [0, 2, 2, 1, 1],
            [[5, 5], [9, 0]],
            "Two-qubit instructions",
        ),
    ]

    fig, axes = plot_complexity_evolution(
        history, show_rejected=show_rejected
    )
    try:
        for ax, (counts, trials, ylabel) in zip(axes, expected):
            np.testing.assert_array_equal(
                ax.lines[0].get_xdata(), [0, 2, 5, 7, 9]
            )
            np.testing.assert_array_equal(
                ax.lines[0].get_ydata(), counts
            )

            if show_rejected:
                np.testing.assert_array_equal(
                    ax.collections[0].get_offsets(), trials
                )
            else:
                assert len(ax.collections) == 0

            assert ax.get_ylabel() == ylabel

        fig.canvas.draw()
    finally:
        plt.close(fig)


def test_complexity_plot_requires_history():
    with pytest.raises(ValueError, match="at least one record"):
        plot_complexity_evolution([])


@pytest.mark.parametrize("saved_records", [False, True])
def test_outer_plot_preserves_retained_costs(saved_records):
    records = [
        _step(0, True, None, 0.4),
        _step(2, True, 0.4, 0.3),
        _step(5, False, 0.3, 0.1),
        _step(7, True, 0.3, 0.35),
    ]
    history = (
        [asdict(record) for record in records]
        if saved_records
        else records
    )

    fig, ax = plot_outer_history(history, ylabel="Energy")
    try:
        retained = next(
            line for line in ax.lines
            if line.get_label() == "Accepted state"
        )
        np.testing.assert_array_equal(
            retained.get_xdata(), [0, 2, 5, 7]
        )
        np.testing.assert_allclose(
            retained.get_ydata(), [0.4, 0.3, 0.3, 0.35]
        )

        rejected = next(
            artist for artist in ax.collections
            if artist.get_label() == "Rejected proposal"
        )
        np.testing.assert_allclose(
            rejected.get_offsets(), [[5, 0.1]]
        )
        assert ax.get_ylabel() == "Energy"
    finally:
        plt.close(fig)


def test_outer_plot_requires_history():
    with pytest.raises(ValueError, match="at least one record"):
        plot_outer_history([])
