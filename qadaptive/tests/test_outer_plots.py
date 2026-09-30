from dataclasses import asdict

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from qadaptive.outer.outer_loop import OuterStepResult
from qadaptive.utils.plotting.outer_plots import plot_outer_history


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
