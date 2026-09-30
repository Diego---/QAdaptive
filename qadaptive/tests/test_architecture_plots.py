import matplotlib
import pytest

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter

from qadaptive.outer.outer_loop import AcceptedAnsatzRecord
from qadaptive.utils.plotting.architecture_plots import (
    plot_architecture_evolution,
)


def _snapshot(iteration, circuit):
    return AcceptedAnsatzRecord(
        outer_iteration=iteration,
        action=f"step {iteration}",
        cost=None,
        num_parameters=circuit.num_parameters,
        num_two_qubit_gates=sum(
            len(instruction.qubits) == 2
            for instruction in circuit.data
        ),
        ansatz=circuit,
        parameter_values={
            parameter.name: 0.2
            for parameter in circuit.parameters
        },
    )


@pytest.mark.parametrize(
    "indices, expected_iterations",
    [(None, [0, 7]), ([1], [7])],
)
def test_architecture_plot_preserves_snapshots(
    indices, expected_iterations
):
    initial = QuantumCircuit(2)
    initial.rx(Parameter("theta"), 0)

    grown = initial.copy()
    grown.cx(0, 1)

    records = [_snapshot(0, initial), _snapshot(7, grown)]
    originals = [record.ansatz.copy() for record in records]

    fig, axes = plot_architecture_evolution(
        records, indices=indices
    )
    try:
        assert axes.shape == (len(expected_iterations),)
        assert [ax.get_title(loc="left") for ax in axes] == [
            f"Outer iteration {iteration}: step {iteration}"
            for iteration in expected_iterations
        ]

        fig.canvas.draw()

        for record, original in zip(records, originals):
            assert record.ansatz == original
    finally:
        plt.close(fig)


@pytest.mark.parametrize(
    "history, indices, message",
    [
        ([], None, "No accepted ansatz"),
        ([_snapshot(0, QuantumCircuit(1))], [], "at least one"),
    ],
)
def test_architecture_plot_requires_snapshots(
    history, indices, message
):
    with pytest.raises(ValueError, match=message):
        plot_architecture_evolution(history, indices=indices)
