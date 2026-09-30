from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any, TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np

from qiskit.circuit import QuantumCircuit

if TYPE_CHECKING:
    from qadaptive.outer.outer_loop import AcceptedAnsatzRecord


def _field(record: Any, name: str) -> Any:
    return record[name] if isinstance(record, Mapping) else getattr(record, name)


def plot_architecture_evolution(
    accepted_ansatz_history: Sequence[AcceptedAnsatzRecord | Mapping[str, Any]],
    *,
    indices: Sequence[int] | None = None,
    load_circuit: Callable[[int], QuantumCircuit] | None = None,
    figsize: tuple[float, float] | None = None,
) -> tuple[plt.Figure, np.ndarray]:
    """
    Draw selected accepted ansatz snapshots, keeping symbolic parameters.

    Parameters
    ----------
    accepted_ansatz_history : Sequence[AcceptedAnsatzRecord | Mapping[str, Any]]
        Sequence of accepted ansatz records containing circuit structures and 
        outer-iteration metadata.
    indices : Sequence[int] | None, optional
        Positions in ``accepted_ansatz_history`` to plot (not outer-loop iteration numbers).
        If None, all available accepted snapshots are plotted. Default is None.
    load_circuit : Callable[[int], QuantumCircuit] | None, optional
        Optional callback function accepting a snapshot index and returning a loaded
        ``QuantumCircuit`` (e.g., retrieved from QPY storage). If None, the circuit is
        extracted directly from the record. Default is None.
    figsize : tuple[float, float] | None, optional
        Overall figure size as ``(width, height)``. If None, height is calculated
        dynamically based on the number of qubits and classical bits across the plotted
        circuits. Default is None.

    Returns
    -------
    tuple[matplotlib.figure.Figure, numpy.ndarray]
        Created figure and a 1D NumPy array of Matplotlib axes, one per plotted snapshot.

    Raises
    ------
    ValueError
        If ``accepted_ansatz_history`` is empty or if ``indices`` resolves to an empty sequence.
    """
    if len(accepted_ansatz_history) == 0:
        raise ValueError("No accepted ansatz snapshots are available.")

    selected = (
        list(range(len(accepted_ansatz_history)))
        if indices is None
        else list(indices)
    )
    if not selected:
        raise ValueError("Select at least one accepted ansatz snapshot.")

    records = [accepted_ansatz_history[index] for index in selected]
    circuits = [
        load_circuit(index)
        if load_circuit is not None
        else _field(record, "ansatz")
        for index, record in zip(selected, records)
    ]

    # Calculate height per circuit based on qubit count
    heights = [
        max(2.0, 0.8 * (circuit.num_qubits + circuit.num_clbits))
        for circuit in circuits
    ]

    if figsize is None:
        figsize = (14, sum(heights))

    fig, axes = plt.subplots(
        len(records),
        1,
        figsize=figsize,
        squeeze=False,
        gridspec_kw={"height_ratios": heights},
    )
    axes = axes[:, 0]

    # Find maximum depth/width among all circuits to set a unified x-limit scale
    max_depth = max(circuit.depth() for circuit in circuits)

    try:
        for record, circuit, ax in zip(records, circuits, axes):
            # Draw circuit with fixed scale
            circuit.draw(
                output="mpl",
                ax=ax,
                fold=-1,
                idle_wires=True,
                scale=0.7,  # <--- Prevents over-scaling short circuits
            )
            
            # Align horizontal limits across subplots if desired:
            # depth_ratio = circuit.depth() / max_depth
            # ax.set_xlim(right=...)
            
            ax.set_title(
                f"Outer iteration {_field(record, 'outer_iteration')}: "
                f"{_field(record, 'action')}",
                loc="left",
                fontsize=10,
            )
        fig.tight_layout()
    except Exception:
        plt.close(fig)
        raise

    return fig, axes
