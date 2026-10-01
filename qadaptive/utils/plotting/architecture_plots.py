from __future__ import annotations

import io
from collections.abc import Callable, Mapping, Sequence
from typing import Any, TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np

from qiskit.circuit import QuantumCircuit

if TYPE_CHECKING:
    from qadaptive.outer.outer_loop import AcceptedAnsatzRecord


def _field(record: Any, name: str) -> Any:
    return record[name] if isinstance(record, Mapping) else getattr(record, name)


def _render_circuit(
    circuit: QuantumCircuit,
    *,
    scale: float,
    dpi: int,
    idle_wires: bool,
) -> np.ndarray:
    """
    Draw ``circuit`` in its own figure and return the tightly cropped RGBA image.

    Drawing without a user-supplied ``ax`` is what keeps the gate size fixed: when
    Qiskit's mpl drawer receives an ``ax``, it rescales the drawing to the axes width,
    which blows up short circuits.
    """
    circuit_fig = circuit.draw(
        output="mpl",
        fold=-1,
        idle_wires=idle_wires,
        scale=scale,
    )
    try:
        buffer = io.BytesIO()
        circuit_fig.savefig(
            buffer,
            format="png",
            dpi=dpi,
            bbox_inches="tight",
            pad_inches=0.05,
            facecolor="white",
        )
    finally:
        plt.close(circuit_fig)
    buffer.seek(0)
    return plt.imread(buffer)


def plot_architecture_evolution(
    accepted_ansatz_history: Sequence[AcceptedAnsatzRecord | Mapping[str, Any]],
    *,
    indices: Sequence[int] | None = None,
    load_circuit: Callable[[int], QuantumCircuit] | None = None,
    figsize: tuple[float, float] | None = None,
    scale: float = 0.7,
    dpi: int = 150,
    title_fontsize: float = 10,
) -> tuple[plt.Figure, np.ndarray]:
    """
    Draw selected accepted ansatz snapshots, keeping symbolic parameters.

    Every circuit is rendered at the same drawer ``scale``, so gate sizes are
    identical across snapshots and short circuits are neither enlarged nor clipped.

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
        Overall figure size as ``(width, height)``. If None, the size follows from the
        rendered circuits (width of the deepest circuit, sum of all heights). If given,
        the whole layout is scaled uniformly to fit inside it and anchored top-left, so
        relative gate sizes stay equal. Default is None.
    scale : float, optional
        Qiskit drawer scale applied identically to all circuits. Default is 0.7.
    dpi : int, optional
        Resolution at which circuits are rasterised and the figure is created. Save the
        figure with the same dpi to keep the circuits at 1:1 resolution. Default is 150.
    title_fontsize : float, optional
        Font size of the per-snapshot titles. Default is 10.

    Returns
    -------
    tuple[matplotlib.figure.Figure, numpy.ndarray]
        Created figure and a 1D NumPy array of Matplotlib axes, one per plotted snapshot.
        Each axes holds the rendered circuit as an image.

    Raises
    ------
    ValueError
        If ``accepted_ansatz_history`` is empty or if ``indices`` resolves to an empty sequence.

    Notes
    -----
    Circuits are embedded as raster images, so vector output formats (PDF, SVG) will
    contain bitmaps. Increase ``dpi`` for print quality.
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

    images = [
        _render_circuit(circuit, scale=scale, dpi=dpi, idle_wires=True)
        for circuit in circuits
    ]

    # Layout in pixels at the given dpi.
    title_px = 2.0 * title_fontsize * dpi / 72.0
    gap_px = 0.15 * dpi
    content_w = max(image.shape[1] for image in images)
    content_h = sum(image.shape[0] + title_px + gap_px for image in images)

    if figsize is None:
        fig_w_px, fig_h_px = content_w, content_h
        factor = 1.0
    else:
        fig_w_px, fig_h_px = figsize[0] * dpi, figsize[1] * dpi
        factor = min(fig_w_px / content_w, fig_h_px / content_h)

    fig = plt.figure(figsize=(fig_w_px / dpi, fig_h_px / dpi), dpi=dpi)
    axes = np.empty(len(records), dtype=object)

    try:
        y_top = fig_h_px
        for i, (record, image) in enumerate(zip(records, images)):
            img_h = image.shape[0] * factor
            img_w = image.shape[1] * factor
            y0 = y_top - title_px * factor - img_h

            # Axes box matches the image's aspect exactly -> no distortion, no clipping.
            ax = fig.add_axes(
                [0.0, y0 / fig_h_px, img_w / fig_w_px, img_h / fig_h_px]
            )
            ax.imshow(image, interpolation="none", aspect="auto")
            ax.set_axis_off()
            ax.set_title(
                f"Outer iteration {_field(record, 'outer_iteration')}: "
                f"{_field(record, 'action')}",
                loc="left",
                fontsize=title_fontsize * factor,
                pad=2,
            )
            axes[i] = ax

            y_top = y0 - gap_px * factor
    except Exception:
        plt.close(fig)
        raise

    return fig, axes
