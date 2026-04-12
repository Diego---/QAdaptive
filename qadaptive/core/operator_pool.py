from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from qiskit import QuantumCircuit
from qiskit.circuit import Parameter


@dataclass(frozen=True)
class PoolBlock:
    """
    A parametrized circuit block that can be inserted into an AdaptiveAnsatz.

    Attributes
    ----------
    name : str
        Name of the block.
    num_qubits : int
        Number of qubits the block acts on.
    num_parameters : int
        Number of free parameters introduced by the block.
    builder : Callable[[list[Parameter]], QuantumCircuit]
        Function that returns a QuantumCircuit implementing the block when
        given a list of fresh parameters.
    """

    name: str
    num_qubits: int
    num_parameters: int
    builder: Callable[[list[Parameter]], QuantumCircuit]

    def build(self, params: list[Parameter]) -> QuantumCircuit:
        """
        Build the block circuit from a list of parameters.

        Parameters
        ----------
        params : list[Parameter]
            Parameters to use in the block.

        Returns
        -------
        QuantumCircuit
            Circuit implementing the block.
        """
        if len(params) != self.num_parameters:
            raise ValueError(
                f"Block '{self.name}' expects {self.num_parameters} parameters, "
                f"but got {len(params)}."
            )
        return self.builder(params)


def _build_rz_rx_rz(params: list[Parameter]) -> QuantumCircuit:
    """
    One-qubit Euler block.

    This is not identity for arbitrary parameters, but it is identity when all
    three parameters are initialized to zero.
    """
    qc = QuantumCircuit(1, name="rz_rx_rz")
    qc.rz(params[0], 0)
    qc.rx(params[1], 0)
    qc.rz(params[2], 0)
    return qc


def _build_cx_identity(params: list[Parameter]) -> QuantumCircuit:
    """
    Two-qubit identity-initializable block.

    The block is
        CX(0, 1)
        RZ(b) on q0
        RX(c) on q0
        CX(0, 1)
        RZ(d) on q1
        RX(e) on q1

    For all parameters initialized to zero, the block is exactly identity.
    """
    qc = QuantumCircuit(2, name="cx_identity")
    qc.rz(params[0], 0)
    qc.ry(params[1], 0)
    qc.cx(0, 1)
    qc.rz(params[2], 1)
    qc.ry(params[3], 1)
    return qc

def _build_cz_identity(params: list[Parameter]) -> QuantumCircuit:
    """
    Two-qubit identity-initializable block.

    The block is
        CZ(0, 1)
        RY(b) on q0
        RX(c) on q0
        CZ(0, 1)
        RY(d) on q1
        RX(e) on q1

    For all parameters initialized to zero, the block is exactly identity.
    """
    qc = QuantumCircuit(2, name="cz_identity")
    qc.ry(params[0], 0)
    qc.rx(params[1], 0)
    qc.cz(0, 1)
    qc.ry(params[2], 1)
    qc.rx(params[3], 1)
    return qc


DEFAULT_BLOCK_POOL: dict[str, PoolBlock] = {
    "rz_rx_rz": PoolBlock(
        name="rz_rx_rz",
        num_qubits=1,
        num_parameters=3,
        builder=_build_rz_rx_rz,
    ),
    "cx_identity": PoolBlock(
        name="cx_identity",
        num_qubits=2,
        num_parameters=4,
        builder=_build_cx_identity,
    ),
    "cz_identity": PoolBlock(
        name="cz_identity",
        num_qubits=2,
        num_parameters=4,
        builder=_build_cz_identity
    ),
}
