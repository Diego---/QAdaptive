import pytest
import numpy as np

from qiskit.circuit import QuantumCircuit, ParameterVector
from qiskit.quantum_info import Operator

from qadaptive.adaptive_ansatz import AdaptiveAnsatz
from qadaptive.operator_pool import DEFAULT_BLOCK_POOL


# -----------------------------------------------------------------------------
# Shared test fixtures / reference circuits
# -----------------------------------------------------------------------------

th = ParameterVector("t", 20)
pi = np.pi

ansatz = QuantumCircuit(3)

ansatz.ry(0.5 * pi, 1)
ansatz.rx(pi, 1)
ansatz.cz(0, 1)
ansatz.ry(0.5 * pi, 1)
ansatz.rx(pi, 1)

ansatz.ry(0.5 * pi, 2)
ansatz.rx(pi, 2)
ansatz.cz(0, 2)
ansatz.ry(0.5 * pi, 2)
ansatz.rx(pi, 2)

ansatz.ry(th[0], 0)
ansatz.rx(th[1], 0)

ansatz.ry(0.5 * pi, 0)
ansatz.rx(pi, 0)
ansatz.cz(0, 1)
ansatz.ry(0.5 * pi, 0)
ansatz.rx(pi, 0)
ansatz.rz(th[2], 0)

ansatz.ry(0.5 * pi, 0)
ansatz.rx(pi, 0)
ansatz.cz(0, 2)
ansatz.ry(0.5 * pi, 0)
ansatz.rx(pi, 0)
ansatz.rz(th[3], 0)

ansatz.ry(0.5 * pi, 0)
ansatz.rx(pi, 0)
ansatz.cz(0, 1)
ansatz.ry(0.5 * pi, 0)
ansatz.rx(pi, 0)
ansatz.rz(th[4], 0)

ansatz.ry(0.5 * pi, 0)
ansatz.rx(pi, 0)
ansatz.cz(0, 2)
ansatz.ry(0.5 * pi, 0)
ansatz.rx(pi, 0)
ansatz.rz(th[5], 0)

ansatz.ry(th[6], 0)
ansatz.rx(th[7], 0)


# -----------------------------------------------------------------------------
# Helper utilities
# -----------------------------------------------------------------------------


def operation_names(circuit: QuantumCircuit) -> list[str]:
    """Return the ordered list of operation names in a circuit."""
    return [inst.operation.name for inst in circuit.data]


# -----------------------------------------------------------------------------
# Initialization
# -----------------------------------------------------------------------------


def test_adaptive_ansatz_initialization():
    """AdaptiveAnsatz should detect the trainable parameters in a complex ansatz."""
    adaptive_ansatz = AdaptiveAnsatz(ansatz)
    assert len(adaptive_ansatz.get_current_ansatz().parameters) == 8


# -----------------------------------------------------------------------------
# Single-gate editing
# -----------------------------------------------------------------------------


def test_add_gate_at_index():
    """Adding a gate at a specific index should modify the circuit in place."""
    circuit = QuantumCircuit(2)
    adapt_ansatz = AdaptiveAnsatz(circuit)

    adapt_ansatz.add_gate_at_index("rx", 0, [0])

    assert len(adapt_ansatz.get_current_ansatz().data) == 1
    assert adapt_ansatz.get_current_ansatz().data[0].name == "rx"



def test_remove_gate_at_index():
    """Removing gates by index should update circuit structure and parameters."""
    adapt_ansatz = AdaptiveAnsatz(ansatz)

    adapt_ansatz.remove_gate_by_index([8, 9])
    assert (
        len(adapt_ansatz.current_ansatz.data),
        len(adapt_ansatz.current_ansatz.parameters),
    ) == (36, 6)

    adapt_ansatz.remove_gate_by_index(0)
    assert adapt_ansatz.current_ansatz.data[0].name == "rx"

    adapt_ansatz.remove_gate_by_index(0)
    assert adapt_ansatz.current_ansatz.data[0].name == "cz"


# -----------------------------------------------------------------------------
# History tracking
# -----------------------------------------------------------------------------


def test_history_tracking():
    """With history tracking enabled, each edit should append one snapshot."""
    circuit = QuantumCircuit(2)
    adapt_ansatz = AdaptiveAnsatz(circuit, track_history=True)

    initial_history_len = len(adapt_ansatz.history)

    adapt_ansatz.add_gate_at_index("rx", 0, [0])
    assert len(adapt_ansatz.history) == initial_history_len + 1

    adapt_ansatz.add_gate_at_index("cz", 1, [0, 1])
    assert len(adapt_ansatz.history) == initial_history_len + 2



def test_no_history_tracking():
    """With history tracking disabled, edits should not be stored."""
    circuit = QuantumCircuit(2)
    adapt_ansatz = AdaptiveAnsatz(circuit, track_history=False)

    initial_history_len = len(adapt_ansatz.history)

    adapt_ansatz.add_gate_at_index("rx", 0, [0])
    adapt_ansatz.add_gate_at_index("cz", 1, [0, 1])

    assert len(adapt_ansatz.history) == initial_history_len



def test_history_tracking_after_block_addition():
    """Adding a whole block should append exactly one snapshot to history."""
    circuit = QuantumCircuit(2)
    adapt_ansatz = AdaptiveAnsatz(
        circuit,
        track_history=True,
        block_pool=DEFAULT_BLOCK_POOL,
    )

    initial_history_len = len(adapt_ansatz.history)

    adapt_ansatz.add_block_at_index("cx_identity", 0, [0, 1])

    assert len(adapt_ansatz.history) == initial_history_len + 1
    assert len(adapt_ansatz.history[0].data) == 0
    assert operation_names(adapt_ansatz.history[-1]) == [
        "rz",
        "rx",
        "rz",
        "rx",
        "cx",
        "rz",
        "rx",
        "cx",
    ]


# -----------------------------------------------------------------------------
# Rollback behavior
# -----------------------------------------------------------------------------


def test_rollback():
    """Rollback should restore the circuit to an earlier state."""
    circuit = QuantumCircuit(2)
    adapt_ansatz = AdaptiveAnsatz(circuit, track_history=True)

    initial_ansatz = adapt_ansatz.get_current_ansatz().copy()

    adapt_ansatz.add_gate_at_index("rx", 0, [0])
    adapt_ansatz.add_gate_at_index("cz", 1, [0, 1])

    assert len(adapt_ansatz.get_current_ansatz().data) != len(initial_ansatz.data)

    adapt_ansatz.rollback(2)
    rolled_back = adapt_ansatz.get_current_ansatz()

    assert len(rolled_back.data) == len(initial_ansatz.data)
    assert operation_names(rolled_back) == operation_names(initial_ansatz)
    assert set(rolled_back.parameters) == set(initial_ansatz.parameters)



def test_rollback_single_qubit_block_addition():
    """Rollback should undo insertion of a one-qubit block."""
    circuit = QuantumCircuit(1)
    adapt_ansatz = AdaptiveAnsatz(
        circuit,
        track_history=True,
        block_pool=DEFAULT_BLOCK_POOL,
    )

    initial_ansatz = adapt_ansatz.get_current_ansatz().copy()

    adapt_ansatz.add_block_at_index("rz_rx_rz", 0, [0])

    assert len(adapt_ansatz.get_current_ansatz().data) == 3
    assert len(adapt_ansatz.get_current_ansatz().parameters) == 3
    assert len(adapt_ansatz.params) == 3
    assert len(adapt_ansatz.history) == 2

    adapt_ansatz.rollback(1)
    rolled_back = adapt_ansatz.get_current_ansatz()

    assert len(rolled_back.data) == len(initial_ansatz.data)
    assert operation_names(rolled_back) == operation_names(initial_ansatz)
    assert list(rolled_back.parameters) == list(initial_ansatz.parameters)
    assert len(adapt_ansatz.params) == 0
    assert set(adapt_ansatz.params) == set(rolled_back.parameters)



def test_rollback_two_qubit_block_addition():
    """Rollback should undo insertion of a two-qubit block."""
    circuit = QuantumCircuit(2)
    adapt_ansatz = AdaptiveAnsatz(
        circuit,
        track_history=True,
        block_pool=DEFAULT_BLOCK_POOL,
    )

    initial_ansatz = adapt_ansatz.get_current_ansatz().copy()

    adapt_ansatz.add_block_at_index("cx_identity", 0, [0, 1])

    assert len(adapt_ansatz.get_current_ansatz().data) == 8
    assert len(adapt_ansatz.get_current_ansatz().parameters) == 6
    assert len(adapt_ansatz.params) == 6
    assert len(adapt_ansatz.history) == 2

    adapt_ansatz.rollback(1)
    rolled_back = adapt_ansatz.get_current_ansatz()

    assert len(rolled_back.data) == len(initial_ansatz.data)
    assert operation_names(rolled_back) == operation_names(initial_ansatz)
    assert list(rolled_back.parameters) == list(initial_ansatz.parameters)
    assert len(adapt_ansatz.params) == 0
    assert set(adapt_ansatz.params) == set(rolled_back.parameters)



def test_rollback_block_preserves_previous_gate_edit():
    """Rolling back a block addition should preserve earlier edits."""
    circuit = QuantumCircuit(2)
    adapt_ansatz = AdaptiveAnsatz(
        circuit,
        track_history=True,
        block_pool=DEFAULT_BLOCK_POOL,
    )

    adapt_ansatz.add_gate_at_index("rx", 0, [0])
    adapt_ansatz.add_block_at_index("cx_identity", 1, [0, 1])

    assert operation_names(adapt_ansatz.get_current_ansatz()) == [
        "rx",
        "rz",
        "rx",
        "rz",
        "rx",
        "cx",
        "rz",
        "rx",
        "cx",
    ]

    adapt_ansatz.rollback(1)

    assert operation_names(adapt_ansatz.get_current_ansatz()) == ["rx"]


# -----------------------------------------------------------------------------
# Block insertion
# -----------------------------------------------------------------------------


def test_add_single_qubit_block_at_index():
    """A one-qubit identity-initializable block should be inserted correctly."""
    circuit = QuantumCircuit(1)
    adapt_ansatz = AdaptiveAnsatz(circuit, block_pool=DEFAULT_BLOCK_POOL)

    adapt_ansatz.add_block_at_index("rz_rx_rz", 0, [0])

    assert operation_names(adapt_ansatz.get_current_ansatz()) == ["rz", "rx", "rz"]
    assert len(adapt_ansatz.get_current_ansatz().parameters) == 3
    assert len(adapt_ansatz.params) == 3



def test_add_two_qubit_block_at_index():
    """A two-qubit identity-initializable block should be inserted correctly."""
    circuit = QuantumCircuit(2)
    adapt_ansatz = AdaptiveAnsatz(circuit, block_pool=DEFAULT_BLOCK_POOL)

    adapt_ansatz.add_block_at_index("cx_identity", 0, [0, 1])

    assert operation_names(adapt_ansatz.get_current_ansatz()) == [
        "rz",
        "rx",
        "rz",
        "rx",
        "cx",
        "rz",
        "rx",
        "cx",
    ]
    assert len(adapt_ansatz.get_current_ansatz().parameters) == 6
    assert len(adapt_ansatz.params) == 6



def test_add_block_wrong_qubit_count_raises():
    """Adding a block to the wrong number of qubits should raise."""
    circuit = QuantumCircuit(2)
    adapt_ansatz = AdaptiveAnsatz(circuit, block_pool=DEFAULT_BLOCK_POOL)

    with pytest.raises(ValueError):
        adapt_ansatz.add_block_at_index("cx_identity", 0, [0])



def test_add_unknown_block_raises():
    """Adding an unknown block name should raise."""
    circuit = QuantumCircuit(2)
    adapt_ansatz = AdaptiveAnsatz(circuit, block_pool=DEFAULT_BLOCK_POOL)

    with pytest.raises(AssertionError):
        adapt_ansatz.add_block_at_index("not_a_block", 0, [0, 1])


# -----------------------------------------------------------------------------
# Identity-at-zero checks
# -----------------------------------------------------------------------------


def test_rz_rx_rz_block_is_identity_at_zero():
    """The single-qubit block should evaluate to identity at zero parameters."""
    circuit = QuantumCircuit(1)
    adapt_ansatz = AdaptiveAnsatz(circuit, block_pool=DEFAULT_BLOCK_POOL)

    adapt_ansatz.add_block_at_index("rz_rx_rz", 0, [0])

    block = adapt_ansatz.get_current_ansatz()
    zero_map = {p: 0.0 for p in block.parameters}
    bound = block.assign_parameters(zero_map)

    assert Operator(bound).equiv(Operator(QuantumCircuit(1)))



def test_cx_identity_block_is_identity_at_zero():
    """The two-qubit block should evaluate to identity at zero parameters."""
    circuit = QuantumCircuit(2)
    adapt_ansatz = AdaptiveAnsatz(circuit, block_pool=DEFAULT_BLOCK_POOL)

    adapt_ansatz.add_block_at_index("cx_identity", 0, [0, 1])

    block = adapt_ansatz.get_current_ansatz()
    zero_map = {p: 0.0 for p in block.parameters}
    bound = block.assign_parameters(zero_map)

    assert Operator(bound).equiv(Operator(QuantumCircuit(2)))
