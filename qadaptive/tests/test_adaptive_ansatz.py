import pytest
import numpy as np

from qiskit.circuit import QuantumCircuit, ParameterVector
from qiskit.circuit.library.standard_gates import get_standard_gate_name_mapping
from qiskit.quantum_info import Operator

from qadaptive.operator_pool import DEFAULT_BLOCK_POOL
from qadaptive.adaptive_ansatz import AdaptiveAnsatz

INSTRUCTION_MAP = get_standard_gate_name_mapping()

th = ParameterVector('t', 20)
pi = np.pi
ansatz = QuantumCircuit(3)

ansatz.ry(0.5*pi, 1)
ansatz.rx(pi, 1)
ansatz.cz(0,1)
ansatz.ry(0.5*pi, 1)
ansatz.rx(pi, 1)

ansatz.ry(0.5*pi, 2)
ansatz.rx(pi, 2)
ansatz.cz(0,2)
ansatz.ry(0.5*pi, 2)
ansatz.rx(pi, 2)

ansatz.ry(th[0], 0)
ansatz.rx(th[1], 0)

ansatz.ry(0.5*pi, 0)
ansatz.rx(pi, 0)
ansatz.cz(0,1)
ansatz.ry(0.5*pi, 0)
ansatz.rx(pi, 0)
ansatz.rz(th[2], 0)

ansatz.ry(0.5*pi, 0)
ansatz.rx(pi, 0)
ansatz.cz(0,2)
ansatz.ry(0.5*pi, 0)
ansatz.rx(pi, 0)
ansatz.rz(th[3], 0)

ansatz.ry(0.5*pi, 0)
ansatz.rx(pi, 0)
ansatz.cz(0,1)
ansatz.ry(0.5*pi, 0)
ansatz.rx(pi, 0)
ansatz.rz(th[4], 0)

ansatz.ry(0.5*pi, 0)
ansatz.rx(pi, 0)
ansatz.cz(0,2)
ansatz.ry(0.5*pi, 0)
ansatz.rx(pi, 0)
ansatz.rz(th[5], 0)

ansatz.ry(th[6], 0)
ansatz.rx(th[7],0)

def test_adaptive_ansatz_initialization():
    """Test that AdaptiveAnsatz correctly initializes with a complex ansatz."""
    adaptive_ansatz = AdaptiveAnsatz(ansatz)
    assert len(adaptive_ansatz.get_current_ansatz().parameters) == 8


def test_add_gate_at_index():
    """Test that adding a gate at a specific index correctly modifies the circuit."""
    circuit = QuantumCircuit(2)
    adapt_ansatz = AdaptiveAnsatz(circuit)
    adapt_ansatz.add_gate_at_index('rx', 0, [0])
    assert len(adapt_ansatz.get_current_ansatz().data) == 1
    assert adapt_ansatz.get_current_ansatz().data[0].name == 'rx'

def test_remove_gate_at_index():
    adapt_ansatz = AdaptiveAnsatz(ansatz)
    adapt_ansatz.remove_gate_by_index([8, 9])
    assert (
        len(adapt_ansatz.current_ansatz.data), len(adapt_ansatz.current_ansatz.parameters)
        ) == (36, 6)
    adapt_ansatz.remove_gate_by_index(0)
    assert adapt_ansatz.current_ansatz.data[0].name == 'rx'
    adapt_ansatz.remove_gate_by_index(0)
    assert adapt_ansatz.current_ansatz.data[0].name == 'cz'

def test_rollback():
    """Test that rollback restores the circuit to a previous state."""
    circuit = QuantumCircuit(2)
    adapt_ansatz = AdaptiveAnsatz(circuit, track_history=True)

    initial_ansatz = adapt_ansatz.get_current_ansatz().copy()

    adapt_ansatz.add_gate_at_index("rx", 0, [0])
    adapt_ansatz.add_gate_at_index("cz", 1, [0, 1])

    assert len(adapt_ansatz.get_current_ansatz().data) != len(initial_ansatz.data)

    adapt_ansatz.rollback(2)

    rolled_back = adapt_ansatz.get_current_ansatz()

    assert len(rolled_back.data) == len(initial_ansatz.data)
    assert [inst.operation.name for inst in rolled_back.data] == [
        inst.operation.name for inst in initial_ansatz.data
    ]
    assert set(rolled_back.parameters) == set(initial_ansatz.parameters)
    
def test_rollback_single_qubit_block_addition():
    """Test that rollback restores the circuit after adding a one-qubit block."""
    circuit = QuantumCircuit(1)
    adapt_ansatz = AdaptiveAnsatz(
        circuit,
        track_history=True,
        block_pool=DEFAULT_BLOCK_POOL,
    )

    initial_ansatz = adapt_ansatz.get_current_ansatz().copy()

    adapt_ansatz.add_block_at_index("rz_rx_rz", 0, [0])

    # Sanity check: the circuit changed
    assert len(adapt_ansatz.get_current_ansatz().data) == 3
    assert len(adapt_ansatz.get_current_ansatz().parameters) == 3
    assert len(adapt_ansatz.params) == 3
    assert len(adapt_ansatz.history) == 2

    adapt_ansatz.rollback(1)

    rolled_back = adapt_ansatz.get_current_ansatz()

    assert len(rolled_back.data) == len(initial_ansatz.data)
    assert [inst.operation.name for inst in rolled_back.data] == [
        inst.operation.name for inst in initial_ansatz.data
    ]
    assert list(rolled_back.parameters) == list(initial_ansatz.parameters)
    assert len(adapt_ansatz.params) == 0
    assert set(adapt_ansatz.params) == set(rolled_back.parameters)

def test_rollback_two_qubit_block_addition():
    """Test that rollback restores the circuit after adding a two-qubit block."""
    circuit = QuantumCircuit(2)
    adapt_ansatz = AdaptiveAnsatz(
        circuit,
        track_history=True,
        block_pool=DEFAULT_BLOCK_POOL,
    )

    initial_ansatz = adapt_ansatz.get_current_ansatz().copy()

    adapt_ansatz.add_block_at_index("cx_identity", 0, [0, 1])

    # Sanity check: the circuit changed
    assert len(adapt_ansatz.get_current_ansatz().data) == 8
    assert len(adapt_ansatz.get_current_ansatz().parameters) == 6
    assert len(adapt_ansatz.params) == 6
    assert len(adapt_ansatz.history) == 2

    adapt_ansatz.rollback(1)

    rolled_back = adapt_ansatz.get_current_ansatz()

    assert len(rolled_back.data) == len(initial_ansatz.data)
    assert [inst.operation.name for inst in rolled_back.data] == [
        inst.operation.name for inst in initial_ansatz.data
    ]
    assert list(rolled_back.parameters) == list(initial_ansatz.parameters)
    assert len(adapt_ansatz.params) == 0
    assert set(adapt_ansatz.params) == set(rolled_back.parameters)


def test_history_tracking_after_block_addition():
    """Test that adding a block appends exactly one snapshot to history."""
    circuit = QuantumCircuit(2)
    adapt_ansatz = AdaptiveAnsatz(
        circuit,
        track_history=True,
        block_pool=DEFAULT_BLOCK_POOL,
    )

    initial_history_len = len(adapt_ansatz.history)

    adapt_ansatz.add_block_at_index("cx_identity", 0, [0, 1])

    assert len(adapt_ansatz.history) == initial_history_len + 1

    # First entry should still be the original empty circuit
    assert len(adapt_ansatz.history[0].data) == 0

    # Latest entry should contain the inserted block
    assert len(adapt_ansatz.history[-1].data) == 8
    assert [inst.operation.name for inst in adapt_ansatz.history[-1].data] == [
        "rz", "rx", "rz", "rx", "cx", "rz", "rx", "cx"
    ]
    
def test_rollback_block_preserves_previous_gate_edit():
    """Test that rolling back a block addition preserves earlier edits."""
    circuit = QuantumCircuit(2)
    adapt_ansatz = AdaptiveAnsatz(
        circuit,
        track_history=True,
        block_pool=DEFAULT_BLOCK_POOL,
    )

    adapt_ansatz.add_gate_at_index("rx", 0, [0])
    adapt_ansatz.add_block_at_index("cx_identity", 1, [0, 1])

    current_names = [inst.operation.name for inst in adapt_ansatz.get_current_ansatz().data]
    assert current_names == ["rx", "rz", "rx", "rz", "rx", "cx", "rz", "rx", "cx"]

    adapt_ansatz.rollback(1)

    rolled_back_names = [inst.operation.name for inst in adapt_ansatz.get_current_ansatz().data]
    assert rolled_back_names == ["rx"]

def test_history_tracking():
    """Test that enabling history tracking correctly stores previous circuit states."""
    circuit = QuantumCircuit(2)
    adapt_ansatz = AdaptiveAnsatz(circuit, track_history=True)

    initial_history_len = len(adapt_ansatz.history)

    adapt_ansatz.add_gate_at_index("rx", 0, [0])
    assert len(adapt_ansatz.history) == initial_history_len + 1

    adapt_ansatz.add_gate_at_index("cz", 1, [0, 1])
    assert len(adapt_ansatz.history) == initial_history_len + 2
    
def test_no_history_tracking():
    """Test that disabling history tracking does not store circuit states."""
    circuit = QuantumCircuit(2)
    adapt_ansatz = AdaptiveAnsatz(circuit, track_history=False)

    # Adjust this depending on whether `history` exists when tracking is off.
    initial_history_len = len(adapt_ansatz.history)

    adapt_ansatz.add_gate_at_index("rx", 0, [0])
    adapt_ansatz.add_gate_at_index("cz", 1, [0, 1])

    assert len(adapt_ansatz.history) == initial_history_len

def test_add_single_qubit_block_at_index():
    circuit = QuantumCircuit(1)
    adapt_ansatz = AdaptiveAnsatz(circuit, block_pool=DEFAULT_BLOCK_POOL)

    adapt_ansatz.add_block_at_index("rz_rx_rz", 0, [0])

    assert len(adapt_ansatz.get_current_ansatz().data) == 3
    assert adapt_ansatz.get_current_ansatz().data[0].operation.name == "rz"
    assert adapt_ansatz.get_current_ansatz().data[1].operation.name == "rx"
    assert adapt_ansatz.get_current_ansatz().data[2].operation.name == "rz"
    assert len(adapt_ansatz.get_current_ansatz().parameters) == 3
    assert len(adapt_ansatz.params) == 3

def test_add_two_qubit_block_at_index():
    circuit = QuantumCircuit(2)
    adapt_ansatz = AdaptiveAnsatz(circuit, block_pool=DEFAULT_BLOCK_POOL)

    adapt_ansatz.add_block_at_index("cx_identity", 0, [0, 1])

    names = [inst.operation.name for inst in adapt_ansatz.get_current_ansatz().data]
    assert names == ["rz", "rx", "rz", "rx", "cx", "rz", "rx", "cx"]
    assert len(adapt_ansatz.get_current_ansatz().parameters) == 6
    assert len(adapt_ansatz.params) == 6

def test_add_block_wrong_qubit_count_raises():
    circuit = QuantumCircuit(2)
    adapt_ansatz = AdaptiveAnsatz(circuit, block_pool=DEFAULT_BLOCK_POOL)

    with pytest.raises(ValueError):
        adapt_ansatz.add_block_at_index("cx_identity", 0, [0])

def test_add_unknown_block_raises():
    circuit = QuantumCircuit(2)
    adapt_ansatz = AdaptiveAnsatz(circuit, block_pool=DEFAULT_BLOCK_POOL)

    with pytest.raises(AssertionError):
        adapt_ansatz.add_block_at_index("not_a_block", 0, [0, 1])

def test_rz_rx_rz_block_is_identity_at_zero():
    circuit = QuantumCircuit(1)
    adapt_ansatz = AdaptiveAnsatz(circuit, block_pool=DEFAULT_BLOCK_POOL)

    adapt_ansatz.add_block_at_index("rz_rx_rz", 0, [0])

    block = adapt_ansatz.get_current_ansatz()
    zero_map = {p: 0.0 for p in block.parameters}
    bound = block.assign_parameters(zero_map)

    assert Operator(bound).equiv(Operator(QuantumCircuit(1)))
    
def test_cx_identity_block_is_identity_at_zero():
    circuit = QuantumCircuit(2)
    adapt_ansatz = AdaptiveAnsatz(circuit, block_pool=DEFAULT_BLOCK_POOL)

    adapt_ansatz.add_block_at_index("cx_identity", 0, [0, 1])

    block = adapt_ansatz.get_current_ansatz()
    zero_map = {p: 0.0 for p in block.parameters}
    bound = block.assign_parameters(zero_map)

    assert Operator(bound).equiv(Operator(QuantumCircuit(2)))
