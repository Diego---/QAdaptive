import pytest
from qiskit import QuantumCircuit
from qiskit.circuit.library.standard_gates import get_standard_gate_name_mapping

from qadaptive.adaptive_ansatz import AdaptiveAnsatz

INSTRUCTION_MAP = get_standard_gate_name_mapping()

def test_add_gate_at_index():
    """Test that adding a gate at a specific index correctly modifies the circuit."""
    circuit = QuantumCircuit(2)
    ansatz = AdaptiveAnsatz(circuit)
    ansatz.add_gate_at_index('rx', 0, [0])
    assert len(ansatz.get_current_ansatz().data) == 1
    assert ansatz.get_current_ansatz().data[0].operation.name == 'rx'

def test_remove_gate():
    """Test that removing a gate correctly decreases the number of circuit operations."""
    pass

def test_rollback():
    """Test that rollback restores the circuit to a previous state."""
    pass

def test_history_tracking():
    """Test that enabling history tracking correctly stores previous circuit states."""
    pass
