import pytest
from qiskit import QuantumCircuit
from qiskit.circuit.library import RXGate
from qadaptive.adaptive_ansatz import AdaptiveAnsatz

def test_add_gate():
    """Test that adding a random gate increases the number of circuit operations."""
    circuit = QuantumCircuit(2)
    ansatz = AdaptiveAnsatz(circuit)
    ansatz.add_random_gate()
    assert len(ansatz.get_current_ansatz().data) == 1

def test_remove_gate():
    """Test that removing a gate correctly decreases the number of circuit operations."""
    circuit = QuantumCircuit(2)
    ansatz = AdaptiveAnsatz(circuit)
    ansatz.add_random_gate()
    ansatz.remove_gate(0)
    assert len(ansatz.get_current_ansatz().data) == 0

def test_rollback():
    """Test that rollback restores the circuit to a previous state."""
    circuit = QuantumCircuit(2)
    ansatz = AdaptiveAnsatz(circuit)
    ansatz.add_random_gate()
    ansatz.rollback()
    assert len(ansatz.get_current_ansatz().data) == 0

def test_history_tracking():
    """Test that enabling history tracking correctly stores previous circuit states."""
    circuit = QuantumCircuit(2)
    ansatz = AdaptiveAnsatz(circuit, track_history=True)
    ansatz.add_random_gate()
    assert len(ansatz.history) == 2  # Initial state + 1 modification
