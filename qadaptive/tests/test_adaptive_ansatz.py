import pytest
import numpy as np

from qiskit.circuit import QuantumCircuit, ParameterVector
from qiskit.circuit.library.standard_gates import get_standard_gate_name_mapping

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
    assert len(adaptive_ansatz.get_current_ansatz().parameters) == len(th)


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
