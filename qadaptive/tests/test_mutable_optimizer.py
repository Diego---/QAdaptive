import pytest

from qiskit.circuit import QuantumCircuit, ParameterVector

from qae.optimization.my_spsa import SPSA
from qadaptive.adaptive_ansatz import AdaptiveAnsatz
from qadaptive.trainer import InnerLoopTrainer
from qadaptive.mutable_ansatz_experiment import MutableAnsatzExperiment

th = ParameterVector("t", 10)
tiny_ansatz = QuantumCircuit(3)
tiny_ansatz.rx(th[0], 0)
tiny_ansatz.rx(th[1], 1)
tiny_ansatz.rx(th[2], 2)
tiny_ansatz.cz(0, 1)
tiny_ansatz.cz(1, 2)
tiny_ansatz.rx(th[3], 0)
tiny_ansatz.rx(th[4], 1)
tiny_ansatz.rx(th[5], 2)
simple_ansatz = AdaptiveAnsatz(tiny_ansatz)

simple_trainer = InnerLoopTrainer(SPSA())

def test_muable_optimizer_initialization():
    """Test that MutableOptimizer correctly initializes."""
    mo = MutableAnsatzExperiment(simple_ansatz, simple_trainer)
    assert isinstance(mo.ansatz, QuantumCircuit)

def test_insert_at():
    mo = MutableAnsatzExperiment(simple_ansatz, simple_trainer)
    mo.insert_at('cz', [0, 1], 0)
    mo.insert_at('cz', [1, 2], 0)
    mo.insert_at('rx', [0], 3)
    mo.insert_at('ry', [0], 3)
    mo.insert_at('rx', [0], 3)
    mo.insert_at('ry', [0], 3)
    mo.insert_at('rz', [0], 2)
    
    with pytest.raises(AssertionError):
        mo.insert_at('h', [0], 0)

    assert len(mo.ansatz.data) > len(tiny_ansatz.data)

def test_insert_random():
    pass

def test_simplify_methods():
    pass

def test_insert_block_at():
    mo = MutableAnsatzExperiment(simple_ansatz, simple_trainer)
    old_len = len(mo.ansatz.data)
    old_num_params = len(mo.ansatz.parameters)

    mo.insert_block_at("rz_rx_rz", [0], 0)

    assert len(mo.ansatz.data) == old_len + 3
    assert len(mo.ansatz.parameters) == old_num_params + 3

def test_insert_two_qubit_block_at():
    mo = MutableAnsatzExperiment(simple_ansatz, simple_trainer)
    old_len = len(mo.ansatz.data)
    old_num_params = len(mo.ansatz.parameters)

    mo.insert_block_at("cx_identity", [0, 1], 0)

    assert len(mo.ansatz.data) == old_len + 6
    assert len(mo.ansatz.parameters) == old_num_params + 4

def test_get_pair_occurrence_from_circuit_index():
    mo = MutableAnsatzExperiment(simple_ansatz, simple_trainer)
    mo._2qbg_positions = {
        2: (0, 1),
        5: (0, 1),
        9: (2, 3),
        12: (2, 1),
    }

    assert mo._get_pair_occurrence_from_circuit_index(2) == (0, (0, 1))
    assert mo._get_pair_occurrence_from_circuit_index(5) == (1, (0, 1))
    assert mo._get_pair_occurrence_from_circuit_index(9) == (0, (2, 3))
    assert mo._get_pair_occurrence_from_circuit_index(12) == (0, (2, 1))
    
def test_get_pair_occurrence_from_circuit_index_invalid():
    mo = MutableAnsatzExperiment(simple_ansatz, simple_trainer)
    mo._2qbg_positions = {
        2: (0, 1),
    }

    with pytest.raises(KeyError):
        mo._get_pair_occurrence_from_circuit_index(3)
        
def test_get_circuit_index_from_pair_occurrence():
    mo = MutableAnsatzExperiment(simple_ansatz, simple_trainer)
    mo._2qbg_positions = {
        2: (0, 1),
        5: (0, 1),
        9: (2, 3),
        12: (2, 1),
    }

    assert mo._get_circuit_index_from_pair_occurrence(0, (0, 1)) == 2
    assert mo._get_circuit_index_from_pair_occurrence(1, (0, 1)) == 5
    assert mo._get_circuit_index_from_pair_occurrence(0, (2, 3)) == 9
    assert mo._get_circuit_index_from_pair_occurrence(0, (2, 1)) == 12
    
def test_get_circuit_index_from_pair_occurrence_missing():
    mo = MutableAnsatzExperiment(simple_ansatz, simple_trainer)
    mo._2qbg_positions = {
        2: (0, 1),
        5: (0, 1),
    }

    assert mo._get_circuit_index_from_pair_occurrence(2, (0, 1)) is None
    assert mo._get_circuit_index_from_pair_occurrence(0, (2, 3)) is None
    assert mo._get_circuit_index_from_pair_occurrence(-1, (0, 1)) is None
    
def test_pair_occurrence_and_circuit_index_are_inverse():
    mo = MutableAnsatzExperiment(simple_ansatz, simple_trainer)
    mo._2qbg_positions = {
        2: (0, 1),
        5: (0, 1),
        9: (2, 3),
        12: (2, 1),
    }

    for circ_index in mo._2qbg_positions:
        occ, pair = mo._get_pair_occurrence_from_circuit_index(circ_index)
        recovered = mo._get_circuit_index_from_pair_occurrence(occ, pair)
        assert recovered == circ_index

def test_lock_and_check_circuit_index():
    mo = MutableAnsatzExperiment(simple_ansatz, simple_trainer)
    mo._2qbg_positions = {
        2: (0, 1),
        5: (0, 1),
        9: (2, 3),
    }
    mo.locked_gates = set()

    mo._lock_circuit_index(5)

    assert (1, (0, 1)) in mo.locked_gates
    assert mo._is_locked_circuit_index(5)
    assert not mo._is_locked_circuit_index(2)
    assert not mo._is_locked_circuit_index(9)
    
def test_get_locked_circuit_indices():
    mo = MutableAnsatzExperiment(simple_ansatz, simple_trainer)
    mo._2qbg_positions = {
        2: (0, 1),
        5: (0, 1),
        9: (2, 3),
        12: (2, 1),
    }
    mo.locked_gates = {
        (1, (0, 1)),
        (0, (2, 1)),
    }

    assert mo._get_locked_circuit_indices() == [5, 12]
    
def test_pair_occurrence_and_circuit_index_are_inverse():
    mo = MutableAnsatzExperiment(simple_ansatz, simple_trainer)
    mo._2qbg_positions = {
        2: (0, 1),
        5: (0, 1),
        9: (2, 3),
        12: (2, 1),
    }

    for circ_index in mo._2qbg_positions:
        occ, pair = mo._get_pair_occurrence_from_circuit_index(circ_index)
        recovered = mo._get_circuit_index_from_pair_occurrence(occ, pair)
        assert recovered == circ_index

def test_remove_at_locked_two_qubit_gate_does_nothing():
    mo = MutableAnsatzExperiment(simple_ansatz, simple_trainer)
    mo.lock_gates([2])

    original_len = len(mo.ansatz.data)
    original_locked = mo.locked_gates.copy()

    mo.remove_at(2)

    assert len(mo.ansatz.data) == original_len
    assert mo.locked_gates == original_locked
