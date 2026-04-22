import numpy as np
from qiskit import QuantumCircuit

from qadaptive.core.mutation import get_pair_occurrence_from_circuit_index, get_two_qubit_gate_indices, is_locked_circuit_index
from qadaptive.core.pruning import make_two_qubit_pruning_proposal, get_removable_two_qubit_gate_indices


class StubRNG:
    def __init__(self, choice_value, random_values):
        self.choice_value = choice_value
        self._random_values = iter(random_values)

    def choice(self, seq):
        return self.choice_value

    def random(self):
        return next(self._random_values)



def build_reference_ansatz() -> QuantumCircuit:
    qc = QuantumCircuit(3)
    qc.rx(0.1, 0)
    qc.cz(0, 1)
    qc.rx(0.2, 1)
    qc.cz(1, 2)
    return qc



def test_get_removable_two_qubit_gate_indices_excludes_locked_gates():
    ansatz = build_reference_ansatz()
    two_q_map = get_two_qubit_gate_indices(ansatz)
    locked_gates = {
        get_pair_occurrence_from_circuit_index(1, two_q_map),
    }

    removable = get_removable_two_qubit_gate_indices(
        two_q_map=two_q_map,
        locked_gates=locked_gates,
        is_locked=is_locked_circuit_index,
    )

    assert removable == [3]



def test_pruning_returns_not_attempted_when_no_gate_is_removable():
    ansatz = build_reference_ansatz()
    two_q_map = get_two_qubit_gate_indices(ansatz)
    locked_gates = {
        get_pair_occurrence_from_circuit_index(circ_ind, two_q_map)
        for circ_ind in two_q_map
    }

    proposal = make_two_qubit_pruning_proposal(
        ansatz=ansatz,
        two_q_map=two_q_map,
        locked_gates=locked_gates,
        is_locked=is_locked_circuit_index,
        gate_to_remove=1,
        rng=StubRNG(choice_value=1, random_values=[]),
    )

    assert not proposal.attempted
    assert proposal.gate_to_remove is None
    assert proposal.trial_ansatz is None
