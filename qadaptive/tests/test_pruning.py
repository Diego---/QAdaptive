import numpy as np
from qiskit import QuantumCircuit

from qadaptive.mutation import get_pair_occurrence_from_circuit_index, get_two_qubit_gate_indices, is_locked_circuit_index
from qadaptive.pruning import evaluate_two_qubit_gate_pruning, get_removable_two_qubit_gate_indices


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

    decision = evaluate_two_qubit_gate_pruning(
        ansatz=ansatz,
        two_q_map=two_q_map,
        locked_gates=locked_gates,
        last_params=np.array([]),
        last_cost=1.0,
        cost=lambda params, ansatz: float(len(ansatz.data)),
        is_locked=is_locked_circuit_index,
        rng=StubRNG(choice_value=1, random_values=[]),
    )

    assert not decision.attempted
    assert decision.gate_to_remove is None
    assert decision.trial_ansatz is None



def test_pruning_accepts_immediately_when_trial_cost_decreases_enough():
    ansatz = build_reference_ansatz()
    two_q_map = get_two_qubit_gate_indices(ansatz)

    decision = evaluate_two_qubit_gate_pruning(
        ansatz=ansatz,
        two_q_map=two_q_map,
        locked_gates=set(),
        last_params=np.array([]),
        last_cost=float(len(ansatz.data)),
        cost=lambda params, ansatz: float(len(ansatz.data)),
        is_locked=is_locked_circuit_index,
        accept_tol=0.2,
        rng=StubRNG(choice_value=1, random_values=[]),
    )

    assert decision.attempted
    assert decision.accepted
    assert not decision.should_lock
    assert decision.gate_to_remove == 1
    assert len(decision.trial_ansatz.data) == len(ansatz.data) - 1
    assert decision.delta_cost < 0



def test_pruning_rejection_can_lock_the_gate():
    ansatz = build_reference_ansatz()
    two_q_map = get_two_qubit_gate_indices(ansatz)

    decision = evaluate_two_qubit_gate_pruning(
        ansatz=ansatz,
        two_q_map=two_q_map,
        locked_gates=set(),
        last_params=np.array([]),
        last_cost=6.0,
        cost=lambda params, ansatz: float(10 - len(ansatz.data)),
        is_locked=is_locked_circuit_index,
        temperature=0.01,
        alpha=100.0,
        accept_tol=0.2,
        rng=StubRNG(choice_value=1, random_values=[0.999999, 0.0]),
    )

    assert decision.attempted
    assert not decision.accepted
    assert decision.should_lock
    assert decision.gate_to_remove == 1
    assert decision.delta_cost > 0
    assert decision.acceptance_probability is not None
    assert decision.lock_probability is not None
