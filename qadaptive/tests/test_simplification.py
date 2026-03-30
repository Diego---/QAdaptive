from qiskit import QuantumCircuit
from qiskit.transpiler import PassManager
from qiskit.transpiler.passes import RemoveBarriers

from qadaptive.simplification import (
    build_pass_manager,
    can_preserve_locked_gates_trivially,
    simplify_ansatz,
    simplify_circuit_with_transpiler,
    two_q_pair_sequence,
)



def test_two_q_pair_sequence_ignores_absolute_indices():
    two_q_map = {2: (0, 1), 5: (1, 2), 9: (0, 1)}
    assert two_q_pair_sequence(two_q_map) == [(0, 1), (1, 2), (0, 1)]



def test_can_preserve_locked_gates_trivially_when_only_indices_shift():
    old_two_q_map = {1: (0, 1), 4: (1, 2)}
    new_two_q_map = {0: (0, 1), 2: (1, 2)}

    assert can_preserve_locked_gates_trivially(old_two_q_map, new_two_q_map)



def test_can_preserve_locked_gates_trivially_is_false_when_pair_order_changes():
    old_two_q_map = {1: (0, 1), 4: (1, 2)}
    new_two_q_map = {0: (1, 2), 2: (0, 1)}

    assert not can_preserve_locked_gates_trivially(old_two_q_map, new_two_q_map)



def test_build_pass_manager_returns_user_supplied_manager():
    pm = PassManager()
    assert build_pass_manager(pm) is pm



def test_simplify_circuit_with_transpiler_empty_manager_leaves_circuit_unchanged():
    qc = QuantumCircuit(2)
    qc.cz(0, 1)
    qc.rx(0.1, 0)

    simplified = simplify_circuit_with_transpiler(qc, PassManager(), repetitions=2)

    assert simplified == qc



def test_simplify_ansatz_reports_barrier_removal_and_preserves_two_qubit_sequence():
    qc = QuantumCircuit(2)
    qc.cz(0, 1)
    qc.barrier()
    qc.cz(0, 1)

    result = simplify_ansatz(
        circuit=qc,
        pass_manager=PassManager([RemoveBarriers()]),
        repetitions=1,
    )

    assert result.changed
    assert result.old_two_q_map == {0: (0, 1), 1: (0, 1), 2: (0, 1)}
    assert result.new_two_q_map == {0: (0, 1), 1: (0, 1)}
    assert not result.preserve_locked_gates
