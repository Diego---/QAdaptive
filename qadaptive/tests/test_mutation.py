import pytest
from qiskit import QuantumCircuit

from qadaptive.core.mutation import (
    get_circuit_index_from_pair_occurrence,
    get_locked_circuit_indices,
    get_pair_occurrence_from_circuit_index,
    get_two_qubit_gate_indices,
    get_two_qubit_gate_offsets,
    is_locked_circuit_index,
    lock_circuit_index,
    update_locked_gates_on_insert,
    update_locked_gates_on_removal,
)



def test_get_two_qubit_gate_indices_tracks_pairs_in_circuit_order():
    qc = QuantumCircuit(3)
    qc.rx(0.1, 0)
    qc.cz(0, 1)
    qc.cx(1, 2)
    qc.rz(0.2, 2)
    qc.cz(0, 1)

    assert get_two_qubit_gate_indices(qc) == {
        1: (0, 1),
        2: (1, 2),
        4: (0, 1),
    }



def test_pair_occurrence_helpers_are_inverse_on_repeated_pairs():
    two_q_map = {
        1: (0, 1),
        2: (0, 1),
        5: (1, 2),
        7: (0, 1),
    }

    assert get_pair_occurrence_from_circuit_index(1, two_q_map) == (0, (0, 1))
    assert get_pair_occurrence_from_circuit_index(2, two_q_map) == (1, (0, 1))
    assert get_pair_occurrence_from_circuit_index(5, two_q_map) == (0, (1, 2))
    assert get_pair_occurrence_from_circuit_index(7, two_q_map) == (2, (0, 1))

    for circ_index in two_q_map:
        occ, pair = get_pair_occurrence_from_circuit_index(circ_index, two_q_map)
        assert get_circuit_index_from_pair_occurrence(occ, pair, two_q_map) == circ_index



def test_lock_helpers_mark_and_recover_locked_indices():
    two_q_map = {
        1: (0, 1),
        2: (0, 1),
        5: (1, 2),
    }
    locked = set()

    locked = lock_circuit_index(2, two_q_map, locked)
    locked = lock_circuit_index(5, two_q_map, locked)

    assert is_locked_circuit_index(2, two_q_map, locked)
    assert is_locked_circuit_index(5, two_q_map, locked)
    assert not is_locked_circuit_index(1, two_q_map, locked)
    assert get_locked_circuit_indices(two_q_map, locked) == [2, 5]



def test_update_locked_gates_on_insert_shifts_later_occurrences_of_same_pair():
    old_two_q_map = {
        1: (0, 1),
        2: (0, 1),
        3: (1, 2),
    }
    locked_gates = {
        (1, (0, 1)),
        (0, (1, 2)),
    }

    updated_circuit = QuantumCircuit(3)
    updated_circuit.rx(0.1, 0)
    updated_circuit.cz(0, 1)
    updated_circuit.cz(0, 1)  # newly inserted at index 2
    updated_circuit.cx(0, 1)
    updated_circuit.cz(1, 2)

    updated_locked = update_locked_gates_on_insert(
        circuit=updated_circuit,
        circ_ind=2,
        old_two_q_map=old_two_q_map,
        locked_gates=locked_gates,
    )

    assert updated_locked == {
        (2, (0, 1)),
        (0, (1, 2)),
    }



def test_update_locked_gates_on_insert_raises_for_non_two_qubit_gate():
    old_two_q_map = {1: (0, 1)}

    qc = QuantumCircuit(2)
    qc.rx(0.1, 0)
    qc.cz(0, 1)

    with pytest.raises(ValueError):
        update_locked_gates_on_insert(
            circuit=qc,
            circ_ind=0,
            old_two_q_map=old_two_q_map,
            locked_gates=set(),
        )



def test_update_locked_gates_on_removal_shifts_later_occurrences_down():
    old_two_q_map = {
        1: (0, 1),
        2: (0, 1),
        3: (1, 2),
        4: (0, 1),
    }
    locked_gates = {
        (2, (0, 1)),
        (0, (1, 2)),
    }

    updated_locked = update_locked_gates_on_removal(
        circ_ind=2,
        old_two_q_map=old_two_q_map,
        locked_gates=locked_gates,
    )

    assert updated_locked == {
        (1, (0, 1)),
        (0, (1, 2)),
    }



def test_update_locked_gates_on_removal_raises_for_locked_gate():
    old_two_q_map = {
        1: (0, 1),
        2: (0, 1),
    }
    locked_gates = {(1, (0, 1))}

    with pytest.raises(ValueError):
        update_locked_gates_on_removal(
            circ_ind=2,
            old_two_q_map=old_two_q_map,
            locked_gates=locked_gates,
        )



def test_get_two_qubit_gate_offsets_returns_only_two_qubit_positions():
    qc = QuantumCircuit(3)
    qc.rx(0.1, 0)
    qc.cz(0, 1)
    qc.rz(0.2, 1)
    qc.cx(1, 2)

    assert get_two_qubit_gate_offsets(qc) == [1, 3]
