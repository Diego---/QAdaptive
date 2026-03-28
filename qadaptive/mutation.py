import logging

from qiskit import QuantumCircuit

logger = logging.getLogger(__name__)

LockId = tuple[int, tuple[int, int]]
TwoQMap = dict[int, tuple[int, int]]


def get_two_qubit_gate_indices(circuit: QuantumCircuit) -> TwoQMap:
    """
    Return a mapping from circuit-data indices to the qubit pairs acted on by
    two-qubit gates in the given circuit.

    Parameters
    ----------
    circuit : QuantumCircuit
        Circuit to inspect.

    Returns
    -------
    dict[int, tuple[int, int]]
        Dictionary whose keys are circuit-data indices and whose values are
        the corresponding ordered qubit pairs.
    """
    pairs: TwoQMap = {}

    for idx, instruction in enumerate(circuit.data):
        if len(instruction.qubits) != 2:
            continue

        q0 = circuit.find_bit(instruction.qubits[0]).index
        q1 = circuit.find_bit(instruction.qubits[1]).index
        pairs[idx] = (q0, q1)

    return pairs


def get_pair_occurrence_from_circuit_index(
    circ_index: int,
    two_q_map: TwoQMap,
) -> LockId:
    """
    Return the pair-local occurrence index and qubit pair for a tracked
    two-qubit gate identified by its circuit-data index.

    Parameters
    ----------
    circ_index : int
        Circuit-data index of the tracked two-qubit gate.
    two_q_map : dict[int, tuple[int, int]]
        Mapping from circuit-data indices to qubit pairs.

    Returns
    -------
    tuple[int, tuple[int, int]]
        Tuple ``(occurrence, pair)`` where ``occurrence`` counts how many times
        the same pair appeared at smaller circuit-data indices.

    Raises
    ------
    KeyError
        If ``circ_index`` is not in ``two_q_map``.
    """
    if circ_index not in two_q_map:
        raise KeyError(
            f"Circuit index {circ_index} does not correspond to a tracked two-qubit gate."
        )

    pair = two_q_map[circ_index]
    occurrence = sum(
        1 for idx in sorted(two_q_map)
        if idx < circ_index and two_q_map[idx] == pair
    )

    return occurrence, pair


def get_circuit_index_from_pair_occurrence(
    occurrence: int,
    pair: tuple[int, int],
    two_q_map: TwoQMap,
) -> int | None:
    """
    Return the circuit-data index of a two-qubit gate identified by its
    pair-local occurrence and qubit pair.

    Parameters
    ----------
    occurrence : int
        Pair-local occurrence index.
    pair : tuple[int, int]
        Ordered qubit pair.
    two_q_map : dict[int, tuple[int, int]]
        Mapping from circuit-data indices to qubit pairs.

    Returns
    -------
    int | None
        Circuit-data index of the corresponding gate, or None if no such gate exists.
    """
    if occurrence < 0:
        return None

    current_occurrence = 0

    for circ_index in sorted(two_q_map):
        if two_q_map[circ_index] != pair:
            continue

        if current_occurrence == occurrence:
            return circ_index

        current_occurrence += 1

    return None


def is_locked_circuit_index(
    circ_index: int,
    two_q_map: TwoQMap,
    locked_gates: set[LockId],
) -> bool:
    """
    Return whether the tracked two-qubit gate at ``circ_index`` is locked.
    """
    lock_id = get_pair_occurrence_from_circuit_index(circ_index, two_q_map)
    return lock_id in locked_gates


def lock_circuit_index(
    circ_index: int,
    two_q_map: TwoQMap,
    locked_gates: set[LockId],
) -> set[LockId]:
    """
    Return a new locked-gate set with the tracked two-qubit gate at
    ``circ_index`` marked as locked.
    """
    lock_id = get_pair_occurrence_from_circuit_index(circ_index, two_q_map)
    new_locked_gates = set(locked_gates)
    new_locked_gates.add(lock_id)
    return new_locked_gates


def get_locked_circuit_indices(
    two_q_map: TwoQMap,
    locked_gates: set[LockId],
) -> list[int]:
    """
    Return sorted circuit-data indices of all currently locked two-qubit gates
    that still exist in the tracked map.
    """
    locked_indices = []

    for occurrence, pair in locked_gates:
        circ_index = get_circuit_index_from_pair_occurrence(
            occurrence,
            pair,
            two_q_map,
        )
        if circ_index is not None:
            locked_indices.append(circ_index)

    return sorted(locked_indices)


def update_locked_gates_on_insert(
    circuit: QuantumCircuit,
    circ_ind: int,
    old_two_q_map: TwoQMap,
    locked_gates: set[LockId],
) -> set[LockId]:
    """
    Update two-qubit bookkeeping after inserting a new two-qubit gate.

    Parameters
    ----------
    circuit : QuantumCircuit
        Updated circuit after insertion.
    circ_ind : int
        Circuit-data index where the new gate was inserted.
    old_two_q_map : dict[int, tuple[int, int]]
        Pre-insertion two-qubit gate map.
    locked_gates : set[tuple[int, tuple[int, int]]]
        Pre-insertion locked-gate set.

    Returns
    -------
    tuple[dict[int, tuple[int, int]], set[tuple[int, tuple[int, int]]]]
        Updated two-qubit map and updated locked-gate set.

    Notes
    -----
    The inserted gate starts unlocked.
    """
    inserted_instruction = circuit.data[circ_ind]
    if len(inserted_instruction.qubits) != 2:
        raise ValueError(
            f"Gate at circuit index {circ_ind} is not a two-qubit gate."
        )

    q0 = circuit.find_bit(inserted_instruction.qubits[0]).index
    q1 = circuit.find_bit(inserted_instruction.qubits[1]).index
    inserted_pair = (q0, q1)

    inserted_occurrence = 0
    for old_idx in sorted(old_two_q_map):
        if old_idx >= circ_ind:
            break
        if old_two_q_map[old_idx] == inserted_pair:
            inserted_occurrence += 1

    new_locked_gates: set[LockId] = set()
    for old_occurrence, old_pair in locked_gates:
        if old_pair != inserted_pair:
            new_locked_gates.add((old_occurrence, old_pair))
        else:
            if old_occurrence >= inserted_occurrence:
                new_locked_gates.add((old_occurrence + 1, old_pair))
            else:
                new_locked_gates.add((old_occurrence, old_pair))

    logger.info("Updated locked gates after insertion: %s", new_locked_gates)

    return new_locked_gates

def update_locked_gates_on_removal(
    circ_ind: int,
    old_two_q_map: TwoQMap,
    locked_gates: set[LockId],
) -> set[LockId]:
    """
    Update two-qubit bookkeeping after removing an unlocked two-qubit gate.

    Parameters
    ----------
    circ_ind : int
        Circuit-data index of the removed two-qubit gate.
    old_two_q_map : dict[int, tuple[int, int]]
        Pre-removal two-qubit gate map.
    locked_gates : set[tuple[int, tuple[int, int]]]
        Pre-removal locked-gate set.

    Returns
    -------
    tuple[dict[int, tuple[int, int]], set[tuple[int, tuple[int, int]]]]
        Updated two-qubit map and updated locked-gate set.

    Raises
    ------
    KeyError
        If ``circ_ind`` is not a tracked two-qubit gate.
    ValueError
        If the removed gate was locked.
    """
    removed_lock_id = get_pair_occurrence_from_circuit_index(circ_ind, old_two_q_map)
    removed_occurrence, removed_pair = removed_lock_id

    if removed_lock_id in locked_gates:
        raise ValueError(
            f"Attempted to remove locked two-qubit gate at circuit index {circ_ind}."
        )

    new_locked_gates: set[LockId] = set()
    for old_occurrence, old_pair in locked_gates:
        if old_pair != removed_pair:
            new_locked_gates.add((old_occurrence, old_pair))
        else:
            if old_occurrence > removed_occurrence:
                new_locked_gates.add((old_occurrence - 1, old_pair))
            else:
                new_locked_gates.add((old_occurrence, old_pair))

    logger.info("Updated locked gates after removal: %s", new_locked_gates)

    return new_locked_gates

def get_two_qubit_gate_offsets(circuit: QuantumCircuit) -> list[int]:
    """
    Return the instruction offsets of all two-qubit gates in a circuit.

    Parameters
    ----------
    circuit : QuantumCircuit
        Circuit to inspect.

    Returns
    -------
    list[int]
        Sorted list of instruction offsets corresponding to two-qubit gates.
    """
    return [
        idx
        for idx, instruction in enumerate(circuit.data)
        if len(instruction.qubits) == 2
    ]


def update_locked_gates_on_multiple_inserts(
    circuit: QuantumCircuit,
    inserted_indices: list[int],
    old_two_q_map: TwoQMap,
    locked_gates: set[LockId],
) -> tuple[TwoQMap, set[LockId]]:
    """
    Update two-qubit bookkeeping after inserting multiple two-qubit gates.

    Parameters
    ----------
    circuit : QuantumCircuit
        Updated circuit after all insertions have already been applied.
    inserted_indices : list[int]
        Final circuit-data indices of the newly inserted two-qubit gates.
    old_two_q_map : TwoQMap
        Pre-insertion two-qubit gate map.
    locked_gates : set[LockId]
        Pre-insertion locked-gate set.

    Returns
    -------
    tuple[TwoQMap, set[LockId]]
        Updated two-qubit map and updated locked-gate set.

    Notes
    -----
    The inserted indices must refer to the final post-insertion circuit and must
    be provided in ascending order. Each inserted gate starts unlocked.
    """
    new_two_q_map = dict(old_two_q_map)
    new_locked_gates = set(locked_gates)

    for circ_ind in sorted(inserted_indices):
        new_locked_gates = update_locked_gates_on_insert(
            circuit=circuit,
            circ_ind=circ_ind,
            old_two_q_map=new_two_q_map,
            locked_gates=new_locked_gates,
        )
    
    return new_locked_gates
