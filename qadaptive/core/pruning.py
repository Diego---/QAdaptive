import logging
from dataclasses import dataclass
from typing import Callable

import numpy as np
from qiskit import QuantumCircuit

from qadaptive.core.mutation import (
    TwoQMap,
    LockId,
    get_pair_occurrence_from_circuit_index
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PruningProposal:
    """
    Structural proposal for removing one tracked two-qubit gate.

    Attributes
    ----------
    attempted : bool
        Whether a removable target was successfully selected.
    gate_to_remove : int | None
        Circuit-data index of the selected two-qubit gate in the original ansatz.
    target_occurrence : int | None
        Pair-local occurrence index of the selected two-qubit gate.
    target_pair : tuple[int, int] | None
        Ordered qubit pair of the selected two-qubit gate.
    trial_ansatz : QuantumCircuit | None
        Trial ansatz obtained by removing the selected gate.
    note : str | None
        Optional note explaining why no proposal was made.
    """

    attempted: bool
    gate_to_remove: int | None
    target_occurrence: int | None
    target_pair: tuple[int, int] | None
    trial_ansatz: QuantumCircuit | None
    note: str | None = None

def get_removable_two_qubit_gate_indices(
    two_q_map: TwoQMap,
    locked_gates: set[LockId],
    is_locked: Callable[[int, TwoQMap, set[LockId]], bool],
) -> list[int]:
    """
    Return the circuit-data indices of tracked two-qubit gates that are not locked.

    Parameters
    ----------
    two_q_map : TwoQMap
        Mapping from circuit-data indices to tracked two-qubit gate pairs.
    locked_gates : set[LockId]
        Set of locked-gate identifiers.
    is_locked : Callable[[int, TwoQMap, set[LockId]], bool]
        Function that determines whether a tracked two-qubit gate is locked.

    Returns
    -------
    list[int]
        Sorted list of removable circuit-data indices.
    """
    return [
        circ_ind
        for circ_ind in sorted(two_q_map)
        if not is_locked(circ_ind, two_q_map, locked_gates)
    ]


def make_two_qubit_pruning_proposal(
    ansatz: QuantumCircuit,
    two_q_map: TwoQMap,
    locked_gates: set[LockId],
    is_locked: Callable[[int, TwoQMap, set[LockId]], bool],
    gate_to_remove: int | None = None,
    rng: np.random.Generator | None = None,
) -> PruningProposal:
    """
    Build a structural proposal for removing one non-locked two-qubit gate.

    This function does not evaluate cost, does not decide acceptance, and does
    not lock gates. It only selects a removable target and constructs the
    corresponding reduced ansatz.
    """
    if rng is None:
        rng = np.random.default_rng()

    removable_indices = get_removable_two_qubit_gate_indices(
        two_q_map=two_q_map,
        locked_gates=locked_gates,
        is_locked=is_locked,
    )

    if not removable_indices:
        return PruningProposal(
            attempted=False,
            gate_to_remove=None,
            target_occurrence=None,
            target_pair=None,
            trial_ansatz=None,
            note="No removable two-qubit gates are available.",
        )

    if gate_to_remove is not None:
        if gate_to_remove not in removable_indices:
            return PruningProposal(
                attempted=False,
                gate_to_remove=gate_to_remove,
                target_occurrence=None,
                target_pair=None,
                trial_ansatz=None,
                note=(
                    f"Requested gate {gate_to_remove} is not removable. "
                    f"Removable gates are {removable_indices}."
                ),
            )
    else:
        gate_to_remove = int(rng.choice(removable_indices))

    target_occurrence, target_pair = get_pair_occurrence_from_circuit_index(
        gate_to_remove,
        two_q_map,
    )

    trial_ansatz = ansatz.copy()
    trial_ansatz.data.pop(gate_to_remove)

    return PruningProposal(
        attempted=True,
        gate_to_remove=gate_to_remove,
        target_occurrence=target_occurrence,
        target_pair=target_pair,
        trial_ansatz=trial_ansatz,
        note=None,
    )
