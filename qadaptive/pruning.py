import logging
from dataclasses import dataclass
from typing import Callable

import numpy as np
from qiskit import QuantumCircuit

from qadaptive.mutation import TwoQMap, LockId

logger = logging.getLogger("qadaptive.core.pruning")


@dataclass(frozen=True)
class PruningDecision:
    """
    Result of evaluating one attempted two-qubit-gate pruning step.

    Attributes
    ----------
    attempted : bool
        Whether a pruning attempt was actually carried out. This is False when
        there are no removable two-qubit gates left to consider.
    accepted : bool
        Whether the proposed gate removal was accepted.
    should_lock : bool
        Whether the considered gate should be locked after rejection.
    gate_to_remove : int | None
        Circuit-data index of the considered two-qubit gate in the original
        ansatz. None if no pruning attempt was made.
    trial_ansatz : QuantumCircuit | None
        Trial ansatz obtained by removing the selected gate. None if no pruning
        attempt was made.
    trial_cost : float | None
        Objective value of the trial ansatz evaluated at the current parameter
        vector. None if no pruning attempt was made.
    current_cost : float | None
        Objective value of the current ansatz before pruning. None if no
        pruning attempt was made.
    delta_cost : float | None
        Difference ``trial_cost - current_cost``. None if no pruning attempt
        was made.
    acceptance_probability : float | None
        Acceptance probability used when the removal increased the cost. None
        if no pruning attempt was made or if immediate acceptance occurred.
    lock_probability : float | None
        Probability of locking the gate after rejection. None if no pruning
        attempt was made or if no rejection occurred.
    """

    attempted: bool
    accepted: bool
    should_lock: bool
    gate_to_remove: int | None
    trial_ansatz: QuantumCircuit | None
    trial_cost: float | None
    current_cost: float | None
    delta_cost: float | None
    acceptance_probability: float | None
    lock_probability: float | None


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


def evaluate_two_qubit_gate_pruning(
    ansatz: QuantumCircuit,
    two_q_map: TwoQMap,
    locked_gates: set[LockId],
    last_params: np.ndarray,
    last_cost: float,
    cost: Callable[[np.ndarray, QuantumCircuit], float],
    is_locked: Callable[[int, TwoQMap, set[LockId]], bool],
    temperature: float = 0.08,
    alpha: float = 0.1,
    accept_tol: float = 0.2,
    gate_to_remove: int | None = None,
    rng: np.random.Generator | None = None,
) -> PruningDecision:
    """
    Evaluate one attempted removal of a non-locked two-qubit gate.

    Parameters
    ----------
    ansatz : QuantumCircuit
        Current ansatz circuit.
    two_q_map : TwoQMap
        Mapping from circuit-data indices to tracked two-qubit gate pairs.
    locked_gates : set[LockId]
        Set of locked-gate identifiers.
    last_params : np.ndarray
        Current parameter vector used to evaluate the trial ansatz.
    last_cost : float
        Current objective value before pruning.
    cost : Callable[[np.ndarray, QuantumCircuit], float]
        Objective function used to score the trial ansatz.
    is_locked : Callable[[int, TwoQMap, set[LockId]], bool]
        Function that determines whether a tracked two-qubit gate is locked.
    temperature : float, optional
        Temperature factor for Metropolis-like acceptance of uphill moves.
    alpha : float, optional
        Scaling factor for gate-locking probability after rejection.
    accept_tol : float, optional
        Tolerance for immediate acceptance when the trial cost is lower.
    gate_to_remove : int | None, optional
        Circuit-data index of a specific gate to attempt to remove. If None,
        a random removable gate is selected for the pruning attempt.
    rng : np.random.Generator | None, optional
        Random-number generator used for gate selection and probabilistic
        acceptance. If None, a new default generator is created.

    Returns
    -------
    PruningDecision
        Decision object describing the attempted pruning step.
    """
    if rng is None:
        rng = np.random.default_rng()

    removable_indices = get_removable_two_qubit_gate_indices(
        two_q_map=two_q_map,
        locked_gates=locked_gates,
        is_locked=is_locked,
    )
    logger.info(
        "Found %s to be the removable two-qubit gates for pruning.", removable_indices
    )

    if not removable_indices:
        logger.info("No removable two-qubit gates available for pruning.")
        return PruningDecision(
            attempted=False,
            accepted=False,
            should_lock=False,
            gate_to_remove=None,
            trial_ansatz=None,
            trial_cost=None,
            current_cost=None,
            delta_cost=None,
            acceptance_probability=None,
            lock_probability=None,
        )
    
    if gate_to_remove is not None:
        if gate_to_remove not in removable_indices:
            logger.info(
                "Requested gate %s is not removable. Removable gates are %s.",
                gate_to_remove,
                removable_indices,
            )
            return PruningDecision(
                attempted=False,
                accepted=False,
                should_lock=False,
                gate_to_remove=gate_to_remove,
                trial_ansatz=None,
                trial_cost=None,
                current_cost=float(last_cost),
                delta_cost=None,
                acceptance_probability=None,
                lock_probability=None,
            )
    else:
        gate_to_remove = int(rng.choice(removable_indices))
        
    logger.info("Selected gate at circuit index %s for pruning attempt.", gate_to_remove)
    trial_ansatz = ansatz.copy()
    trial_ansatz.data.pop(gate_to_remove)

    current_cost = float(last_cost)
    trial_cost = float(cost(last_params, trial_ansatz))
    delta_cost = trial_cost - current_cost

    logger.info(
        "Pruning candidate at circuit index %s: current cost %s, trial cost %s, delta %s.",
        gate_to_remove,
        current_cost,
        trial_cost,
        delta_cost,
    )

    # Immediate acceptance if cost decreases enough.
    if delta_cost <= -accept_tol:
        logger.info("Accepted pruning because the trial cost decreased sufficiently.")
        return PruningDecision(
            attempted=True,
            accepted=True,
            should_lock=False,
            gate_to_remove=gate_to_remove,
            trial_ansatz=trial_ansatz,
            trial_cost=trial_cost,
            current_cost=current_cost,
            delta_cost=delta_cost,
            acceptance_probability=None,
            lock_probability=None,
        )

    # Metropolis-like acceptance for uphill moves.
    denom = abs(current_cost) if abs(current_cost) > 0 else 1.0
    beta = 1 / temperature if temperature > 0 else float("inf")
    acceptance_probability = float(np.exp(-beta * delta_cost / denom))

    if rng.random() < acceptance_probability:
        logger.info(
            "Accepted pruning probabilistically with acceptance probability %s.",
            acceptance_probability,
        )
        return PruningDecision(
            attempted=True,
            accepted=True,
            should_lock=False,
            gate_to_remove=gate_to_remove,
            trial_ansatz=trial_ansatz,
            trial_cost=trial_cost,
            current_cost=current_cost,
            delta_cost=delta_cost,
            acceptance_probability=acceptance_probability,
            lock_probability=None,
        )

    lock_probability = float(1 - np.exp(-alpha * delta_cost / denom))
    should_lock = bool(rng.random() < lock_probability)

    logger.info(
        "Rejected pruning. Lock probability for gate %s is %s. Locking: %s.",
        gate_to_remove,
        lock_probability,
        should_lock,
    )

    return PruningDecision(
        attempted=True,
        accepted=False,
        should_lock=should_lock,
        gate_to_remove=gate_to_remove,
        trial_ansatz=trial_ansatz,
        trial_cost=trial_cost,
        current_cost=current_cost,
        delta_cost=delta_cost,
        acceptance_probability=acceptance_probability,
        lock_probability=lock_probability,
    )
