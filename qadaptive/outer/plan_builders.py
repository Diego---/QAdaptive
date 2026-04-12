from __future__ import annotations

import random
from collections import Counter
from typing import Any, Callable, TYPE_CHECKING

from qadaptive.outer.action_definitions import INSERT_BLOCK, SIMPLIFY, PRUNE_TWO_QUBIT
from qadaptive.outer.outer_loop import OuterStepPlan, ActionSpec
from qadaptive.core.mutation import TwoQMap

if TYPE_CHECKING:
    from qadaptive.outer.mutable_ansatz_experiment import MutableAnsatzExperiment


def default_append_index_policy(experiment, target, insertion_number: int) -> int:
    """Insert at the end of the current circuit."""
    return len(experiment.ansatz.data)

def random_insert_index_policy(experiment, target, insertion_number: int) -> int:
    """Insert at a random circuit-data index."""
    num_indices = len(experiment.ansatz.data)
    return random.randint(0, num_indices)

def pair_counts(two_q_map: TwoQMap) -> dict[tuple[int, int], int]:
    """
    Return the number of occurrences of each ordered two-qubit pair.

    Parameters
    ----------
    two_q_map : TwoQMap
        Mapping from circuit-data indices to ordered qubit pairs.

    Returns
    -------
    dict[tuple[int, int], int]
        Dictionary mapping each ordered pair to its occurrence count.
    """
    return dict(Counter(two_q_map.values()))

def select_star_targets(
    num_qubits: int,
    two_q_map: TwoQMap,
    center_qubit: int = 0,
    max_targets: int | None = None,
    allowed_targets: list[int] | None = None,
    forbidden_targets: list[int] | None = None,
) -> list[int]:
    """
    Select target qubits for star-centered growth around one qubit.

    Targets are prioritized by how rarely the ordered pair
    ``(center_qubit, target)`` appears in the current ansatz. This favors
    pairs that are absent or underused.

    Parameters
    ----------
    num_qubits : int
        Total number of qubits in the ansatz.
    two_q_map : TwoQMap
        Current two-qubit gate map.
    center_qubit : int, optional
        Central qubit from which star-like growth is generated.
    max_targets : int | None, optional
        Maximum number of targets to return. If None, all eligible targets are
        returned.
    allowed_targets : list[int] | None, optional
        Explicit whitelist of allowed target qubits. If None, all qubits except
        the center are considered.
    forbidden_targets : list[int] | None, optional
        Explicit blacklist of forbidden target qubits.

    Returns
    -------
    list[int]
        Ordered list of selected target qubits, from most preferred to least
        preferred.
    """
    if center_qubit < 0 or center_qubit >= num_qubits:
        raise ValueError(
            f"`center_qubit` must be between 0 and {num_qubits - 1}, got {center_qubit}."
        )

    if allowed_targets is None:
        candidates = [q for q in range(num_qubits) if q != center_qubit]
    else:
        candidates = [q for q in allowed_targets if q != center_qubit]

    if forbidden_targets is not None:
        forbidden = set(forbidden_targets)
        candidates = [q for q in candidates if q not in forbidden]

    counts = pair_counts(two_q_map)

    # Prefer absent / lightly used pairs first, break ties by target index.
    ordered = sorted(
        candidates,
        key=lambda q: (counts.get((center_qubit, q), 0), q),
    )

    if max_targets is not None:
        ordered = ordered[:max_targets]

    return ordered

def select_nearest_neighbor_pairs(
    num_qubits: int,
    two_q_map: TwoQMap,
    max_pairs: int | None = None,
    periodic: bool = False,
    allowed_pairs: list[tuple[int, int]] | None = None,
    forbidden_pairs: list[tuple[int, int]] | None = None,
) -> list[tuple[int, int]]:
    """
    Select nearest-neighbor qubit pairs for growth.

    Pairs are prioritized by how rarely they appear in the current ansatz,
    favoring absent or lightly used nearest-neighbor interactions first.

    Parameters
    ----------
    num_qubits : int
        Total number of qubits in the ansatz.
    two_q_map : TwoQMap
        Current two-qubit gate map.
    max_pairs : int | None, optional
        Maximum number of nearest-neighbor pairs to return. If None, all
        eligible nearest-neighbor pairs are returned.
    periodic : bool, optional
        If True, also include the ring-closing pair ``(num_qubits - 1, 0)``.
        If False, use an open linear chain. Default is False.
    allowed_pairs : list[tuple[int, int]] | None, optional
        Explicit whitelist of allowed nearest-neighbor pairs. If None, all
        nearest-neighbor pairs implied by `num_qubits` and `periodic` are
        considered.
    forbidden_pairs : list[tuple[int, int]] | None, optional
        Explicit blacklist of forbidden pairs.

    Returns
    -------
    list[tuple[int, int]]
        Ordered list of selected nearest-neighbor pairs from most preferred
        to least preferred.

    Raises
    ------
    ValueError
        If `num_qubits` is less than 2.
    """
    if num_qubits < 2:
        raise ValueError("`num_qubits` must be at least 2.")

    base_pairs = [(q, q + 1) for q in range(num_qubits - 1)]
    if periodic:
        base_pairs.append((num_qubits - 1, 0))

    if allowed_pairs is None:
        candidates = list(base_pairs)
    else:
        base_set = set(base_pairs)
        candidates = [pair for pair in allowed_pairs if pair in base_set]

    if forbidden_pairs is not None:
        forbidden = set(forbidden_pairs)
        candidates = [pair for pair in candidates if pair not in forbidden]

    counts = pair_counts(two_q_map)

    ordered = sorted(
        candidates,
        key=lambda pair: (counts.get(pair, 0), pair[0], pair[1]),
    )

    if max_pairs is not None:
        ordered = ordered[:max_pairs]

    return ordered


def build_nearest_neighbor_growth_plan(
    experiment: MutableAnsatzExperiment,
    block_name: str = "cx_identity",
    max_insertions: int | None = None,
    periodic: bool = False,
    insert_index_policy: Callable | None = None,
    force_accept: bool = False,
    allowed_pairs: list[tuple[int, int]] | None = None,
    forbidden_pairs: list[tuple[int, int]] | None = None,
    add_simplify: bool = True,
    simplify_kwargs: dict[str, Any] | None = None,
    label: str | None = None,
) -> OuterStepPlan:
    """
    Build a macro-plan that grows entangling structure only between
    nearest-neighbor qubits.

    The plan inserts identity-initialized two-qubit blocks on selected
    nearest-neighbor pairs, then optionally appends a simplification action.

    Pairs are chosen by favoring nearest-neighbor interactions that are
    currently absent or lightly used in the ansatz.

    Parameters
    ----------
    experiment : MutableAnsatzExperiment
        Experiment instance used to access the experiment state.
    block_name : str, optional
        Name of the block to insert for each selected pair.
    max_insertions : int | None, optional
        Maximum number of nearest-neighbor insertions to include in the plan.
        If None, all eligible nearest-neighbor pairs are used.
    periodic : bool, optional
        If True, include the ring-closing nearest-neighbor pair
        ``(num_qubits - 1, 0)``. If False, use an open linear chain.
        Default is False.
    insert_index_policy : Callable | None, optional
        Function that determines the circuit-data index for each insertion.
        It should have the signature
        ``(experiment, target_qubits, insertion_number) -> int``.
        If None, a default policy that appends at the end of the circuit is used.
    force_accept : bool, optional
        Whether to set the acceptance mode to ``"force"``, which accepts all
        changes made by the plan without evaluating the cost.
    allowed_pairs : list[tuple[int, int]] | None, optional
        Explicit whitelist of allowed nearest-neighbor pairs.
    forbidden_pairs : list[tuple[int, int]] | None, optional
        Explicit blacklist of forbidden nearest-neighbor pairs.
    add_simplify : bool, optional
        Whether to append a simplification action after the insertion burst.
    simplify_kwargs : dict[str, Any] | None, optional
        Keyword arguments for the simplification action.
    label : str | None, optional
        Optional descriptive label for the plan.

    Returns
    -------
    OuterStepPlan
        Nearest-neighbor growth plan.

    Raises
    ------
    ValueError
        If no eligible nearest-neighbor pairs are available.
    """
    if insert_index_policy is None:
        insert_index_policy = default_append_index_policy

    pairs = select_nearest_neighbor_pairs(
        num_qubits=experiment.ansatz.num_qubits,
        two_q_map=experiment._2qbg_positions,
        max_pairs=max_insertions,
        periodic=periodic,
        allowed_pairs=allowed_pairs,
        forbidden_pairs=forbidden_pairs,
    )

    if not pairs:
        raise ValueError("No eligible nearest-neighbor pairs are available for growth.")

    actions: list[ActionSpec] = []

    for insertion_number, pair in enumerate(pairs):
        target_qubits = list(pair)
        circ_ind = insert_index_policy(
            experiment,
            target_qubits,
            insertion_number,
        )

        actions.append(
            ActionSpec(
                action=INSERT_BLOCK,
                kwargs={
                    "block_name": block_name,
                    "qubits": target_qubits,
                    "circ_ind": circ_ind,
                },
                label=f"{block_name}_{pair[0]}_{pair[1]}",
            )
        )

    if add_simplify:
        actions.append(
            ActionSpec(
                action=SIMPLIFY,
                kwargs={} if simplify_kwargs is None else dict(simplify_kwargs),
                label="simplify_after_nn_growth",
            )
        )

    return OuterStepPlan(
        name="nearest_neighbor_growth",
        actions=actions,
        acceptance_mode="force" if force_accept else "outer",
        label=label,
    )

def build_star_growth_plan(
    experiment: MutableAnsatzExperiment,
    center_qubit: int = 0,
    block_name: str = "cx_identity",
    max_insertions: int = 2,
    insert_index_policy: Callable | None = None,
    force_accept: bool = False,
    allowed_targets: list[int] | None = None,
    forbidden_targets: list[int] | None = None,
    add_simplify: bool = True,
    simplify_kwargs: dict[str, Any] | None = None,
    label: str | None = None,
) -> OuterStepPlan:
    """
    Build a macro-plan that grows a star-shaped entangling structure.

    The plan inserts identity-initialized two-qubit blocks from a central qubit
    to a selected set of target qubits, then optionally appends a
    simplification action.

    Targets are chosen by favoring ordered pairs ``(center_qubit, target)``
    that are currently absent or lightly used in the ansatz.

    Parameters
    ----------
    experiment : MutableAnsatzExperiment
        Experiment instance used to access the experiment state.
    center_qubit : int, optional
        Central qubit from which star-like growth is generated.
    block_name : str, optional
        Name of the block to insert for each selected pair.
    max_insertions : int, optional
        Maximum number of block insertions to include in the plan.
    insert_index_policy : Callable | None, optional
        Function that determines the circuit-data index for each insertion.
        It should have the signature ``(experiment, target_qubits, insertion_number) -> int``. 
        If None, a default policy that appends at the end of the circuit is used.
    force_accept : bool, optional
        Whether to set the acceptance mode to "force", which accepts all changes
        made by the plan without evaluating the cost.
    allowed_targets : list[int] | None, optional
        Explicit whitelist of allowed target qubits.
    forbidden_targets : list[int] | None, optional
        Explicit blacklist of forbidden target qubits.
    add_simplify : bool, optional
        Whether to append a simplification action after the insertion burst.
    simplify_kwargs : dict[str, Any] | None, optional
        Keyword arguments for the simplification action.
    label : str | None, optional
        Optional descriptive label for the plan.

    Returns
    -------
    OuterStepPlan
        Growth plan centered on `center_qubit`.

    Raises
    ------
    ValueError
        If no eligible targets are available.
    """
    if max_insertions <= 0:
        raise ValueError("`max_insertions` must be positive.")
    
    if insert_index_policy is None:
        def insert_index_policy(experiment, target_qubits, insertion_number) -> int:
            return len(experiment.ansatz.data)

    targets = select_star_targets(
        num_qubits=experiment.ansatz.num_qubits,
        two_q_map=experiment._2qbg_positions,
        center_qubit=center_qubit,
        max_targets=max_insertions,
        allowed_targets=allowed_targets,
        forbidden_targets=forbidden_targets,
    )

    if not targets:
        raise ValueError("No eligible targets are available for star growth.")
    
    actions: list[ActionSpec] = []

    for insertion_number, target in enumerate(targets):
        target_qubits = [center_qubit, target]
        circ_ind = insert_index_policy(
            experiment,
            target_qubits,
            insertion_number,
        )

        actions.append(
            ActionSpec(
                action=INSERT_BLOCK,
                kwargs={
                    "block_name": block_name,
                    "qubits": target_qubits,
                    "circ_ind": circ_ind,
                },
                label=f"{block_name}_{center_qubit}_{target}",
            )
        )

    if add_simplify:
        actions.append(
            ActionSpec(
                action=SIMPLIFY,
                kwargs={} if simplify_kwargs is None else dict(simplify_kwargs),
                label="simplify_after_growth",
            )
        )

    return OuterStepPlan(
        name=f"star_growth_q{center_qubit}",
        actions=actions,
        acceptance_mode="force" if force_accept else "outer",
        label=label,
    )
    
def build_uniform_growth_plan(
    experiment: MutableAnsatzExperiment,
    block_name: str = "cx_identity",
    insert_index_policy: Callable | None = None,
    force_accept: bool = False,
    add_simplify: bool = True,
    simplify_kwargs: dict[str, Any] | None = None,
    label: str | None = None,
) -> OuterStepPlan:
    """
    Build a macro-plan that grows uniform entangling structure across all qubits.

    The plan inserts identity-initialized two-qubit blocks between all pairs of
    qubits, then optionally appends a simplification action.

    Parameters
    ----------
    experiment : MutableAnsatzExperiment
        Experiment instance used to access the experiment state.
    block_name : str, optional
        Name of the block to insert for each pair.
    insert_index_policy : Callable | None, optional
        Function that determines the circuit-data index for each insertion.
        It should have the signature ``(experiment, target_qubits, insertion_number) -> int``. 
        If None, a default policy that appends at the end of the circuit is used.
    force_accept : bool, optional
        Whether to set the acceptance mode to "force", which accepts all changes
        made by the plan without evaluating the cost.
    add_simplify : bool, optional
        Whether to append a simplification action after the insertion burst.
    simplify_kwargs : dict[str, Any] | None, optional
        Keyword arguments for the simplification action.
    label : str | None, optional
        Optional descriptive label for the plan.

    Returns
    -------
    OuterStepPlan
        Uniform growth plan covering all qubit pairs.

    Raises
    ------
    ValueError
        If num_qubits is less than 2.
    """
    num_qubits = experiment.ansatz.num_qubits

    if num_qubits < 2:
        raise ValueError("`num_qubits` must be at least 2.")

    if insert_index_policy is None:
        insert_index_policy = default_append_index_policy

    actions: list[ActionSpec] = []
    insertion_number = 0

    for q1 in range(num_qubits):
        for q2 in range(q1 + 1, num_qubits):
            circ_ind = insert_index_policy(experiment, (q1, q2), insertion_number)

            actions.append(
                ActionSpec(
                    action=INSERT_BLOCK,
                    kwargs={
                        "block_name": block_name,
                        "qubits": [q1, q2],
                        "circ_ind": circ_ind,
                    },
                    label=f"{block_name}_{q1}_{q2}",
                )
            )
            insertion_number += 1

    if add_simplify:
        actions.append(
            ActionSpec(
                action=SIMPLIFY,
                kwargs={} if simplify_kwargs is None else dict(simplify_kwargs),
                label="simplify_after_growth",
            )
        )

    return OuterStepPlan(
        name="uniform_growth",
        actions=actions,
        acceptance_mode="force" if force_accept else "outer",
        label=label,
    )

def build_single_qubit_block_plan(
    experiment: MutableAnsatzExperiment,
    block_name: str = "rz_rx_rz",
    qubits: list[int] | None = None,
    insert_index_policy: Callable | None = None,
    force_accept: bool = False,
    add_simplify: bool = True,
    simplify_kwargs: dict[str, Any] | None = None,
    label: str | None = None,
) -> OuterStepPlan:
    """
    Build a macro-plan that inserts one-qubit blocks on selected qubits.
    
    The plan inserts identity-initialized one-qubit blocks on the specified qubits, 
    then optionally appends a simplification action.
    
    Parameters
    ----------
    experiment : MutableAnsatzExperiment
        Experiment instance used to access the experiment state.
    block_name : str, optional
         Name of the block to insert for each selected qubit.
    qubits : list[int] | None, optional
        List of target qubits for block insertion. If None, blocks are inserted on all qubits.
    insert_index_policy : Callable | None, optional
        Function that determines the circuit-data index for each insertion.
        It should have the signature ``(experiment, target_qubits, insertion_number) -> int``. 
        If None, a default policy that appends at the end of the circuit is used.
    force_accept : bool, optional
        Whether to set the acceptance mode to "force", which accepts all changes
        made by the plan without evaluating the cost.
    add_simplify : bool, optional
        Whether to append a simplification action after the insertion burst.
    simplify_kwargs : dict[str, Any] | None, optional
        Keyword arguments for the simplification action.
    label : str | None, optional
        Optional descriptive label for the plan.
        
    Returns
    -------
    OuterStepPlan
        Plan for inserting one-qubit blocks on selected qubits.
        
    Raises
    ------
    ValueError
        If any specified qubit index is invalid.
    """
    num_qubits = experiment.ansatz.num_qubits

    if qubits is None:
        qubits = list(range(num_qubits))

    if insert_index_policy is None:
        insert_index_policy = default_append_index_policy

    actions: list[ActionSpec] = []

    for insertion_number, qubit in enumerate(qubits):
        if qubit < 0 or qubit >= num_qubits:
            raise ValueError(
                f"Qubit index {qubit} is invalid for a {num_qubits}-qubit ansatz."
            )

        circ_ind = insert_index_policy(experiment, qubit, insertion_number)

        actions.append(
            ActionSpec(
                action=INSERT_BLOCK,
                kwargs={
                    "block_name": block_name,
                    "qubits": [qubit],
                    "circ_ind": circ_ind,
                },
                label=f"{block_name}_{qubit}",
            )
        )

    if add_simplify:
        actions.append(
            ActionSpec(
                action=SIMPLIFY,
                kwargs={} if simplify_kwargs is None else dict(simplify_kwargs),
                label="simplify_after_single_qubit_growth",
            )
        )

    return OuterStepPlan(
        name="single_qubit_block_growth",
        actions=actions,
        acceptance_mode="force" if force_accept else "outer",
        label=label,
    )

def build_simplification_plan(
    experiment: MutableAnsatzExperiment,
    simplify_kwargs: dict[str, Any] | None = None,
    label: str | None = None,
) -> OuterStepPlan:
    """
    Build a macro-plan that performs a single simplification action.

    Parameters
    ----------
    experiment : MutableAnsatzExperiment
        Experiment instance used to access the experiment state.
    simplify_kwargs : dict[str, Any] | None, optional
        Keyword arguments for the simplification action.
    label : str | None, optional
        Optional descriptive label for the plan.

    Returns
    -------
    OuterStepPlan
        Simplification plan.
    """
    actions: list[ActionSpec] = [
        ActionSpec(
            action=SIMPLIFY,
            kwargs={} if simplify_kwargs is None else dict(simplify_kwargs),
            label="simplify",
        )
    ]

    return OuterStepPlan(
        name="simplification",
        actions=actions,
        acceptance_mode="outer",
        label=label,
    )

def build_prune_sweep_plan(
    experiment: MutableAnsatzExperiment,
    gate_indices: list[int] | None = None,
    temperature: float = 0.08,
    alpha: float = 0.1,
    accept_tol: float = 0.2,
    label: str | None = None,
    add_simplify: bool = False,
    simplify_kwargs: dict[str, Any] | None = None,
) -> OuterStepPlan:
    """
    Build a macro-plan that performs a sweep of targeted prune attempts over
    selected two-qubit gates.

    The selected gates are stored in the plan using stable pair-occurrence
    identifiers rather than raw circuit-data indices, so that later actions in
    the same sweep can still target the intended gates after earlier accepted
    removals shift circuit indices.

    Parameters
    ----------
    experiment : MutableAnsatzExperiment
        Experiment instance used to access the current two-qubit gate map.
    gate_indices : list[int] | None, optional
        Circuit-data indices of two-qubit gates to consider in the sweep.
        If None, all currently present two-qubit gates are used.
    temperature : float, optional
        Temperature parameter passed to the pruning routine.
    alpha : float, optional
        Locking-probability scaling parameter passed to the pruning routine.
    accept_tol : float, optional
        Acceptance tolerance passed to the pruning routine.
    label : str | None, optional
        Optional descriptive label for the plan.

    Returns
    -------
    OuterStepPlan
        Plan that performs a prune sweep in internal-acceptance mode.

    Raises
    ------
    ValueError
        If any requested circuit index is not a tracked two-qubit gate.
    """
    if gate_indices is None:
        gate_indices = list(sorted(experiment._2qbg_positions.keys()))

    actions: list[ActionSpec] = []

    for circ_ind in gate_indices:
        if circ_ind not in experiment._2qbg_positions:
            raise ValueError(
                f"Circuit index {circ_ind} is not a tracked two-qubit gate."
            )

        occurrence, pair = experiment._get_pair_occurrence_from_circuit_index(circ_ind)

        actions.append(
            ActionSpec(
                action=PRUNE_TWO_QUBIT,
                kwargs={
                    "target_occurrence": occurrence,
                    "target_pair": pair,
                    "temperature": temperature,
                    "alpha": alpha,
                    "accept_tol": accept_tol,
                },
                label=f"prune_{pair}_{occurrence}",
            )
        )
        
    if add_simplify:
        actions.append(
            ActionSpec(
                action=SIMPLIFY,
                kwargs={} if simplify_kwargs is None else dict(simplify_kwargs),
                label="simplify_after_prune_sweep",
            )
        )

    return OuterStepPlan(
        name="prune_sweep",
        actions=actions,
        acceptance_mode="internal",
        label=label,
    )
