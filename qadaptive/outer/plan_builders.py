from __future__ import annotations

import random
from collections.abc import Iterable
from typing import Any, Callable, TYPE_CHECKING

from qadaptive.outer.action_definitions import INSERT_BLOCK, SIMPLIFY, PRUNE_TWO_QUBIT
from qadaptive.outer.outer_loop import OuterStepPlan, ActionSpec
from qadaptive.core.mutation import TwoQMap

from .plan_helpers import (
    select_star_targets,
    select_nearest_neighbor_pairs,
    select_most_frequent_pair_gates
)

if TYPE_CHECKING:
    from qadaptive.outer.mutable_ansatz_experiment import MutableAnsatzExperiment


def default_append_index_policy(experiment, _target, _insertion_number: int) -> int:
    """Insert at the end of the current circuit."""
    return len(experiment.ansatz.data)

def random_insert_index_policy(experiment, _target, _insertion_number: int) -> int:
    """Insert at a random circuit-data index."""
    num_indices = len(experiment.ansatz.data)
    return random.randint(0, num_indices)

def between_2qg_indices_policy(experiment, target, insertion_number: int) -> int:
    """
    Insert between consecutive relevant two-qubit gates, distributing repeated
    insertions across available windows.

    Relevance is defined with respect to the target qubit(s): only two-qubit
    gates acting on at least one target qubit are considered.

    Candidate windows are ranked by the number of instructions between the two
    relevant consecutive two-qubit gates, with smaller gaps preferred.
    Repeated insertions are distributed across ranked windows using
    `insertion_number`, so they do not all accumulate in the same place.

    If there are fewer than two relevant two-qubit gates, append at the end.
    """
    if isinstance(target, int):
        target_qubits = {target}
    elif isinstance(target, Iterable):
        target_qubits = set(target)
    else:
        raise TypeError("`target` must be an int or an iterable of ints.")

    # Keep only 2Q gates touching the target qubit(s).
    relevant_2q = [
        (circ_index, pair)
        for circ_index, pair in sorted(experiment._2qbg_positions.items())
        if any(q in target_qubits for q in pair)
    ]

    if len(relevant_2q) < 2:
        return len(experiment.ansatz.data)

    # Build windows between consecutive relevant 2Q gates.
    candidate_windows = []
    for (left_index, left_pair), (right_index, right_pair) in zip(
        relevant_2q[:-1], relevant_2q[1:]
    ):
        gap = right_index - left_index - 1
        candidate_windows.append((gap, left_index, right_index, left_pair, right_pair))

    # Rank by smallest gap first.
    candidate_windows.sort(key=lambda x: x[0])

    # Distribute insertions across windows instead of always taking the first.
    selected_idx = insertion_number % len(candidate_windows)
    _, left_index, right_index, _, _ = candidate_windows[selected_idx]

    # Choose a position within that window.
    return random.randint(left_index + 1, right_index)

def combine_plan_builders(*builders: Callable[..., OuterStepPlan]) -> Callable[..., OuterStepPlan]:
    """
    Return a builder that combines the plans produced by all input builders.

    Any positional or keyword arguments received by the combined builder are
    forwarded to each underlying builder.
    """
    if not builders:
        raise ValueError("At least one builder must be provided.")

    def _combined_builder(*args, **kwargs) -> OuterStepPlan:
        plan = builders[0](*args, **kwargs)
        for builder in builders[1:]:
            plan = plan + builder(*args, **kwargs)
        return plan

    return _combined_builder

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
    
def build_three_qubit_block_growth_plan(
    experiment: MutableAnsatzExperiment,
    block_name: str = "big_cz_identity",
    qubits: list[int] | None = None,
    max_insertions: int = 1,
    repetitions: int = 1,
    insert_index_policy: Callable | None = None,
    force_accept: bool = False,
    add_simplify: bool = True,
    simplify_kwargs: dict[str, Any] | None = None,
    label: str | None = None,
) -> OuterStepPlan:
    """
    Build a macro-plan that grows three-qubit entangling structure across selected triplets.
    
    The plan inserts identity-initialized three-qubit blocks on selected triplets of qubits, then
    optionally appends a simplification action.
    
    Parameters
    ----------
    experiment : MutableAnsatzExperiment
        Experiment instance used to access the experiment state.
    block_name : str, optional
        Name of the block to insert for each selected triplet.
    qubits : list[int] | None, optional
        List of qubits to consider for three-qubit block insertion. If None, all qubits are considered.
    max_insertions : int, optional
        Maximum number of three-qubit block insertions to include in the plan. Default is 1.
    repetitions : int, optional
        Number of times to repeat the three-qubit growth pattern. Default is 1.
    insert_index_policy : Callable | None, optional
        Function that determines the circuit-data index for each insertion. It should have the signature
        ``(experiment, target_qubits, insertion_number) -> int``. If None, a default policy that appends at the end of the circuit is used.
    force_accept : bool, optional
        Whether to set the acceptance mode to "force", which accepts all changes made by the plan without evaluating the cost.
    add_simplify : bool, optional
        Whether to append a simplification action after the insertion burst.
    simplify_kwargs : dict[str, Any] | None, optional
        Keyword arguments for the simplification action.
    label : str | None, optional
        Optional descriptive label for the plan.
        
    Returns
    -------
    OuterStepPlan
        Three-qubit block growth plan.
    """
    if insert_index_policy is None:
        insert_index_policy = default_append_index_policy
        
    num_qubits = experiment.ansatz.num_qubits
    if num_qubits < 3:
        raise ValueError("`num_qubits` must be at least 3 for three-qubit block growth.")
    
    if num_qubits == 3 and qubits is None:
        triplet = [n for n in range(num_qubits)]
    elif qubits is not None:
        assert len(qubits) == 3, "Exactly three qubits must be specified for three-qubit block insertion."
        triplet = qubits.copy()
    else:
        triplet = random.sample(range(num_qubits), 3)
        
    actions: list[ActionSpec] = []
    insertion_number = 0
    for _ in range(repetitions):
        if insertion_number >= max_insertions:
            break
        actions.append(
            ActionSpec(
                action=INSERT_BLOCK,
                kwargs={
                    "block_name": block_name,
                    "qubits": triplet,
                    "insert_policy": insert_index_policy,
                    "insertion_number": insertion_number,
                },
                label=f"{block_name}_{'_'.join(map(str, triplet))}",
            )
        )
        insertion_number += 1
        
    if add_simplify:
        actions.append(
            ActionSpec(
                action=SIMPLIFY,
                kwargs={} if simplify_kwargs is None else dict(simplify_kwargs),
                label="simplify_after_3q_growth",
            )
        )
        
    return OuterStepPlan(
        name="three_qubit_block_growth",
        actions=actions,
        acceptance_mode="force" if force_accept else "outer",
        label=label,
    )

def build_star_growth_plan(
    experiment: MutableAnsatzExperiment,
    center_qubit: int = 0,
    block_name: str = "cx_identity",
    max_insertions: int = 2,
    repetitions: int = 1,
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
    repetitions : int, optional
        Number of times to repeat the star-growth pattern.
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

    insertion_number = 0
    
    for _ in range(repetitions):
        for target in targets:
            target_qubits = [center_qubit, target]
            actions.append(
                ActionSpec(
                    action=INSERT_BLOCK,
                    kwargs={
                        "block_name": block_name,
                        "qubits": target_qubits,
                        "insert_policy": insert_index_policy,
                        "insertion_number": insertion_number,
                    },
                    label=f"{block_name}_{center_qubit}_{target}",
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
                        "insert_policy": insert_index_policy,
                        "insertion_number": insertion_number,
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
    num_insertions: int = 1,
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
    num_insertions : int, optional
        Number of blocks to insert on each selected qubit. Default is 1.
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
    
    insertion_number = 0

    for _ in range(num_insertions):
        for qubit in qubits:
            if qubit < 0 or qubit >= num_qubits:
                raise ValueError(
                    f"Qubit index {qubit} is invalid for a {num_qubits}-qubit ansatz."
                )

            target_qubits = [qubit]
            actions.append(
                ActionSpec(
                    action=INSERT_BLOCK,
                    kwargs={
                        "block_name": block_name,
                        "qubits": target_qubits,
                        "insert_policy": insert_index_policy,
                        "insertion_number": insertion_number,
                    },
                    label=f"{block_name}_{qubit}",
                )
            )
            insertion_number += 1

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
    label : str | None, optional
        Optional descriptive label for the plan.
    add_simplify : bool, optional
        Whether to append a simplification action after the prune sweep.
    simplify_kwargs : dict[str, Any] | None, optional
        Keyword arguments for the simplification action.

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
        acceptance_mode="outer",
        label=label,
    )

def build_targeted_prune_plan(
    experiment: MutableAnsatzExperiment,
    targeting_function: Callable[[MutableAnsatzExperiment], TwoQMap] | None = None,
    max_num_2q_gates: int = 1,
    label: str | None = None,
    add_simplify: bool = False,
    simplify_kwargs: dict[str, Any] | None = None,
) -> OuterStepPlan:
    """
    Build a macro-plan that performs targeted structural prune proposals on
    selected two-qubit gates.

    The targeted gates are stored in the plan using stable pair-occurrence
    identifiers rather than raw circuit-data indices, so that later actions in
    the same plan can still target the intended gates after earlier accepted
    removals shift circuit indices.

    Parameters
    ----------
    experiment : MutableAnsatzExperiment
        Experiment instance used to access the current two-qubit gate map.
    targeting_function : Callable[[MutableAnsatzExperiment], TwoQMap] | None, optional
        Function that takes the experiment and returns a subset of
        ``experiment._2qbg_positions`` indicating which two-qubit gates should
        be targeted for pruning.

        The returned mapping must have circuit-data indices as keys and ordered
        qubit pairs as values.

        If None, a default targeting function is used that selects all gates
        belonging to the most frequent two-qubit pair.
    max_num_2q_gates : int, optional
        Maximum number of targeted two-qubit gates to include in the plan.
        Defaults to a single gate.
    label : str | None, optional
        Optional descriptive label for the plan.
    add_simplify : bool, optional
        Whether to append a simplification action after the targeted prune
        attempts.
    simplify_kwargs : dict[str, Any] | None, optional
        Keyword arguments for the simplification action.

    Returns
    -------
    OuterStepPlan
        Plan that performs targeted structural prune proposals.

    Raises
    ------
    ValueError
        If there are no tracked two-qubit gates, if the targeting function
        returns no gates, if it returns invalid indices/pairs, if
        ``max_num_2q_gates`` is not positive, or if ``acceptance_mode`` is
        invalid.
    """
    if max_num_2q_gates <= 0:
        raise ValueError("`max_num_2q_gates` must be positive.")

    current_two_q_map = dict(experiment._2qbg_positions)

    if not current_two_q_map:
        raise ValueError("No tracked two-qubit gates are available for pruning.")

    if targeting_function is None:
        targeted_map = select_most_frequent_pair_gates(experiment)
    else:
        targeted_map = dict(targeting_function(experiment))

    if not targeted_map:
        raise ValueError("The targeting function selected no two-qubit gates.")

    # Validate that the returned map is a subset of the currently tracked gates.
    for circ_ind, pair in targeted_map.items():
        if circ_ind not in current_two_q_map:
            raise ValueError(
                f"Targeted circuit index {circ_ind} is not a tracked two-qubit gate."
            )
        if current_two_q_map[circ_ind] != pair:
            raise ValueError(
                f"Targeted pair mismatch at circuit index {circ_ind}: "
                f"expected {current_two_q_map[circ_ind]}, got {pair}."
            )

    # Filter out locked targets before applying the cap.
    available_indices = [
        circ_ind
        for circ_ind in sorted(targeted_map.keys())
        if not experiment._is_locked_circuit_index(circ_ind)
    ]

    if not available_indices:
        raise ValueError("All targeted two-qubit gates are currently locked.")

    selected_indices = available_indices[:max_num_2q_gates]

    actions: list[ActionSpec] = []

    for circ_ind in selected_indices:
        occurrence, pair = experiment._get_pair_occurrence_from_circuit_index(circ_ind)

        actions.append(
            ActionSpec(
                action=PRUNE_TWO_QUBIT,
                kwargs={
                    "target_occurrence": occurrence,
                    "target_pair": pair,
                },
                label=f"targeted_prune_{pair}_{occurrence}",
            )
        )

    if add_simplify:
        actions.append(
            ActionSpec(
                action=SIMPLIFY,
                kwargs={} if simplify_kwargs is None else dict(simplify_kwargs),
                label="simplify_after_targeted_prune",
            )
        )

    return OuterStepPlan(
        name="targeted_prune",
        actions=actions,
        acceptance_mode="outer",
        label=label,
    )
