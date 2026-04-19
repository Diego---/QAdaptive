from __future__ import annotations

import random
from typing import Any, Callable, TYPE_CHECKING
from collections import Counter

from qadaptive.core.mutation import TwoQMap

if TYPE_CHECKING:
    from qadaptive.outer.mutable_ansatz_experiment import MutableAnsatzExperiment


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

def take_first_n_gates(two_q_map: TwoQMap, n: int | None) -> TwoQMap:
    """
    Return at most the first `n` two-qubit gates from a TwoQMap, ordered by
    circuit-data index.

    Parameters
    ----------
    two_q_map : TwoQMap
        Mapping from circuit-data indices to ordered qubit pairs.
    n : int | None
        Maximum number of gates to keep. If None, all gates are kept.

    Returns
    -------
    TwoQMap
        Possibly truncated TwoQMap.

    Raises
    ------
    ValueError
        If `n` is not positive when provided.
    """
    if n is None:
        return dict(sorted(two_q_map.items()))

    if n <= 0:
        raise ValueError("`n` must be positive if provided.")

    return dict(sorted(two_q_map.items())[:n])

def select_most_frequent_pair_gates(
    experiment: MutableAnsatzExperiment,
) -> TwoQMap:
    """
    Return all currently tracked two-qubit gates belonging to the most
    frequently occurring ordered pair.

    If several pairs are tied, one of them is selected uniformly at random.
    """
    two_q_map = dict(experiment._2qbg_positions)

    if not two_q_map:
        return {}

    counts = pair_counts(two_q_map)
    max_count = max(counts.values())
    most_frequent_pairs = [pair for pair, count in counts.items() if count == max_count]
    chosen_pair = random.choice(most_frequent_pairs)

    return {
        circ_ind: pair
        for circ_ind, pair in two_q_map.items()
        if pair == chosen_pair
    }

def select_most_frequent_pair_gates(
    experiment: MutableAnsatzExperiment,
) -> TwoQMap:
    """
    Select all tracked two-qubit gates belonging to the most frequently
    occurring ordered pair.

    If several pairs are tied, one of them is selected uniformly at random.

    Parameters
    ----------
    experiment : MutableAnsatzExperiment
        Experiment instance providing the current two-qubit gate map.

    Returns
    -------
    TwoQMap
        Subset of the current two-qubit map containing only the selected pair.
        Returns an empty mapping if no two-qubit gates are present.
    """
    two_q_map = dict(experiment._2qbg_positions)

    if not two_q_map:
        return {}

    counts = pair_counts(two_q_map)
    max_count = max(counts.values())
    candidate_pairs = [pair for pair, count in counts.items() if count == max_count]
    chosen_pair = random.choice(candidate_pairs)

    return {
        circ_ind: pair
        for circ_ind, pair in sorted(two_q_map.items())
        if pair == chosen_pair
    }
    
def select_least_frequent_pair_gates(
    experiment: MutableAnsatzExperiment,
) -> TwoQMap:
    """
    Select all tracked two-qubit gates belonging to the least frequently
    occurring ordered pair.

    If several pairs are tied, one of them is selected uniformly at random.

    Parameters
    ----------
    experiment : MutableAnsatzExperiment
        Experiment instance providing the current two-qubit gate map.

    Returns
    -------
    TwoQMap
        Subset of the current two-qubit map containing only the selected pair.
        Returns an empty mapping if no two-qubit gates are present.
    """
    two_q_map = dict(experiment._2qbg_positions)

    if not two_q_map:
        return {}

    counts = pair_counts(two_q_map)
    min_count = min(counts.values())
    candidate_pairs = [pair for pair, count in counts.items() if count == min_count]
    chosen_pair = random.choice(candidate_pairs)

    return {
        circ_ind: pair
        for circ_ind, pair in sorted(two_q_map.items())
        if pair == chosen_pair
    }
    
def select_gates_on_qubit(
    experiment: MutableAnsatzExperiment,
    qubit: int,
) -> TwoQMap:
    """
    Select all tracked two-qubit gates that act on a given qubit.

    Parameters
    ----------
    experiment : MutableAnsatzExperiment
        Experiment instance providing the current two-qubit gate map.
    qubit : int
        Qubit index that must participate in the selected gates.

    Returns
    -------
    TwoQMap
        Subset of the current two-qubit map containing only gates that touch
        `qubit`.
    """
    return {
        circ_ind: pair
        for circ_ind, pair in sorted(experiment._2qbg_positions.items())
        if qubit in pair
    }
    
def select_gates_on_pair(
    experiment: MutableAnsatzExperiment,
    pair: tuple[int, int],
) -> TwoQMap:
    """
    Select all tracked two-qubit gates belonging to a specific ordered pair.

    Parameters
    ----------
    experiment : MutableAnsatzExperiment
        Experiment instance providing the current two-qubit gate map.
    pair : tuple[int, int]
        Ordered qubit pair to select.

    Returns
    -------
    TwoQMap
        Subset of the current two-qubit map containing only gates on `pair`.
    """
    return {
        circ_ind: stored_pair
        for circ_ind, stored_pair in sorted(experiment._2qbg_positions.items())
        if stored_pair == pair
    }
    
def select_locked_gates(
    experiment: MutableAnsatzExperiment,
) -> TwoQMap:
    """
    Select all currently tracked two-qubit gates that are marked as locked.

    This helper assumes that `experiment.locked_gates` stores lock identifiers
    compatible with the experiment's pair-occurrence bookkeeping.

    Parameters
    ----------
    experiment : MutableAnsatzExperiment
        Experiment instance providing two-qubit bookkeeping and lock state.

    Returns
    -------
    TwoQMap
        Subset of the current two-qubit map containing only locked gates.
    """
    selected: TwoQMap = {}

    for circ_ind, pair in sorted(experiment._2qbg_positions.items()):
        occurrence, stored_pair = experiment._get_pair_occurrence_from_circuit_index(circ_ind)
        lock_id = (occurrence, stored_pair)

        if lock_id in experiment.locked_gates:
            selected[circ_ind] = pair

    return selected

# Closures for common selection patterns can be easily constructed using the above helpers.

def make_select_gates_on_qubit(
    qubit: int,
    max_gates: int | None = None,
) -> Callable[[MutableAnsatzExperiment], TwoQMap]:
    """
    Build a selector that returns gates acting on a specific qubit.

    Parameters
    ----------
    qubit : int
        Qubit index that must participate in the selected gates.
    max_gates : int | None, optional
        Maximum number of gates to return, ordered by circuit-data index.

    Returns
    -------
    Callable[[MutableAnsatzExperiment], TwoQMap]
        Selector function compatible with `build_targeted_prune_plan`.
    """
    def selector(experiment: MutableAnsatzExperiment) -> TwoQMap:
        selected = select_gates_on_qubit(experiment, qubit=qubit)
        return take_first_n_gates(selected, max_gates)

    return selector

def make_select_gates_on_pair(
    pair: tuple[int, int],
    max_gates: int | None = None,
) -> Callable[[MutableAnsatzExperiment], TwoQMap]:
    """
    Build a selector that returns gates belonging to a specific ordered pair.

    Parameters
    ----------
    pair : tuple[int, int]
        Ordered qubit pair to select.
    max_gates : int | None, optional
        Maximum number of gates to return, ordered by circuit-data index.

    Returns
    -------
    Callable[[MutableAnsatzExperiment], TwoQMap]
        Selector function compatible with `build_targeted_prune_plan`.
    """
    def selector(experiment: MutableAnsatzExperiment) -> TwoQMap:
        selected = select_gates_on_pair(experiment, pair=pair)
        return take_first_n_gates(selected, max_gates)

    return selector

def make_select_locked_gates(
    max_gates: int | None = None,
) -> Callable[[MutableAnsatzExperiment], TwoQMap]:
    """
    Build a selector that returns locked two-qubit gates.

    Parameters
    ----------
    max_gates : int | None, optional
        Maximum number of locked gates to return, ordered by circuit-data index.

    Returns
    -------
    Callable[[MutableAnsatzExperiment], TwoQMap]
        Selector function compatible with `build_targeted_prune_plan`.
    """
    def selector(experiment: MutableAnsatzExperiment) -> TwoQMap:
        selected = select_locked_gates(experiment)
        return take_first_n_gates(selected, max_gates)

    return selector

def make_select_random_gates(
    num_gates: int = 1,
) -> Callable[[MutableAnsatzExperiment], TwoQMap]:
    """
    Build a selector that returns a random subset of tracked two-qubit gates.

    Parameters
    ----------
    num_gates : int, optional
        Number of gates to sample.

    Returns
    -------
    Callable[[MutableAnsatzExperiment], TwoQMap]
        Selector function compatible with `build_targeted_prune_plan`.

    Raises
    ------
    ValueError
        If `num_gates` is not positive.
    """
    if num_gates <= 0:
        raise ValueError("`num_gates` must be positive.")

    def selector(experiment: MutableAnsatzExperiment) -> TwoQMap:
        items = list(sorted(experiment._2qbg_positions.items()))

        if not items:
            return {}

        sampled = random.sample(items, k=min(num_gates, len(items)))
        return dict(sorted(sampled))

    return selector

def make_select_first_gates(
    num_gates: int = 1,
) -> Callable[[MutableAnsatzExperiment], TwoQMap]:
    """
    Build a selector that returns the first tracked two-qubit gates in
    circuit-data order.

    Parameters
    ----------
    num_gates : int, optional
        Number of gates to return.

    Returns
    -------
    Callable[[MutableAnsatzExperiment], TwoQMap]
        Selector function compatible with `build_targeted_prune_plan`.
    """
    def selector(experiment: MutableAnsatzExperiment) -> TwoQMap:
        return take_first_n_gates(dict(experiment._2qbg_positions), num_gates)

    return selector


def make_select_last_gates(
    num_gates: int = 1,
) -> Callable[[MutableAnsatzExperiment], TwoQMap]:
    """
    Build a selector that returns the last tracked two-qubit gates in
    circuit-data order.

    Parameters
    ----------
    num_gates : int, optional
        Number of gates to return.

    Returns
    -------
    Callable[[MutableAnsatzExperiment], TwoQMap]
        Selector function compatible with `build_targeted_prune_plan`.
    """
    def selector(experiment: MutableAnsatzExperiment) -> TwoQMap:
        items = list(sorted(experiment._2qbg_positions.items()))
        if num_gates <= 0:
            raise ValueError("`num_gates` must be positive.")
        return dict(items[-num_gates:])

    return selector
