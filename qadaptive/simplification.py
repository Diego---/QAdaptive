import logging
from dataclasses import dataclass

from qiskit import QuantumCircuit
from qiskit.transpiler import PassManager

from qadaptive.mutation import get_two_qubit_gate_indices
from qadaptive.utils import custom_pass_manager

logger = logging.getLogger("qadaptive.core.simplification")

TwoQMap = dict[int, tuple[int, int]]

@dataclass(frozen=True)
class SimplificationResult:
    """
    Result of applying transpiler-based simplification to an ansatz.
    """

    circuit: QuantumCircuit
    old_two_q_map: TwoQMap
    new_two_q_map: TwoQMap
    preserve_locked_gates: bool
    changed: bool


def build_pass_manager(
    pass_manager: PassManager | None = None,
) -> PassManager:
    """
    Return the transpilation pass manager to use for ansatz simplification.

    Parameters
    ----------
    pass_manager : PassManager | None, optional
        User-provided pass manager. If None, the default custom pass manager is used.

    Returns
    -------
    PassManager
        Pass manager used for simplification.
    """
    return custom_pass_manager() if pass_manager is None else pass_manager


def simplify_circuit_with_transpiler(
    circuit: QuantumCircuit,
    pass_manager: PassManager,
    repetitions: int = 2,
) -> QuantumCircuit:
    """
    Apply a transpiler pass manager repeatedly to a circuit.

    Parameters
    ----------
    circuit : QuantumCircuit
        Circuit to simplify.
    pass_manager : PassManager
        Pass manager to apply.
    repetitions : int, optional
        Number of consecutive applications.

    Returns
    -------
    QuantumCircuit
        Simplified circuit.
    """
    simplified = circuit.copy()

    for _ in range(repetitions):
        simplified = pass_manager.run(simplified)

    return simplified


def can_preserve_locked_gates_trivially(
    old_two_q_map: TwoQMap,
    new_two_q_map: TwoQMap,
) -> bool:
    """
    Return whether locked-gate bookkeeping can be preserved without ambiguity.

    The criterion is intentionally conservative: locked two-qubit gates are
    preserved only if the ordered sequence of tracked qubit pairs is unchanged
    before and after simplification.

    Parameters
    ----------
    old_two_q_map : TwoQMap
        Two-qubit gate map before simplification.
    new_two_q_map : TwoQMap
        Two-qubit gate map after simplification.

    Returns
    -------
    bool
        True if bookkeeping can be preserved trivially, False otherwise.
    """
    return two_q_pair_sequence(old_two_q_map) == two_q_pair_sequence(new_two_q_map)


def simplify_ansatz(
    circuit: QuantumCircuit,
    pass_manager: PassManager | None = None,
    repetitions: int = 2,
) -> SimplificationResult:
    """
    Simplify an ansatz circuit and report whether 2Q bookkeeping can be preserved.

    Parameters
    ----------
    circuit : QuantumCircuit
        Circuit to simplify.
    pass_manager : PassManager | None, optional
        Transpilation pass manager. If None, the default custom pass manager is used.
    repetitions : int, optional
        Number of consecutive pass-manager applications.

    Returns
    -------
    SimplificationResult
        Simplified circuit plus bookkeeping comparison metadata.
    """
    pm = build_pass_manager(pass_manager)

    old_two_q_map = get_two_qubit_gate_indices(circuit)
    simplified = simplify_circuit_with_transpiler(
        circuit=circuit,
        pass_manager=pm,
        repetitions=repetitions,
    )
    new_two_q_map = get_two_qubit_gate_indices(simplified)

    preserve_locked_gates = can_preserve_locked_gates_trivially(
        old_two_q_map=old_two_q_map,
        new_two_q_map=new_two_q_map,
    )

    changed = simplified != circuit

    logger.info("Simplified ansatz with transpiler passes.")
    logger.info("Old 2Q map: %s", old_two_q_map)
    logger.info("New 2Q map: %s", new_two_q_map)
    logger.info("Preserve locked gates trivially: %s", preserve_locked_gates)

    return SimplificationResult(
        circuit=simplified,
        old_two_q_map=old_two_q_map,
        new_two_q_map=new_two_q_map,
        preserve_locked_gates=preserve_locked_gates,
        changed=changed,
    )

def two_q_pair_sequence(two_q_map: TwoQMap) -> list[tuple[int, int]]:
    """
    Return the ordered sequence of qubit pairs corresponding to the tracked
    two-qubit gates in a circuit.

    The sequence is constructed by sorting the circuit-data indices in
    ``two_q_map`` and collecting the associated qubit pairs in that order.
    This provides a softer structural signature than the raw index-based map,
    since it preserves the left-to-right order of two-qubit interactions while
    ignoring absolute circuit-data positions.

    This representation is useful when comparing circuits before and after
    transformations that may shift instruction indices without changing the
    relative order of the tracked two-qubit gates.

    Parameters
    ----------
    two_q_map : TwoQMap
        Mapping from circuit-data indices to the corresponding ordered qubit
        pairs of tracked two-qubit gates.

    Returns
    -------
    list[tuple[int, int]]
        Ordered list of tracked qubit pairs as they appear in the circuit.

    Examples
    --------
    >>> two_q_map = {2: (0, 1), 5: (1, 2), 9: (0, 1)}
    >>> two_q_pair_sequence(two_q_map)
    [(0, 1), (1, 2), (0, 1)]
    """
    return [two_q_map[idx] for idx in sorted(two_q_map)]
