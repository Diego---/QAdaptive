from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Callable, Literal, Sequence

from qiskit import QuantumCircuit
from qiskit.circuit import Parameter


BlockBuilder = Callable[[list[Parameter]], QuantumCircuit]

RotationGate = Literal["rx", "ry", "rz"]
FixedEntangler = Literal["cx", "cz"]
ParametrizedEntangler = Literal["rxx", "ryy", "rzz"]
GateName = RotationGate | FixedEntangler | ParametrizedEntangler

# An operation is (gate_name, qubits), e.g. ("rx", (0,)) or ("cz", (0, 1)).
Op = tuple[GateName, tuple[int, ...]]


# Gate name -> number of qubits it acts on.
_PARAMETRIZED_GATE_ARITY: dict[ParametrizedEntangler | RotationGate, int] = {
    "rx": 1,
    "ry": 1,
    "rz": 1,
    "rxx": 2,
    "ryy": 2,
    "rzz": 2,
}

_FIXED_GATE_ARITY: dict[FixedEntangler, int] = {
    "cx": 2,
    "cz": 2,
}


@dataclass(frozen=True)
class PoolBlock:
    """
    A parametrized circuit block that can be inserted into an AdaptiveAnsatz.

    Attributes
    ----------
    name : str
        Name of the block.
    num_qubits : int
        Number of qubits the block acts on.
    num_parameters : int
        Number of free parameters introduced by the block.
    builder : BlockBuilder
        Function that returns a QuantumCircuit implementing the block when
        given a list of fresh parameters.
    """

    name: str
    num_qubits: int
    num_parameters: int
    builder: BlockBuilder

    def build(self, params: list[Parameter]) -> QuantumCircuit:
        """
        Build the block circuit from a list of parameters.

        Parameters
        ----------
        params : list[Parameter]
            Parameters to use in the block.

        Returns
        -------
        QuantumCircuit
            Circuit implementing the block.

        Raises
        ------
        ValueError
            If the number of provided parameters does not match the number
            expected by the block.
        """
        if len(params) != self.num_parameters:
            raise ValueError(
                f"Block '{self.name}' expects {self.num_parameters} parameters, "
                f"but got {len(params)}."
            )

        return self.builder(params)


# Generic builder
def _build_from_ops(
    params: list[Parameter],
    *,
    num_qubits: int,
    ops: tuple[Op, ...],
    name: str,
) -> QuantumCircuit:
    """
    Build a circuit from an operation specification.

    Parametrized gates consume one parameter each, in the order in which they
    appear in ``ops``. Fixed gates consume no parameters.

    Parameters
    ----------
    params : list[Parameter]
        Parameters consumed by parametrized gates in operation order.
    num_qubits : int
        Number of qubits in the resulting circuit.
    ops : tuple[Op, ...]
        Ordered operation specification.
    name : str
        Name assigned to the resulting circuit.

    Returns
    -------
    QuantumCircuit
        Circuit implementing the operation specification.
    """
    qc = QuantumCircuit(num_qubits, name=name)
    param_iter = iter(params)

    for gate, qubits in ops:
        method = getattr(qc, gate)

        if gate in _PARAMETRIZED_GATE_ARITY:
            method(next(param_iter), *qubits)
        else:
            method(*qubits)

    return qc


def make_block(
    name: str,
    num_qubits: int,
    ops: Sequence[Op],
) -> PoolBlock:
    """
    Create a PoolBlock from an operation specification.

    The operation specification is validated at construction time, and the
    number of free parameters is derived automatically from the parametrized
    gates appearing in ``ops``.

    Parameters
    ----------
    name : str
        Name of the block. The same name is assigned to the built circuit.
    num_qubits : int
        Number of qubits the block acts on.
    ops : Sequence[Op]
        Ordered sequence of ``(gate_name, qubits)`` operations.

    Returns
    -------
    PoolBlock
        Validated block with a picklable ``functools.partial`` builder.

    Raises
    ------
    ValueError
        If ``num_qubits`` is invalid, a gate is unknown, the number of qubits
        supplied to a gate does not match its arity, a gate acts repeatedly on
        the same qubit, or a qubit index is out of range.
    """
    if num_qubits < 1:
        raise ValueError(
            f"Block '{name}': num_qubits must be at least 1, got {num_qubits}."
        )

    normalized_ops: tuple[Op, ...] = tuple(
        (gate, tuple(qubits)) for gate, qubits in ops
    )

    for gate, qubits in normalized_ops:
        arity = _PARAMETRIZED_GATE_ARITY.get(
            gate, _FIXED_GATE_ARITY.get(gate)
        )

        if arity is None:
            raise ValueError(f"Block '{name}': unknown gate '{gate}'.")

        if len(qubits) != arity or len(set(qubits)) != arity:
            raise ValueError(
                f"Block '{name}': gate '{gate}' needs {arity} distinct "
                f"qubit(s), got {qubits}."
            )

        if any(q < 0 or q >= num_qubits for q in qubits):
            raise ValueError(
                f"Block '{name}': qubits {qubits} out of range for "
                f"{num_qubits} qubit(s)."
            )

    num_parameters = sum(
        gate in _PARAMETRIZED_GATE_ARITY
        for gate, _ in normalized_ops
    )

    return PoolBlock(
        name=name,
        num_qubits=num_qubits,
        num_parameters=num_parameters,
        builder=partial(
            _build_from_ops,
            num_qubits=num_qubits,
            ops=normalized_ops,
            name=name,
        ),
    )


# Operation-spec helpers
def rotations(
    gates: Sequence[RotationGate],
    qubit: int,
) -> list[Op]:
    """
    Create a sequence of single-qubit rotations acting on one qubit.

    Each gate in ``gates`` becomes one operation on ``qubit``, in the given
    order. Each rotation consumes one parameter when the block is built.

    Parameters
    ----------
    gates : Sequence[RotationGate]
        Rotation gates to apply, in order, e.g. ``("rx", "ry")``.
    qubit : int
        Index of the qubit the rotations act on.

    Returns
    -------
    list[Op]
        Ordered operation specification, one ``(gate, (qubit,))`` entry
        per gate.

    Examples
    --------
    >>> rotations(("rx", "ry"), 0)
    [('rx', (0,)), ('ry', (0,))]
    """
    return [(gate, (qubit,)) for gate in gates]


def sandwich(
    entangler: FixedEntangler,
    gates: Sequence[RotationGate],
    *,
    control: int = 0,
    target: int = 1,
    middle_qubit: int | None = None,
    close: bool = True,
) -> list[Op]:
    """
    Create a fixed-entangler sandwich.

    The operation sequence is

        rotations on ``control``
        entangler(control, target)
        rotations on ``middle_qubit``
        [entangler(control, target)]

    where ``middle_qubit`` defaults to ``target`` and the final entangler is
    included when ``close=True``.

    For self-inverse entanglers such as CX and CZ, ``close=True`` makes the
    block exactly identity when all rotation parameters are zero.

    With ``close=False``, the remaining bare entangler is generally not the
    identity unitary. It may nevertheless preserve a chosen reference state;
    for example, CX and CZ both leave |00> unchanged.

    Parameters
    ----------
    entangler : FixedEntangler
        Fixed two-qubit entangling gate.
    gates : Sequence[RotationGate]
        Rotation gates applied before and after the entangler.
    control : int, default=0
        First qubit of the entangling gate and qubit receiving the first
        rotation sequence.
    target : int, default=1
        Second qubit of the entangling gate.
    middle_qubit : int or None, default=None
        Qubit receiving the second rotation sequence. If ``None``, ``target``
        is used.
    close : bool, default=True
        Whether to append the fixed entangler a second time.

    Returns
    -------
    list[Op]
        Ordered operation specification.
    """
    middle = target if middle_qubit is None else middle_qubit

    ops: list[Op] = [
        (entangler, (control, target)),
        *rotations(gates, control),
        *rotations(gates, middle),
    ]

    if close:
        ops.append((entangler, (control, target)))

    return ops


# Default pool
_RXRY: tuple[RotationGate, ...] = ("rx", "ry")
_RYRZ: tuple[RotationGate, ...] = ("ry", "rz")

_BLOCKS: tuple[PoolBlock, ...] = (
    # One-qubit Euler block; identity when all angles are zero.
    make_block(
        "rz_rx_rz",
        1,
        rotations(("rz", "rx", "rz"), 0),
    ),

    # Identity-initializable two-qubit blocks.
    make_block(
        "cx_identity",
        2,
        sandwich("cx", _RYRZ),
    ),
    make_block(
        "cz_identity",
        2,
        sandwich("cz", _RXRY),
    ),
    make_block(
        "cz_identity_rotation_on_one_qubit",
        2,
        sandwich("cz", _RXRY, middle_qubit=0),
    ),

    # Identity-initializable three-qubit blocks.
    #
    # At zero rotation angles, the entangler sequence is
    #
    #     E(0, 1) E(0, 2) E(0, 1) E(0, 2),
    #
    # which reduces to identity for the CX and CZ choices below.
    make_block(
        "big_cx_identity",
        3,
        [
            *rotations(_RYRZ, 0),
            ("cx", (0, 1)),
            *rotations(_RYRZ, 0),
            ("cx", (0, 2)),
            *rotations(_RYRZ, 0),
            ("cx", (0, 1)),
            *rotations(_RYRZ, 0),
            ("cx", (0, 2)),
            *rotations(_RYRZ, 0),
        ],
    ),
    make_block(
        "big_cz_identity",
        3,
        [
            *rotations(_RXRY, 0),
            ("cz", (0, 1)),
            *rotations(_RXRY, 0),
            ("cz", (0, 2)),
            *rotations(_RXRY, 0),
            ("cz", (0, 1)),
            *rotations(_RXRY, 0),
            ("cz", (0, 2)),
            *rotations(_RXRY, 0),
        ],
    ),

    # Single fixed-entangler blocks.
    #
    # These are not identity unitaries at zero angles: the block reduces to a
    # bare CX or CZ. They nevertheless preserve |00>, so they can be inserted
    # at the beginning of a VQE ansatz initialized in the computational zero
    # without changing that reference state.
    make_block(
        "single_cx_block",
        2,
        sandwich(
            "cx", 
            ("rz", "ry"), 
            close=False
        ),
    ),
    make_block(
        "single_cz_block",
        2,
        sandwich(
            "cz",
            ("ry", "rx"),
            close=False,
        ),
    ),

    # Blocks with a parametrized two-qubit rotation. All gates reduce to
    # identity at zero angle, so the complete blocks are identity-initializable.
    make_block(
        "single_rxx_block",
        2,
        [
            ("rz", (0,)),
            ("rxx", (0, 1)),
            ("rz", (0,)),
        ],
    ),
    make_block(
        "single_rzz_block",
        2,
        [
            ("rx", (0,)),
            ("rzz", (0, 1)),
            ("rx", (0,)),
        ],
    ),
)


DEFAULT_BLOCK_POOL: dict[str, PoolBlock] = {
    block.name: block for block in _BLOCKS
}
