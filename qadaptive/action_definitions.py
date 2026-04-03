from dataclasses import dataclass

INSERT_RANDOM_GATE = "insert_random_gate"
INSERT_GATE = "insert_gate"
INSERT_BLOCK = "insert_block"
REMOVE_GATE = "remove_gate"
SIMPLIFY = "simplify"
PRUNE_TWO_QUBIT = "prune_two_qubit"


@dataclass(frozen=True)
class ActionDefinition:
    name: str
    required_kwargs: tuple[str, ...] = ()
    requires_cost: bool = False


ACTION_DEFINITIONS: dict[str, ActionDefinition] = {
    INSERT_RANDOM_GATE: ActionDefinition(
        name=INSERT_RANDOM_GATE,
    ),
    INSERT_GATE: ActionDefinition(
        name=INSERT_GATE,
        required_kwargs=("gate", "qubits", "circ_ind"),
    ),
    INSERT_BLOCK: ActionDefinition(
        name=INSERT_BLOCK,
        required_kwargs=("block_name", "qubits", "circ_ind"),
    ),
    REMOVE_GATE: ActionDefinition(
        name=REMOVE_GATE,
        required_kwargs=("circ_ind",),
    ),
    SIMPLIFY: ActionDefinition(
        name=SIMPLIFY,
    ),
    PRUNE_TWO_QUBIT: ActionDefinition(
        name=PRUNE_TWO_QUBIT,
        requires_cost=True,
    ),
}