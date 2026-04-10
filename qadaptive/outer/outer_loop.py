from __future__ import annotations
import numpy as np

from typing import Any, Literal
from dataclasses import dataclass, field

from qiskit.circuit import QuantumCircuit, Parameter

from qadaptive.core.mutation import TwoQMap, LockId
from qadaptive.outer.action_definitions import ACTION_DEFINITIONS, ActionDefinition

AcceptanceMode = Literal["outer", "internal", "force"]

@dataclass
class OuterStepResult:
    """
    Result of one outer-loop step.

    This record summarizes what structural action was attempted, whether it was
    accepted, and how the objective and ansatz complexity changed.

    Attributes
    ----------
    iteration : int
        Outer-loop iteration index of the attempted step.
    action : str
        Name of the action that was attempted, for example ``"insert_block"``
        or ``"prune_two_qubit"``.
    accepted : bool
        Whether the proposed structural update was accepted.
    cost_before : float | None
        Objective value before the action and retraining.
    cost_after : float | None
        Objective value after the action and retraining.
    delta_cost : float | None
        Difference ``cost_after - cost_before``.
    num_parameters_before : int
        Number of ansatz parameters before the attempted step.
    num_parameters_after : int
        Number of ansatz parameters after the attempted step.
    num_two_qubit_before : int
        Number of tracked two-qubit gates before the attempted step.
    num_two_qubit_after : int
        Number of tracked two-qubit gates after the attempted step.
    note : str | None
        Optional comment describing special circumstances, for example
        ``"rejected and rolled back"``.
    """

    iteration: int
    action: str
    accepted: bool
    cost_before: float | None
    cost_after: float | None
    delta_cost: float | None
    num_parameters_before: int
    num_parameters_after: int
    num_two_qubit_before: int
    num_two_qubit_after: int
    note: str | None = None

@dataclass
class ParameterMemoryCache:
    """
    Cache of latest parameter values after a training run, intended for reuse in 
    future proposals.
    
    Attributes
    ----------
    parameters: list[Parameter]
        List of current ansatz parameters.
    parameter_names: list[str]
        List of parameter names.
    dictionary: dict[str, float]
        Mapping from parameter names to their latest values.
    """
    
    parameters: list[Parameter]
    parameter_names: list[str]
    dictionary: dict[str, float]

@dataclass
class ParameterMemoryRecord:
    """
    Record of the parameter-memory state after one inner-loop training run.

    This is a history object for later analysis. Unlike `parameter_memory`,
    which stores only the current reusable parameter cache, this record stores a
    snapshot of that cache together with metadata describing when and under what
    action it was produced.

    Attributes
    ----------
    outer_iteration : int
        Outer-loop iteration associated with this record.
    action : str
        Action associated with the training run that produced this parameter
        state, for example ``"initial_train"``, ``"insert_gate"``, or
        ``"prune_two_qubit"``.
    accepted : bool
        Whether the corresponding outer-loop proposal was ultimately accepted.
    values : ParameterMemoryCache
        Snapshot of the parameter memory at that point in the run.
    cost : float | None
        Objective value associated with this parameter state, if available.
    """

    outer_iteration: int
    action: str
    accepted: bool
    values: ParameterMemoryCache
    cost: float | None = None
    
@dataclass
class AcceptedAnsatzRecord:
    """
    Snapshot of one accepted trained ansatz state.
    """

    outer_iteration: int
    action: str
    cost: float | None
    num_parameters: int
    num_two_qubit_gates: int
    ansatz: QuantumCircuit
    parameter_values: dict[str, float]
    note: str | None = None

@dataclass
class ExperimentSnapshot:
    """
    Snapshot of the mutable experiment state.

    This record is intended for outer-loop steps: before applying
    a structural modification, the experiment can store a snapshot and later
    restore it if the proposed step is rejected.

    Attributes
    ----------
    ansatz : QuantumCircuit
        Copy of the current ansatz circuit.
    locked_gates : set[LockId]
        Set of currently locked two-qubit gates.
    two_q_map : TwoQMap
        Mapping from circuit-data indices to ordered qubit pairs for the current
        two-qubit gates.
    parameter_memory : ParameterMemoryCache
        Current live parameter-value cache.
    parameter_memory_history : list[dict[str, float]]
        History of stored parameter-memory states accumulated so far.
    last_cost : float
        Most recent accepted cost value recorded by the trainer.
    last_params : np.ndarray
        Most recent accepted parameter vector recorded by the trainer.
    outer_iteration : int
        Outer-loop iteration counter at the time of the snapshot.
    """

    ansatz: QuantumCircuit
    locked_gates: set[LockId]
    two_q_map: TwoQMap
    parameter_memory: ParameterMemoryCache
    parameter_memory_history: list[ParameterMemoryRecord]
    last_cost: float
    last_params: np.ndarray
    outer_iteration: int

@dataclass
class ActionSpec:
    """
    Concrete validated instance of one atomic outer-loop action.
    
    An `ActionSpec` is the smallest executable unit in the outer-loop
    proposal machinery. It stores the action name together with the keyword
    arguments required to execute that action through the experiment's action
    dispatcher.
    
    Attributes
    ----------
    action : str
        Name of the atomic action, for example ``"insert_block"``,
        ``"insert_gate"``, ``"simplify"``, or ``"prune_two_qubit"``.
    kwargs : dict[str, Any]
        Keyword arguments required by the action handler.
    label : str | None
        Optional human-readable label for debugging or logging.
    """

    action: str
    kwargs: dict[str, Any] = field(default_factory=dict)
    label: str | None = None

    def __post_init__(self) -> None:
        if self.action not in ACTION_DEFINITIONS:
            raise ValueError(
                f"Unknown action '{self.action}'. "
                f"Available actions are: {list(ACTION_DEFINITIONS)}."
            )

        definition = ACTION_DEFINITIONS[self.action]
        missing = [key for key in definition.required_kwargs if key not in self.kwargs]
        if missing:
            raise ValueError(
                f"Action '{self.action}' is missing required kwargs: {missing}."
            )

    @property
    def definition(self) -> ActionDefinition:
        """Return the action-definition metadata for this action."""
        return ACTION_DEFINITIONS[self.action]

    @property
    def display_name(self) -> str:
        """Return the label if available, otherwise the action name."""
        return self.label if self.label is not None else self.action

    @property
    def requires_cost(self) -> bool:
        """Return whether this action requires a cost callable."""
        return self.definition.requires_cost
    
@dataclass
class OuterStepPlan:
    """
    Proposal for one outer-loop step.

    An `OuterStepPlan` groups several atomic structural actions into a single
    proposal that will be executed before retraining and final acceptance or
    rejection. This allows the outer loop to express richer behaviors such as
    rapid early growth, insertion bursts followed by simplification, or
    pruning/simplification proposals.

    Attributes
    ----------
    name : str
        Human-readable name of the proposal, for example
        ``"star_growth_q0"`` or ``"prune_then_simplify"``.
    actions : list[ActionSpec]
        Ordered sequence of atomic actions to execute as part of the proposal.
    acceptance_mode : {"outer", "internal", "force}
        How acceptance should be handled for the plan.

        - ``"outer"`` means the plan is treated as a standard proposal:
          execute all actions, retrain, then let the outer-loop acceptance
          rule decide whether to keep or roll back the whole proposal.
        - ``"internal"`` means the plan contains action logic that already
          performs its own acceptance or rejection internally. This is useful
          for actions such as pruning attempts that currently implement their
          own Metropolis-like acceptance rule.
    label : str | None
        Optional descriptive label for logging, visualization, or debugging.
    """

    name: str
    actions: list[ActionSpec] = field(default_factory=list)
    acceptance_mode: AcceptanceMode = "outer"
    label: str | None = None

    def __post_init__(self) -> None:
        """Validate the outer-step plan."""
        if not self.name:
            raise ValueError("`name` must be a non-empty string.")

        if self.acceptance_mode not in ("outer", "internal", "force"):
            raise ValueError(
                "`acceptance_mode` must be either 'outer' or 'internal'."
            )

        if len(self.actions) == 0:
            raise ValueError("`actions` must contain at least one ActionSpec.")

    @property
    def display_name(self) -> str:
        """Return the label if available, otherwise the plan name."""
        return self.label if self.label is not None else self.name

    @property
    def action_names(self) -> list[str]:
        """Return the ordered list of atomic action names in the plan."""
        return [action.action for action in self.actions]

    def append(self, action: ActionSpec) -> None:
        """Append one atomic action to the plan."""
        self.actions.append(action)

    def extend(self, actions: list[ActionSpec]) -> None:
        """Append multiple atomic actions to the plan."""
        self.actions.extend(actions)
