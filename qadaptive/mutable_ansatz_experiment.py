import logging
import numpy as np

from typing import Callable, SupportsFloat
from dataclasses import dataclass

from qiskit.circuit import QuantumCircuit, Parameter
from qiskit.transpiler import PassManager
from qiskit_algorithms.optimizers.optimizer import Optimizer, OptimizerResult

from qadaptive.adaptive_ansatz import AdaptiveAnsatz
from qadaptive.trainer import InnerLoopTrainer
from qadaptive.mutation import (
    get_two_qubit_gate_indices,
    get_pair_occurrence_from_circuit_index,
    get_circuit_index_from_pair_occurrence,
    is_locked_circuit_index,
    lock_circuit_index,
    get_locked_circuit_indices,
    update_locked_gates_on_insert,
    update_locked_gates_on_removal,
    get_two_qubit_gate_offsets,
    update_locked_gates_on_multiple_inserts,
    TwoQMap, LockId
)
from qadaptive.simplification import simplify_ansatz
from qadaptive.pruning import evaluate_two_qubit_gate_pruning

CALLBACK = Callable[[int, np.ndarray, float, SupportsFloat, bool], None]
TERMINATIONCHECKER = Callable[[int, np.ndarray, float, SupportsFloat, bool], bool]

logger = logging.getLogger(__name__)

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
    values : dict[str, float]
        Snapshot of the parameter memory at that point in the run.
    cost : float | None
        Objective value associated with this parameter state, if available.
    """

    outer_iteration: int
    action: str
    accepted: bool
    values: dict[str, float]
    cost: float | None = None

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
    parameter_memory : dict[str, float]
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
    parameter_memory: dict[str, float]
    parameter_memory_history: list[ParameterMemoryRecord]
    last_cost: float
    last_params: np.ndarray
    outer_iteration: int

class MutableAnsatzExperiment:
    """
    An experiment in which an ansatz is dynamically modified while training.

    This class wraps an `AdaptiveAnsatz` together with an `InnerLoopTrainer` and
    provides utilities for alternating between inner-loop parameter training and
    outer-loop structural updates such as gate insertion, block insertion,
    simplification, and pruning.
    
    Attributes
    ----------
    adaptive_ansatz : AdaptiveAnsatz
        Copy of the adaptive ansatz object managed by the experiment.
    ansatz : QuantumCircuit
        Current circuit representation of `adaptive_ansatz`.
    trainer : InnerLoopTrainer
        Inner-loop trainer responsible for parameter optimization and tracking
        optimizer-side state.
    cost_history : list[OptimizerResult] | None
        History of optimization results returned by the trainer after each
        `train_one_time` call, if tracking is enabled; otherwise None.
        _outer_iteration : int
        Internal counter for outer-loop structural updates.
    _2qbg_positions : TwoQMap
        Mapping from circuit-data indices of two-qubit gates to the qubit pairs
        they act on.
    locked_gates : set[LockId]
        Set of locked two-qubit gate identifiers used to prevent removal of
        selected gates during pruning.
    optimizer : Optimizer
        Optimizer currently used by the inner-loop trainer. Exposed as a property.
    gradient_history : list[np.ndarray] | None
        Gradient estimates stored by the trainer if gradient tracking is enabled;
        otherwise None. Exposed as a property.
    last_cost : float | None
        Most recent cost value stored by the trainer. Exposed as a property.
    last_params : np.ndarray | None
        Most recent parameter vector stored by the trainer. Exposed as a property.
    parameter_memory : dict[str, float]
        Persistent mapping from parameter names to their most recently accepted
        numerical values.
    parameter_memory_history :list[ParameterMemoryRecord]
        History of `parameter_memory` snapshots taken after each training run.
    outer_step_history : list[OuterStepResult]
        History of outer-loop step results summarizing the structural updates
        attempted throughout the experiment.
    """

    def __init__(
        self, 
        adaptive_ansatz: AdaptiveAnsatz,
        trainer: InnerLoopTrainer,
        track_costs: bool = True,
        ) -> None:
        """
        Initialize the MutableAnsatzExperiment.

        Parameters
        ----------
        adaptive_ansatz : AdaptiveAnsatz
            The adaptive ansatz to be optimized.
        trainer : InnerLoopTrainer
            An inner loop trainer object that handles the inner optimization.
        track_costs : bool, optional
            Indicates whether inner-loop cost history is being tracked. Defaults to True.
        """
        self.adaptive_ansatz = adaptive_ansatz.copy()
        self.ansatz: QuantumCircuit = self.adaptive_ansatz.get_current_ansatz()
        if trainer is None:
            raise ValueError("A trainer instance must be provided.")
        self.trainer = trainer
        self._outer_iteration = 0
        self.cost_history = [] if track_costs else None
        # Two qubit gate positions
        self._2qbg_positions = self._get_two_qubit_gate_indices()
        # Some 2 qubit gates will be important and thus get locked
        # Since the ansatz will be constantly changing, this is tracked by looking at the
        # n'th two qubit gate and the qubits it acts on. No gate is locked by default.
        self.locked_gates = set()
        self.parameter_memory: dict[str, float] = {}
        self.parameter_memory_history: list[ParameterMemoryRecord] = []
        self.outer_step_history: list[OuterStepResult] = []
        
    def set_optimizer(
        self, optimizer: Optimizer | None = None, optimizer_options: dict | None = None
        ) -> None:
        """
        Sets the optimizer for the experiment.

        This method configures the optimizer to be used in the variational experiment.
        The user can either provide a pre-initialized optimizer object or specify
        the options for the default SPSA (Simultaneous Perturbation Stochastic
        Approximation) optimizer.

        Parameters
        ----------
        optimizer : Optimizer | None, optional
            An instance of an Optimizer class. If provided, this optimizer will be
            used for the experiment. Default is None.
        optimizer_options : dict | None, optional
            A dictionary of options to initialize the SPSA optimizer. This is
            only used if the `optimizer` argument is None. Common keys include
            'maxiter', 'learning_rate', and 'perturbation'.

        Raises
        ------
        ValueError
            If both `optimizer` and `optimizer_options` are None.

        Examples
        --------
        >>> experiment = MutableAnsatzExperiment()
        >>> spsa_optimizer = SPSA(maxiter=500)
        >>> experiment.set_optimizer(optimizer=spsa_optimizer)

        >>> experiment = MutableAnsatzExperiment()
        >>> spsa_options = {'maxiter': 100, 'learning_rate': 0.01}
        >>> experiment.set_optimizer(optimizer_options=spsa_options)
        """
        
        self.trainer.set_optimizer(optimizer=optimizer, optimizer_options=optimizer_options)
        
    def reset_optimizer_callback(
        self,
        callback: CALLBACK
        ) -> None:
        """
        Set a callback function to be called at each iteration of the optimizer.
        Parameters
        ----------
        callback : CALLBACK
            A function with signature ``callback(iteration, params, cost, grad, accepted)``
            that will be called at each iteration of the optimizer. The parameters are:
                - eval_count: number of cost function evaluations
                - parameters: the current parameter vector
                - mean: the current cost function value
                - stp_size: step size used for the parameter update
                - accepted: whether the last parameter update was accepted by the optimizer
        """
        self.trainer.optimizer.callback = callback
        
    def reset_optimizer_iteration(self, iteration: int = 0) -> None:
        """
        Reset the optimizer's internal iteration counter.

        Parameters
        ----------
        iteration : int, optional
            The value to which the optimizer's iteration counter should be reset. Default is 0.
        """
        self.trainer.optimizer.last_iteration = iteration

    def get_latest_gradients(self) -> np.ndarray:
        """
        Get the latest gradient values from the optimizer's history.

        Returns
        -------
        np.ndarray
            The last gradient vector recorded by the optimizer, or an empty array if no gradients are available.
        """
        return (
            self.gradient_history[-1] if self.gradient_history 
            is not None and len(self.gradient_history) > 0 else np.array([])
        )

    def _update_params(self):
        """
        Update ansatz parameters using the optimizer.
        """
        self.adaptive_ansatz.update_params()
        
    def _ordered_current_parameters(self) -> list[Parameter]:
        """
        Return the current ansatz parameters in optimizer order.

        Returns
        -------
        list[Parameter]
            Ordered list of parameters currently present in the adaptive ansatz.

        Notes
        -----
        This helper defines the canonical parameter order used when converting
        between a name-based parameter memory and the dense parameter vector
        expected by the inner-loop trainer. The returned order should always match
        the order assumed by `train_one_time`.
        """
        return list(self.adaptive_ansatz.params)
    
    def _store_parameter_values(
        self,
        values: list[float] | np.ndarray,
        params: list[Parameter] | None = None,
    ) -> None:
        """
        Store numerical values for a set of parameters in `parameter_memory`.

        Parameters
        ----------
        values : list[float] | np.ndarray
            Numerical parameter values to store.
        params : list[Parameter] | None, optional
            Parameters associated with `values`. If `None`, the current ansatz
            parameters returned by `_ordered_current_parameters` are used.

        Raises
        ------
        ValueError
            If the number of provided values does not match the number of parameters.

        Notes
        -----
        This method is typically called after a successful inner-loop optimization
        run to cache the final trained parameter values. The stored mapping can then
        be reused to initialize the next training run after a structural ansatz
        update.

        Examples
        --------
        Store the final result of a completed training run:

        >>> result = experiment.train_one_time(loss_function, iterations=100)
        >>> experiment._store_parameter_values(result.x)
        or
        >>> experiment._store_parameter(experiment.last_params)
        """
        if params is None:
            params = self._ordered_current_parameters()

        values = np.asarray(values, dtype=float)

        if len(values) != len(params):
            raise ValueError(
                f"Got {len(values)} values for {len(params)} parameters."
            )

        for param, value in zip(params, values):
            self.parameter_memory[param.name] = float(value)

    def build_warm_start_initial_point(
        self,
        default_value_for_new_params: float = 0.0,
    ) -> np.ndarray:
        """
        Construct an initial parameter vector for the current ansatz from memory.

        Parameters
        ----------
        default_value_for_new_params : float, optional
            Value assigned to parameters that are present in the current ansatz but
            do not yet appear in `parameter_memory`. This is set to `0.0` by default
            so that newly inserted gates or blocks start from their identity or
            neutral initialization.

        Returns
        -------
        np.ndarray
            Initial-point vector ordered according to the current ansatz parameters.

        Notes
        -----
        The vector is constructed by matching each current parameter name against
        `parameter_memory`. Existing parameters inherit their most recently stored
        values, while new parameters receive `default_value_for_new_params`.

        This method is intended for warm-starting successive inner-loop training
        runs in the outer adaptive optimization loop.

        Examples
        --------
        Build a warm-start vector after inserting a new block:

        >>> x0 = experiment.build_warm_start_initial_point()
        >>> result = experiment.train_one_time(loss_function, initial_point=x0)
        """
        params = self._ordered_current_parameters()
        return np.asarray(
            [
                self.parameter_memory.get(param.name, default_value_for_new_params)
                for param in params
            ],
            dtype=float,
        )
        
    def get_current_parameter_dict(self, default_value_for_new_params: float = 0.0) -> dict[str, float]:
        """
        Return the current ansatz parameters as a name-to-value dictionary.

        Parameters
        ----------
        default_value_for_new_params : float, optional
            Value assigned to parameters that are present in the current ansatz but
            do not yet appear in `parameter_memory`.

        Returns
        -------
        dict[str, float]
            Dictionary whose keys are the names of the parameters currently present
            in the ansatz and whose values are taken from `parameter_memory` when
            available, or from `default_value_for_new_params` otherwise.

        Notes
        -----
        This method provides a dictionary view of the current parameter state
        projected onto the active ansatz only. Stored entries in `parameter_memory`
        for parameters that are no longer present in the current ansatz are ignored.

        It is mainly useful for debugging, inspection, serialization, or for
        understanding how the current ansatz would be initialized by
        `build_warm_start_initial_point`.

        Examples
        --------
        Inspect the currently active parameter values:

        >>> experiment.get_current_parameter_dict()

        After inserting a new block, newly introduced parameters appear with the
        specified default value unless they have already been assigned a stored
        value:

        >>> experiment.get_current_parameter_dict(default_value_for_new_params=0.0)
        """

        return {
            p.name: self.parameter_memory.get(p.name, default_value_for_new_params)
            for p in self._ordered_current_parameters()
        }
        
    def prune_parameter_memory_to_current_ansatz(self) -> None:
        """
        Remove stored parameter values that no longer belong to the current ansatz.

        This helper prunes `parameter_memory` so that it only contains entries for
        parameters that are still present in the current adaptive ansatz. Any stored
        values associated with parameters that were removed by gate deletion,
        pruning, rollback to an earlier structure, or transpiler-based
        simplification are discarded.

        Notes
        -----
        This method is mainly a housekeeping utility. It is not required for the
        warm-start mechanism to function correctly, because
        `build_warm_start_initial_point` already ignores parameter names that are
        not present in the current ansatz.
        """
        current_names = {p.name for p in self._ordered_current_parameters()}
        self.parameter_memory = {
            name: value
            for name, value in self.parameter_memory.items()
            if name in current_names
        }
    
    def _update_locked_gates_on_insert(self, circ_ind: int) -> None:
        """
        Update `locked_gates` after inserting a new two-qubit gate.

        Parameters
        ----------
        circ_ind : int
            Circuit-data index at which the new two-qubit gate was inserted.

        Notes
        -----
        This method assumes that the gate has already been inserted into
        `self.adaptive_ansatz.current_ansatz`, while `self._2qbg_positions`
        still contains the pre-insertion two-qubit bookkeeping.
        """
        old_two_q_map = dict(self._2qbg_positions)

        self.locked_gates = update_locked_gates_on_insert(
            circuit=self.adaptive_ansatz.current_ansatz,
            circ_ind=circ_ind,
            old_two_q_map=old_two_q_map,
            locked_gates=self.locked_gates,
        )
        
    def _update_locked_gates_on_removal(self, circ_ind: int) -> None:
        """
        Update `_2qbg_positions` and `locked_gates` after removing an unlocked
        two-qubit gate.

        Parameters
        ----------
        circ_ind : int
            Circuit-data index of the removed two-qubit gate.

        Notes
        -----
        This method assumes the gate has already been removed from the ansatz and
        uses the pre-removal `_2qbg_positions` stored in this object to update
        bookkeeping consistently.

        If the removed gate was locked, no update is performed.
        """
        old_two_q_map = dict(self._2qbg_positions)

        self.locked_gates = update_locked_gates_on_removal(
            circ_ind=circ_ind,
            old_two_q_map=old_two_q_map,
            locked_gates=self.locked_gates,
        )
                
    def _get_two_qubit_gate_indices(self) -> TwoQMap:
        """
        Return a mapping from circuit-data indices to the qubit pairs acted on by
        two-qubit gates in the current ansatz.

        The mapping follows the order of appearance in `self.ansatz.data`, and the
        pair ordering is preserved exactly as it appears in each instruction.

        Returns
        -------
        TwoQMap
            Dictionary whose keys are circuit-data indices and whose values are
            the corresponding two-qubit gate qubit pairs.
        """
        return get_two_qubit_gate_indices(self.adaptive_ansatz.current_ansatz)
    
    def _get_pair_occurrence_from_circuit_index(
        self,
        circ_index: int,
        two_q_map: TwoQMap | None = None,
    ) -> tuple[int, tuple[int, int]]:
        """
        Return the pair occurrence index and qubit pair for a two-qubit gate
        determined by its circuit-data index.

        Parameters
        ----------
        circ_index : int
            Circuit-data index of the two-qubit gate.
        two_q_map : TwoQMap | None, optional
            Mapping from circuit-data indices to qubit pairs. If None,
            `self._2qbg_positions` is used.

        Returns
        -------
        tuple[int, tuple[int, int]]
            A tuple `(occurrence_index, pair)` where `occurrence_index` is the
            number of previous appearances of `pair` at smaller circuit-data
            indices, and `pair` is the qubit pair acted on by the gate.

        Raises
        ------
        KeyError
            If `circ_index` is not the index of a tracked two-qubit gate.
        """
        if two_q_map is None:
            two_q_map = self._2qbg_positions

        return get_pair_occurrence_from_circuit_index(circ_index, two_q_map)
    
    def _get_circuit_index_from_pair_occurrence(
        self,
        occurrence: int,
        pair: tuple[int, int],
        two_q_map: TwoQMap | None = None,
    ) -> int | None:
        """
        Return the circuit-data index of the two-qubit gate identified by its
        pair-local occurrence index and qubit pair.

        Parameters
        ----------
        occurrence : int
            Pair-local occurrence index. For example, occurrence=1 means the
            second two-qubit gate acting on `pair`.
        pair : tuple[int, int]
            Qubit pair identifying the gate.
        two_q_map : TwoQMap | None, optional
            Mapping from circuit-data indices to qubit pairs. If None,
            `self._2qbg_positions` is used.

        Returns
        -------
        int | None
            Circuit-data index of the corresponding two-qubit gate, or None if no
            such gate exists.
        """
        if two_q_map is None:
            two_q_map = self._2qbg_positions
        return get_circuit_index_from_pair_occurrence(occurrence, pair, two_q_map)
    
    def _is_locked_circuit_index(self, circ_index: int) -> bool:
        """
        Return whether the two-qubit gate at the given circuit-data index is locked.

        Parameters
        ----------
        circ_index : int
            Circuit-data index of the gate to check.

        Returns
        -------
        bool
            True if the gate is tracked as locked, False otherwise.

        Raises
        ------
        KeyError
            If `circ_index` does not correspond to a tracked two-qubit gate.
        """
        return is_locked_circuit_index(circ_index, self._2qbg_positions, self.locked_gates)
    
    def _lock_circuit_index(self, circ_index: int) -> None:
        """
        Lock the two-qubit gate at the given circuit-data index.

        Parameters
        ----------
        circ_index : int
            Circuit-data index of the two-qubit gate to lock.

        Raises
        ------
        KeyError
            If `circ_index` does not correspond to a tracked two-qubit gate.
        """
        self.locked_gates = lock_circuit_index(
            circ_index,
            self._2qbg_positions,
            self.locked_gates,
        )
        
    def _get_locked_circuit_indices(self) -> list[int]:
        """
        Return the circuit-data indices of all currently locked two-qubit gates.

        Returns
        -------
        list[int]
            Sorted list of circuit-data indices corresponding to locked two-qubit
            gates that still exist in the current ansatz.
        """
        return get_locked_circuit_indices(self._2qbg_positions, self.locked_gates)

    def _update_ansatz(self) -> None:
        """Update the ansatz attribute."""
        self.ansatz = self.adaptive_ansatz.get_current_ansatz()
        self._update_params()
    
    def _sync_after_ansatz_change(self, reset_locked_gates: bool = False) -> None:
        """
        Synchronize experiment state after a structural modification of the ansatz.

        This updates:
            - self.ansatz
            - two-qubit gate positions
            - locked gate bookkeeping
        """
        # Refresh the ansatz reference
        self._update_ansatz()

        # Recompute 2Q gate positions
        self._2qbg_positions = self._get_two_qubit_gate_indices()

        if reset_locked_gates:
            self.locked_gates = set()
        
    def lock_gates(self, gates_to_lock: list[int]) -> None:
        """
        Lock two-qubit gates specified by their circuit-data indices.

        Parameters
        ----------
        gates_to_lock : list[int]
            Circuit-data indices of two-qubit gates to lock.

        Raises
        ------
        KeyError
            If any provided index does not correspond to a tracked two-qubit gate.
        """
        for circ_index in gates_to_lock:
            self._lock_circuit_index(circ_index)

    def train_one_time(
        self, 
        loss_function: Callable[[np.ndarray], float],
        initial_point: list | np.ndarray | None = None,
        loss_next: Callable[[np.ndarray], float] | None = None,
        iterations: int = 100,
        update_parameter_memory: bool = True,
        reuse_parameter_memory: bool = True,
        default_value_for_new_params: float = 0.0,
        **kwargs
        ) -> OptimizerResult:
        """
        Train the ansatz for a given number of iterations. This is part of the inner loop.

        Parameters
        ----------
        loss_function : Callable[[np.ndarray], float]
            The cost function to evaluate the ansatz.
        initial_point : list | np.ndarray | None, optional
            Initial parameter values for the optimization. If None, the optimizer's default.
        loss_next : Callable[[np.ndarray], float], optional
            An optional function to evaluate the objective at the next step.
        iterations : int
            The number of optimization steps.
        update_parameter_memory : bool, optional
            If `True`, the parameter values at the end of this training run are stored
            in `parameter_memory` for potential reuse in future runs. Defaults to `True`.
        reuse_parameter_memory : bool, optional
            If `True` and `initial_point` is `None`, construct the starting vector
            from `parameter_memory` using the current ansatz parameter names.
        default_value_for_new_params : float, optional
            Default value assigned to current parameters that are not yet present in
            `parameter_memory` when warm-starting a run.
        **kwargs
            Additional configuration. Currently supports:
            - `use_epochs`
            - `num_circs_per_group`
            - `num_circs_per_batch`

        Returns
        -------
        OptimizerResult
            The result of the optimization.
        """
        
        if initial_point is None and reuse_parameter_memory and self.parameter_memory:
            initial_point = self.build_warm_start_initial_point(
                default_value_for_new_params=default_value_for_new_params
            )
        
        current_ansatz = self.adaptive_ansatz.get_current_ansatz()
        result = self.trainer.train_one_time(
            ansatz=current_ansatz,
            loss_function=loss_function,
            initial_point=initial_point,
            loss_next=loss_next,
            iterations=iterations,
            **kwargs,
        )
        
        if self.cost_history is not None:
            self.cost_history.append(result)
            
        if update_parameter_memory:
            self._store_parameter_values(result.x)
            
        return result
    
    def _snapshot_state(self) -> ExperimentSnapshot:
        """
        Return a snapshot of the current mutable-experiment state.

        The snapshot is intended for outer-loop updates. Before
        attempting a structural ansatz modification, the experiment can store a
        snapshot and later restore it if the proposed modification is rejected.

        Returns
        -------
        ExperimentSnapshot
            Independent snapshot of the current experiment state.

        Notes
        -----
        Mutable fields are copied explicitly so that later changes to the live
        experiment do not affect the stored snapshot.

        The snapshot currently includes:
        - the current ansatz,
        - locked-gate bookkeeping,
        - two-qubit gate bookkeeping,
        - live parameter memory,
        - parameter-memory history,
        - the trainer's last accepted cost and parameter vector,
        - the outer-loop iteration counter.
        """
        return ExperimentSnapshot(
            ansatz=self.adaptive_ansatz.get_current_ansatz().copy(),
            locked_gates=set(self.locked_gates),
            two_q_map=dict(self._2qbg_positions),
            parameter_memory=dict(self.parameter_memory),
            parameter_memory_history=[
                ParameterMemoryRecord(
                    outer_iteration=record.outer_iteration,
                    action=record.action,
                    accepted=record.accepted,
                    values=dict(record.values),
                    cost=record.cost,
                )
                for record in self.parameter_memory_history
            ],
            last_cost=float(self.last_cost),
            last_params=np.asarray(self.last_params, dtype=float).copy(),
            outer_iteration=self._outer_iteration,
        )


    def _restore_state(self, snapshot: ExperimentSnapshot) -> None:
        """
        Restore the mutable-experiment state from a previously stored snapshot.

        Parameters
        ----------
        snapshot : ExperimentSnapshot
            Snapshot to restore.

        Notes
        -----
        This method is intended to roll back a rejected outer-loop proposal. It
        restores the ansatz, bookkeeping structures, parameter-memory state, and
        the trainer's cached last accepted evaluation.

        Mutable fields are copied again on restore so that the restored live state
        remains independent of the snapshot object.
        """
        self.adaptive_ansatz.update_ansatz(snapshot.ansatz.copy())
        self._sync_after_ansatz_change()

        self.locked_gates = set(snapshot.locked_gates)
        self._2qbg_positions = dict(snapshot.two_q_map)

        self.parameter_memory = dict(snapshot.parameter_memory)
        self.parameter_memory_history = [
            ParameterMemoryRecord(
                outer_iteration=record.outer_iteration,
                action=record.action,
                accepted=record.accepted,
                values=dict(record.values),
                cost=record.cost,
            )
            for record in snapshot.parameter_memory_history
        ]

        self._outer_iteration = snapshot.outer_iteration

        self.trainer.update_last_evaluation(
            cost=snapshot.last_cost,
            params=np.asarray(snapshot.last_params, dtype=float).copy(),
        )

    def insert_random(self) -> None:
        """
        Insert a new gate into the adaptive ansatz at the current step.
        """
        gate_name, qubits, index = self.adaptive_ansatz.add_random_gate()
        logger.info(f"Inserted {gate_name} gate on qubits {qubits} at position {index}.")
        
        self._sync_after_ansatz_change()
        if len(qubits) == 2:
            self._update_locked_gates_on_insert(index)
            logger.info("Updated 2Q positions after insertion: %s", self._2qbg_positions)
                    
    def insert_at(
        self, gate: str, qubits: list[int], circ_ind: int
        ) -> None:
        """
        Insert a new gate into the adaptive ansatz at the current step.

        Parameters
        ----------
        gate : str
            The type of gate to insert (e.g., 'rx', 'ry', 'cz').
        qubits : list[int]
            The qubits on which to apply the gate.
        circ_ind : int
            Index at which the operation should be added. If index i is chosen,
            the operation is places in position i.
        """
        assert gate in self.adaptive_ansatz.operator_pool, (
            f"Gate {gate} is not part of the available operator pool: {self.adaptive_ansatz.operator_pool}."
        )
        self.adaptive_ansatz.add_gate_at_index(gate, circ_ind, qubits)
        logger.info(f"Inserted {gate} gate on qubits {qubits} at position {circ_ind}.")
            
        self._sync_after_ansatz_change()
        if len(qubits) == 2:
            self._update_locked_gates_on_insert(circ_ind)
            logger.info(f"Updated ansatz. New 2 qubit gate positions are: {self._2qbg_positions}.")

    def remove_at(
        self, circ_ind: int
        ) -> None:
        """
        Remove gate at a given index in the adaptive ansatz.

        Parameters
        ----------
        circ_ind : int
            Position from where to remove the gate.
        """
        if circ_ind < 0 or circ_ind >= len(self.ansatz.data):
            raise IndexError(f"Circuit index {circ_ind} is out of range.")

        is_2qbg = len(self.ansatz.data[circ_ind].qubits) == 2

        # If this is a tracked 2Q gate and it is locked, do nothing.
        if is_2qbg and circ_ind in self._2qbg_positions:
            if self._is_locked_circuit_index(circ_ind):
                logger.info(
                    f"Attempted to remove locked two-qubit gate at circuit index "
                    f"{circ_ind}. No changes made."
                )
                return
            
        self.adaptive_ansatz.remove_gate_by_index(circ_ind)
        
        self._sync_after_ansatz_change()
        if is_2qbg:
            self._update_locked_gates_on_removal(circ_ind)
            logger.info("Updated 2Q positions after removal: %s", self._2qbg_positions)
        
    def insert_block_at(
        self,
        block_name: str,
        qubits: list[int],
        circ_ind: int,
    ) -> None:
        """
        Insert a block from the adaptive ansatz block pool.
        
        Parameters
        ----------
        block_name : str
            The name of the block to insert. Must be a key in `self.adaptive_ansatz.block_pool`.
        qubits : list[int]
            The qubits on which to apply the block. The length of this list must match the number 
            of qubits required by the block.
        circ_ind : int
            Circuit index at which the block should be added.
        
        Notes
        -----
        Two-qubit bookkeeping is updated incrementally for each two-qubit gate
        contained in the inserted block. Newly inserted two-qubit gates start
        unlocked, while existing locked-gate identifiers are shifted consistently.
        """
        assert block_name in self.adaptive_ansatz.block_pool, (
            f"Block {block_name} is not part of the available block pool: "
            f"{list(self.adaptive_ansatz.block_pool.keys())}."
        )
        
        block = self.adaptive_ansatz.block_pool[block_name]
        
        if len(qubits) != block.num_qubits:
            raise ValueError(
                f"Block '{block_name}' acts on {block.num_qubits} qubits, "
                f"but got {len(qubits)}."
            )
            
        # Build a temporary block circuit only to inspect its instruction structure.
        temp_params = [Parameter(f"_tmp_block_{i}") for i in range(block.num_parameters)]
        block_circuit = block.build(temp_params)
        two_q_offsets = get_two_qubit_gate_offsets(block_circuit)

        old_two_q_map = dict(self._2qbg_positions)

        self.adaptive_ansatz.add_block_at_index(block_name, circ_ind, qubits)
        logger.info(
            "Inserted block %s on qubits %s at position %s.",
            block_name,
            qubits,
            circ_ind,
        )
        
        if two_q_offsets:
            inserted_two_q_indices = [circ_ind + offset for offset in two_q_offsets]

            self.locked_gates = update_locked_gates_on_multiple_inserts(
                circuit=self.adaptive_ansatz.current_ansatz,
                inserted_indices=inserted_two_q_indices,
                old_two_q_map=old_two_q_map,
                locked_gates=self.locked_gates,
            )

        self._sync_after_ansatz_change()
        logger.info(f"Updated ansatz. New 2 qubit gate positions are: {self._2qbg_positions}.")

    def simplify_transpiler_passes(
        self,
        pass_manager: PassManager | None = None,
        repetitions: int = 2,
        reset_locks_on_ambiguity: bool = True,
    ) -> QuantumCircuit:
        """
        Simplify the current mutable ansatz with transpiler passes.

        Parameters
        ----------
        pass_manager : PassManager | None, optional
            Pass manager used for simplification.
        repetitions : int, optional
            Number of consecutive pass-manager applications.
        reset_locks_on_ambiguity : bool, optional
            If True, reset locked two-qubit gates whenever the 2Q mapping changes.

        Returns
        -------
        QuantumCircuit
            Simplified circuit.
        """
        result = simplify_ansatz(
            circuit=self.adaptive_ansatz.current_ansatz,
            pass_manager=pass_manager,
            repetitions=repetitions,
        )

        self.adaptive_ansatz.current_ansatz = result.circuit
        self._update_ansatz()
        self._2qbg_positions = result.new_two_q_map

        if not result.preserve_locked_gates and reset_locks_on_ambiguity:
            logger.info(
                "Resetting locked 2Q gates after simplification because the 2Q map changed."
            )
            self.locked_gates = set()

        return self.adaptive_ansatz.current_ansatz

    def prune_two_qubit_gate_attempt(
        self, 
        cost: Callable, 
        temperature: float = 0.08, 
        alpha: float = 0.1, 
        accept_tol: float = 0.2
        ) -> None:
        """
        Attempt to prune one non-locked two-qubit gate from the current ansatz.

        Parameters
        ----------
        cost : Callable
            Cost function with signature ``cost(params, ansatz)``.
        temperature : float, optional
            The temperature factor for Metropolis-like acceptance probability:
            .. math::
                p = exp(-\beta \frac{C_{new} - C_{o}}{C_o})
        alpha : float, optional
            Scaling factor for gate locking probability:
            .. math::
                P_{\text{lock}} = 1 - e^{-\alpha \frac{\Delta C}{|C_o|}}
        accept_tol : float, optional
            Tolerance for accepting the change. Defaults to 0.2.
                
        Notes
        -----
        This method evaluates the trial removal at the current trained parameter
        vector and then either applies the accepted pruning or locks the rejected
        gate according to the pruning policy.
        """
        assert len(self.last_params) > 0, (
            "Ansatz has not been trained yet. Train to set last parameters."
        )
        
        decision = evaluate_two_qubit_gate_pruning(
            ansatz=self.ansatz,
            two_q_map=self._2qbg_positions,
            locked_gates=self.locked_gates,
            last_params=self.last_params,
            last_cost=self.last_cost,
            cost=cost,
            is_locked=is_locked_circuit_index,
            temperature=temperature,
            alpha=alpha,
            accept_tol=accept_tol,
        )
        
        if not decision.attempted:
            return

        assert decision.gate_to_remove is not None

        if decision.accepted:
            self._update_locked_gates_on_removal(decision.gate_to_remove)
            self.adaptive_ansatz.update_ansatz(decision.trial_ansatz)
            self._sync_after_ansatz_change()
            self.trainer.update_last_evaluation(cost=decision.trial_cost)
            return

        if decision.should_lock:
            self._lock_circuit_index(decision.gate_to_remove)

        logger.info(f"Updated ansatz. New 2 qubit gate positions are: {self._2qbg_positions}.")
        
    def get_current_parameters(self) -> list[Parameter]:
        return self.adaptive_ansatz.get_current_ansatz().parameters
    
    def draw_current_ansatz(self) -> None:
        """
        Draw the current ansatz circuit.
        """
        self.adaptive_ansatz.get_current_ansatz().draw('mpl').show()   

    @property
    def optimizer(self):
        return self.trainer.optimizer

    @property
    def gradient_history(self):
        return self.trainer.gradient_history

    @property
    def last_cost(self):
        return self.trainer.last_cost

    @property
    def last_params(self):
        return self.trainer.last_params
