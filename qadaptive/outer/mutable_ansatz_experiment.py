import logging
import numpy as np

from typing import Callable, SupportsFloat

from qiskit.circuit import QuantumCircuit, Parameter
from qiskit.transpiler import PassManager
from qiskit_algorithms.optimizers.optimizer import Optimizer, OptimizerResult

from qadaptive.core.adaptive_ansatz import AdaptiveAnsatz
from qadaptive.training.trainer import InnerLoopTrainer
from qadaptive.core.mutation import (
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
    TwoQMap
)
from qadaptive.core.simplification import simplify_ansatz
from qadaptive.core.pruning import evaluate_two_qubit_gate_pruning
from qadaptive.outer.action_definitions import (
    INSERT_RANDOM_GATE,
    INSERT_GATE,
    INSERT_BLOCK,
    REMOVE_GATE,
    SIMPLIFY,
    PRUNE_TWO_QUBIT,
)
from qadaptive.outer.outer_loop import (
    OuterStepResult,
    ParameterMemoryRecord,
    ParameterMemoryCache,
    AcceptedAnsatzRecord,
    ExperimentSnapshot,
    ActionSpec,
    OuterStepPlan
)
from qadaptive.training.optimizers import create_callback_args

CALLBACK = Callable[[int, np.ndarray, float, SupportsFloat, bool], None]
TERMINATIONCHECKER = Callable[[int, np.ndarray, float, SupportsFloat, bool], bool]

logger = logging.getLogger(__name__)

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
    result_history : list[OptimizerResult] | None
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
    parameter_memory : ParameterMemoryCache
        Parameter cache of most recently completed training run, intended for 
        warm-starting subsequent runs after structural changes.
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
        track_results: bool = True,
        ) -> None:
        """
        Initialize the MutableAnsatzExperiment.

        Parameters
        ----------
        adaptive_ansatz : AdaptiveAnsatz
            The adaptive ansatz to be optimized.
        trainer : InnerLoopTrainer
            An inner loop trainer object that handles the inner optimization.
        track_results : bool, optional
            Indicates whether inner-loop results history is being tracked. Defaults to True.
            
        Raises
        ------
        ValueError
            If `trainer` is None.
        """
        self.adaptive_ansatz = adaptive_ansatz.copy()
        self.ansatz: QuantumCircuit = self.adaptive_ansatz.get_current_ansatz()
        if trainer is None:
            raise ValueError("A trainer instance must be provided.")
        self.trainer = trainer
        self._outer_iteration = 0
        self.result_history = [] if track_results else None
        # Two qubit gate positions
        self._2qbg_positions = self._get_two_qubit_gate_indices()
        # Some 2 qubit gates will be important and thus get locked
        # Since the ansatz will be constantly changing, this is tracked by looking at the
        # n'th two qubit gate and the qubits it acts on. No gate is locked by default.
        self.locked_gates = set()
        self.parameter_memory: ParameterMemoryCache = ParameterMemoryCache(parameters=[], parameter_names=[], dictionary={})
        self.parameter_memory_history: list[ParameterMemoryRecord] = []
        self.outer_step_history: list[OuterStepResult] = []
        self.accepted_ansatz_history: list[AcceptedAnsatzRecord] = []
        
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
        self.trainer.optimizer.reset_runtime_state(iteration)
        
    def _reset_inner_loop_callback(
        self,
        callback_builder: Callable[..., CALLBACK] | None,
        **callback_kwargs,
    ) -> None:
        """Rebuild the optimizer callback and reset optimizer iteration for the next inner-loop run."""
        if callback_builder is None:
            return

        callback_args = create_callback_args(
            json_file_path=callback_kwargs.get("json_file_path"),
            store_data=callback_kwargs.get("store_data", False),
            extra_eval_freq=callback_kwargs.get("extra_eval_freq"),
            cost_extra=callback_kwargs.get("cost_extra"),
            plot=callback_kwargs.get("plot", True),
            use_epoch=callback_kwargs.get("use_epoch", False),
        )

        new_callback = callback_builder(**callback_args)
        self.reset_optimizer_callback(new_callback)

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
        
    def _current_parameters(self) -> list[Parameter]:
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
    
    def _past_parameter_dictionary(self, iteration: int | None = None) -> dict[str, float] | None:
        """
        Return the parameter dictionary from a ParameterMemoryCache corresponding to a 
        past training iteration.

        Parameters
        ----------
        iteration : int | None, optional
            The index of the past training iteration for which to retrieve the parameter list.
            If None, the current index - 1 is used, corresponding to the most recent completed training run. 
            Default is None.

        Returns
        -------
        list[Parameter] | None
            The list of parameters that were active during the specified training iteration, 
            or None if the iteration index is out of bounds.

        Notes
        -----
        This method allows retrieval of the parameter configuration associated with a specific 
        past training run, as recorded in `parameter_memory_history`. 
        """
        
        if iteration is None:
            iteration = self._outer_iteration - 1
        
        if 0 <= iteration < len(self.parameter_memory_history):
            cache = self.parameter_memory_history[iteration].values
            return cache.dictionary
        else:
            logger.warning(
                "Requested past parameter dict for iteration %d, but only %d records are available.",
                iteration,
                len(self.parameter_memory_history),
            )
            return None
    
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
            parameters returned by `_current_parameters` are used.

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
        """
        if params is None:
            params = self._current_parameters()

        values = np.asarray(values, dtype=float)

        if len(values) != len(params):
            raise ValueError(
                f"Got {len(values)} values for {len(params)} parameters."
            )
            
        param_dict = {param.name: float(value) for param, value in zip(params, values)}
        
        cache = ParameterMemoryCache(
            parameters=params,
            parameter_names=[param.name for param in params],
            dictionary=param_dict
        )
        
        self.parameter_memory = cache
            
    def _record_parameter_memory(
        self,
        action: str,
        accepted: bool,
        cost: float | None = None,
    ) -> None:
        """
        Append the current live parameter memory to `parameter_memory_history`.

        Parameters
        ----------
        action : str
            Label describing the training context associated with the current
            parameter-memory state, for example ``"initial_train"``,
            ``"insert_block"``, or ``"prune_two_qubit"``.
        accepted : bool
            Whether the outer-loop proposal associated with this parameter state was
            ultimately accepted.
        cost : float | None, optional
            Objective value associated with the current parameter state. If None,
            the value is left unspecified in the stored record.

        Notes
        -----
        This method records a snapshot of the current live `parameter_memory` for
        later analysis. The stored dictionary is copied so that subsequent updates
        to the live cache do not modify the historical record.

        This history is intended for diagnostic and analysis purposes. Unlike
        `parameter_memory`, which stores only the current reusable parameter cache,
        `parameter_memory_history` preserves the sequence of parameter states
        encountered during the outer-loop workflow.
        """
        self.parameter_memory_history.append(
            ParameterMemoryRecord(
                outer_iteration=self._outer_iteration,
                action=action,
                accepted=accepted,
                values=self.parameter_memory,
                cost=cost,
            )
        )
        
    def _record_accepted_ansatz(
        self,
        action: str,
        cost: float | None,
        note: str | None = None,
    ) -> None:
        """Append the current accepted trained ansatz to history."""
        self.accepted_ansatz_history.append(
            AcceptedAnsatzRecord(
                outer_iteration=self._outer_iteration,
                action=action,
                cost=cost,
                num_parameters=self.ansatz.num_parameters,
                num_two_qubit_gates=len(self._2qbg_positions),
                ansatz=self.ansatz.copy(),
                parameter_values=self.get_current_parameter_dict(),
                note=note,
            )
        )

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
        """
        current_param_names = [param.name for param in self._current_parameters()]
        # Past param names are in the parameter memory cache
        
        logger.debug(
            "Old parameter names: %s \nNew parameter names: %s",
            self.parameter_memory.parameter_names,
            current_param_names,
        )
        
        logger.info(
            "Building warm-start initial point for %d parameters, of which %d are new.",
            len(current_param_names),
            sum(1 for name in current_param_names if name not in self.parameter_memory.parameter_names),
        )
        
        return np.asarray(
            [
                self.parameter_memory.dictionary.get(param_name, default_value_for_new_params)
                for param_name in current_param_names
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
        """

        return {
            p.name: self.parameter_memory.dictionary.get(p.name, default_value_for_new_params)
            for p in self._current_parameters()
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
        current_names = {p.name for p in self._current_parameters()}
        param_dict = {
            name: value
            for name, value in self.parameter_memory.dictionary.items()
            if name in current_names
        }
        
        self.parameter_memory = ParameterMemoryCache(
            parameters=[p for p in self._current_parameters() if p.name in current_names],
            parameter_names=[p.name for p in self._current_parameters() if p.name in current_names],
            dictionary=param_dict
        )
    
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
        trainer_iteration_reset: int | None = None,
        record_run_history: bool = False,
        store_initial_value_in_history: bool = False,
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
        trainer_iteration_reset : int | None, optional
            If not None, reset the optimizer iteration counter to this value before
            training begins.
        record_run_history : bool, optional
            If True, ask the trainer to store a `TrainingRunRecord` for this run.
        store_initial_value_in_history : bool, optional
            If True and `record_run_history=True`, evaluate the loss at the initial
            point and store it in the resulting training-run record.
        **kwargs
            Additional configuration forwarded to the trainer and objective.

        Returns
        -------
        OptimizerResult
            The result of the optimization.
        """
        
        current_ansatz = self.adaptive_ansatz.get_current_ansatz()

        if initial_point is None:
            initial_point = [np.random.choice([-1.0, 1.0]) for _ in range(current_ansatz.num_parameters)]

        initial_point_array = np.asarray(initial_point, dtype=float)

        if len(initial_point_array) != current_ansatz.num_parameters:
            raise ValueError(
                f"Provided `initial_point` has length {len(initial_point_array)}, "
                f"but the current ansatz has {current_ansatz.num_parameters} parameters."
            )

        initial_value = None
        if record_run_history and store_initial_value_in_history:
            objective_for_initial = loss_function if loss_next is None else loss_next
            try:
                initial_value = float(
                    objective_for_initial(initial_point_array, ansatz=current_ansatz, **kwargs)
                )
            except TypeError:
                initial_value = float(objective_for_initial(initial_point_array, current_ansatz))

        result = self.trainer.train_one_time(
            ansatz=current_ansatz,
            loss_function=loss_function,
            initial_point=initial_point_array,
            loss_next=loss_next,
            iterations=iterations,
            iteration_start=trainer_iteration_reset,
            record_run_history=record_run_history,
            initial_value=initial_value,
            **kwargs,
        )
        
        if self.result_history is not None:
            self.result_history.append(result)
            
        if update_parameter_memory:
            self._store_parameter_values(result.x)
            logger.debug(
                "Updated parameter memory with %d active parameters.",
                len(self.parameter_memory.parameter_names),
                )
            
        logger.debug(
            "Completed training with final cost %.10f and parameters %s"
            " in %d inner iterations.",
            result.fun,
            result.x,
            self.trainer._last_num_iterations,
            )
            
        return result
    
    def run_outer_step(
        self,
        loss_function: Callable[[np.ndarray], float],
        plan: OuterStepPlan,
        train_iterations: int = 100,
        initial_point_generator: Callable[..., np.ndarray] | None = None,
        loss_next: Callable[[np.ndarray], float] | None = None,
        train_after_plan: bool = True,
        trainer_iteration_reset: int | None = 0,
        callback_builder: Callable[..., CALLBACK] | None = None,
        callback_kwargs: dict | None = None,
        update_parameter_memory: bool = True,
        reuse_parameter_memory: bool = False,
        default_value_for_new_params: float = 0.0,
        record_parameter_memory: bool = True,
        record_run_history: bool = False,
        store_initial_value_in_history: bool = False,
        accept_tol: float = 0.0,
        complexity_penalty: Callable[[QuantumCircuit], float] | None = None,
        **train_kwargs,
    ) -> OuterStepResult:
        """
        Execute one outer-loop proposal, optionally retrain, and record the result.

        Parameters
        ----------
        loss_function : Callable[[np.ndarray], float]
            Objective function used for training and, when needed, for evaluating
            the current ansatz.
        plan : OuterStepPlan
            Concrete structural proposal to execute.
        train_iterations : int, optional
            Number of inner-loop optimization steps after executing the plan.
        initial_point_generator : Callable[..., np.ndarray] | None, optional
            Optional factory for generating the initial point for each training phase.
            If provided, this is called at each outer step with the current experiment
            state and the reconciled `initial_point` to produce the actual initial
            point used for training. The signature of the generator should be
            `generator(iteration: int) -> np.ndarray`.
        loss_next : Callable[[np.ndarray], float] | None, optional
            Optional objective for next-step evaluation during training.
        train_after_plan : bool, optional
            Whether to retrain after executing the plan.
        trainer_iteration_reset : int | None,  optional
            Value to which the optimizer's iteration counter is reset after retraining.
            If None, the counter is not reset. Default is 0.
        callback_builder : Callable[..., CALLBACK] | None, optional
            Factory used to rebuild the optimizer callback after each retraining phase.
            This is useful when live-plot callbacks should be reset between outer-loop
            steps so that each inner-loop run starts with a fresh plot/history.
        callback_kwargs : dict | None, optional
            Keyword arguments forwarded to `callback_builder` when rebuilding the callback.
        update_parameter_memory : bool, optional
            Whether to update the live parameter cache after retraining.
        reuse_parameter_memory : bool, optional
            Whether to build the retraining initial point from `parameter_memory`
            when `initial_point` is not provided.
        default_value_for_new_params : float, optional
            Default value for parameters not yet present in `parameter_memory`.
        record_parameter_memory : bool, optional
            Whether to append a parameter-memory record for this attempted step.
        record_run_history : bool, optional
            Whether to ask the trainer to store a `TrainingRunRecord` for the retraining phase
            of this step.
        store_initial_value_in_history : bool, optional
            If True and `record_run_history=True`, evaluate the loss at the initial
            point and store it in the resulting training-run record for the retraining phase.
        accept_tol : float, optional
            Required score improvement threshold for generic outer acceptance.
        complexity_penalty : Callable[[QuantumCircuit], float] | None, optional
            Optional penalty added to the objective when deciding acceptance in
            `acceptance_mode="outer"`.
        **train_kwargs
            Additional keyword arguments forwarded to `train_one_time`.

        Returns
        -------
        OuterStepResult
            Summary of the attempted outer-loop step.

        Raises
        ------
        ValueError
            If the current ansatz has not been trained yet and the selected
            acceptance mode requires a valid baseline.
            If `train_after_plan` is False while `plan.acceptance_mode` is "outer", since
            a cost evaluation after the plan execution is required for generic outer acceptance.
            If `initial_point_generator` is not None but does not have the correct signature.
        
        """
        snapshot = self._snapshot_state()

        ansatz_before = snapshot.ansatz.copy()
        num_parameters_before = ansatz_before.num_parameters
        num_two_qubit_before = len(snapshot.two_q_map)

        if self.outer_step_history:
            cost_before = float(snapshot.last_cost)
        else:
            logger.info("No previous training history; first outer step always gets accepted if acceptance_mode='outer'.")
            cost_before = None
        
        logger.info(
            "Starting outer step %d with plan '%s' (%d actions, acceptance_mode=%s).",
            self._outer_iteration,
            plan.display_name,
            len(plan.actions),
            plan.acceptance_mode,
        )
        logger.info(
            "Baseline before outer step %d: cost=%s, params=%d, two_qubit_gates=%d.",
            self._outer_iteration,
            "None" if cost_before is None else f"{cost_before:.10f}",
            num_parameters_before,
            num_two_qubit_before,
        )

        # Execute the structural change proposal.
        self.execute_action_plan(plan, cost=loss_function)

        train_result: OptimizerResult | None = None
        if train_after_plan:     
            logger.info(
                "Retraining after plan '%s' for %d inner-loop iterations.",
                plan.display_name,
                train_iterations,
            )
            
            if initial_point_generator is None and reuse_parameter_memory and self.parameter_memory:
                logger.info(
                    "Reusing parameter memory to build initial point for training after plan '%s'.",
                    plan.display_name,
                )
                initial_point = self.build_warm_start_initial_point(
                    default_value_for_new_params=default_value_for_new_params
                )
                logger.info(
                    "Built initial point from parameter memory for training: %s",
                    initial_point,
                )
            elif initial_point_generator is None:
                initial_point = [np.random.choice([-1.0, 1.0]) for _ in range(self.ansatz.num_parameters)]
            else:
                initial_point = initial_point_generator(self._outer_iteration)
            
            train_result = self.train_one_time(
                loss_function=loss_function,
                initial_point=initial_point,
                loss_next=loss_next,
                iterations=train_iterations,
                update_parameter_memory=update_parameter_memory,
                trainer_iteration_reset=trainer_iteration_reset,
                record_run_history=record_run_history,
                store_initial_value_in_history=store_initial_value_in_history,
                **train_kwargs,
            )
            cost_after = float(train_result.fun)
            
            self._reset_inner_loop_callback(
                callback_builder,
                **({} if callback_kwargs is None else callback_kwargs)
                ) # Will only reset if callback_builder is not None
        else:
            if plan.acceptance_mode == "outer":
                raise ValueError(
                    "`train_after_plan` must be True when `plan.acceptance_mode == 'outer'`."
                )
            cost_after = float(self.last_cost)

        ansatz_after = self.ansatz.copy()
        num_parameters_after = ansatz_after.num_parameters
        num_two_qubit_after = len(self._2qbg_positions)

        # Keep a copy of the trial parameter memory before a possible rollback.
        trial_parameter_memory = self.parameter_memory

        if plan.acceptance_mode == "outer":
            if self.outer_step_history:
                accepted = self._accept_outer_step(
                    cost_before=cost_before,
                    cost_after=cost_after,
                    accept_tol=accept_tol,
                    complexity_penalty=complexity_penalty,
                    ansatz_before=ansatz_before,
                    ansatz_after=ansatz_after,
                )
                note = None
            else:
                accepted = True  # Always accept the first step if no baseline is available.
                note = "First step with no baseline; automatically accepted."

            if accepted:
                logger.info(
                    "Accepted outer step %d for plan '%s'.",
                    self._outer_iteration,
                    plan.display_name,
                )
                
                if record_parameter_memory:
                    self._record_parameter_memory(
                        action=plan.display_name,
                        accepted=True,
                        cost=cost_after,
                    )
                    
                self._record_accepted_ansatz(
                    action=plan.display_name,
                    cost=cost_after,
                    note=note
                )
            else:
                
                logger.info(
                    "Rejected outer step %d for plan '%s'. Restoring previous snapshot.",
                    self._outer_iteration,
                    plan.display_name,
                )
                
                self._restore_state(snapshot)
                note = "Rejected and rolled back."
                if record_parameter_memory:
                    self.parameter_memory_history.append(
                        ParameterMemoryRecord(
                            outer_iteration=self._outer_iteration,
                            action=plan.display_name,
                            accepted=False,
                            values=trial_parameter_memory,
                            cost=cost_after,
                        )
                    )

        elif plan.acceptance_mode == "internal":
            
            logger.info(
                "Plan '%s' used internal acceptance mode; generic outer acceptance was skipped.",
                plan.display_name,
            )
            
            # Best-effort guess for internally accepted plans such as pruning:
            # if the circuit structure changed, we treat the proposal as accepted.
            accepted = (
                len(ansatz_after.data) != len(ansatz_before.data)
                or self._2qbg_positions != snapshot.two_q_map
            )
            note = "Internal acceptance mode; generic outer acceptance was skipped."

            if record_parameter_memory:
                self.parameter_memory_history.append(
                    ParameterMemoryRecord(
                        outer_iteration=self._outer_iteration,
                        action=plan.display_name,
                        accepted=accepted,
                        values=trial_parameter_memory,
                        cost=cost_after,
                    )
                )
                
            if accepted:
                self._record_accepted_ansatz(
                    action=plan.display_name,
                    cost=cost_after,
                    note=note
                )
        
        elif plan.acceptance_mode == "force":
            accepted = True
            note = "Force-accepted by plan configuration."

            logger.info(
                "Plan '%s' used force acceptance; generic outer acceptance was skipped.",
                plan.display_name,
            )

            if record_parameter_memory:
                self._record_parameter_memory(
                    action=plan.display_name,
                    accepted=True,
                    cost=cost_after,
                )
                
            self._record_accepted_ansatz(
                    action=plan.display_name,
                    cost=None,
                    note=note
                )
            
        else:
            raise ValueError(
                f"Unknown acceptance mode '{plan.acceptance_mode}'."
            )
            
        delta_cost = None if cost_before is None else cost_after - cost_before
            
        logger.info(
            "Completed outer step %d: accepted=%s, delta_cost=%s, params %d->%d, two_qubit_gates %d->%d.",
            self._outer_iteration,
            accepted,
            "None" if delta_cost is None else f"{delta_cost:.10f}",
            num_parameters_before,
            num_parameters_after,
            num_two_qubit_before,
            num_two_qubit_after,
        )

        result = OuterStepResult(
            iteration=self._outer_iteration,
            action=plan.display_name,
            accepted=accepted,
            cost_before=cost_before,
            cost_after=cost_after,
            delta_cost=delta_cost,
            num_parameters_before=num_parameters_before,
            num_parameters_after=num_parameters_after,
            num_two_qubit_before=num_two_qubit_before,
            num_two_qubit_after=num_two_qubit_after,
            note=note,
        )

        self.outer_step_history.append(result)
        self._outer_iteration += 1

        return result

    def run_outer_loop(
        self,
        loss_function: Callable[[np.ndarray], float],
        plan_schedule: list[Callable[["MutableAnsatzExperiment"], OuterStepPlan]],
        outer_iterations: int | None = None,
        train_iterations: int = 100,
        train_before_first_plan: bool = True,
        initial_point: list | np.ndarray | None = None,
        initial_point_generator: Callable[..., np.ndarray] | None = None,
        loss_next: Callable[[np.ndarray], float] | None = None,
        train_after_plan: bool = True,
        trainer_iteration_reset: int | None = 0,
        callback_builder: Callable[..., CALLBACK] | None = None,
        callback_kwargs: dict | None = None,
        update_parameter_memory: bool = True,
        reuse_parameter_memory: bool = False,
        default_value_for_new_params: float = 0.0,
        record_parameter_memory: bool = True,
        record_run_history: bool = False,
        store_initial_value_in_history: bool = False,
        accept_tol: float = 0.0,
        complexity_penalty: Callable[[QuantumCircuit], float] | None = None,
        stop_on_error: bool = True,
        **train_kwargs,
    ) -> list[OuterStepResult]:
        """
        Execute a sequence of outer-loop steps generated from a plan schedule.

        Parameters
        ----------
        loss_function : Callable[[np.ndarray], float]
            Objective function used for training and acceptance decisions.
        plan_schedule : list[Callable[[MutableAnsatzExperiment], OuterStepPlan]]
            Ordered list of configured plan builders. At outer iteration `n`, the
            `n`th builder is called with the current experiment state to construct
            the next `OuterStepPlan`. If `outer_iterations` exceeds the schedule
            length, the last builder is reused for all remaining steps.
        outer_iterations : int | None, optional
            Number of outer-loop steps to execute. If None, the schedule length is
            used.
        train_iterations : int, optional
            Number of inner-loop optimization steps after each executed plan.
        train_before_first_plan : bool, optional
            Whether to perform an initial training phase before executing the first plan.
        initial_point : list | np.ndarray | None, optional
            Explicit starting point for first training run. If train_before_actions
            is False, the initial point might not have the correct dimensions and will
            get reconciled with the current ansatz structure after executing the first plan.
        initial_point_generator : Callable[..., np.ndarray] | None, optional
            Optional factory for generating the initial point for each training phase.
            If provided, this is called at each outer step with the current experiment
            state and the reconciled `initial_point` to produce the actual initial
            point used for training. The signature of the generator should be
            `generator(iteration: int) -> np.ndarray`.
        loss_next : Callable[[np.ndarray], float] | None, optional
            Optional objective for next-step evaluation during training.
        train_after_plan : bool, optional
            Whether to retrain after executing each plan.
        trainer_iteration_reset : int | None, optional
            Value to which the optimizer's iteration counter is reset after retraining. 
            If None, the counter is not reset. Default is 0.
        callback_builder : Callable[..., CALLBACK] | None, optional
            Factory used to rebuild the optimizer callback after each retraining phase.
            This is useful when live-plot callbacks should be reset between outer-loop
            steps so that each inner-loop run starts with a fresh plot/history.
        callback_kwargs : dict | None, optional
            Keyword arguments forwarded to `callback_builder` when rebuilding the callback.
        update_parameter_memory : bool, optional
            Whether to update the live parameter cache after retraining.
        reuse_parameter_memory : bool, optional
            Whether to use `parameter_memory` to initialize retraining when
            `initial_point` is not provided.
        default_value_for_new_params : float, optional
            Default value assigned to newly introduced parameters when warm-starting.
        record_parameter_memory : bool, optional
            Whether to append parameter-memory records for each attempted outer step.
        record_run_history : bool, optional
            Whether to ask the trainer to store a `TrainingRunRecord` for each retraining phase
            of the outer steps.
        store_initial_value_in_history : bool, optional
            If True and `record_run_history=True`, evaluate the loss at the initial point and
            store it in the resulting training-run record for each retraining phase of the outer steps.
        accept_tol : float, optional
            Required score improvement threshold for generic outer acceptance.
        complexity_penalty : Callable[[QuantumCircuit], float] | None, optional
            Optional penalty added to the objective in generic outer acceptance.
        stop_on_error : bool, optional
            If True, raise immediately when a plan builder or outer step fails.
            If False, log the exception and stop the loop.
        **train_kwargs
            Additional keyword arguments forwarded to `run_outer_step`.

        Returns
        -------
        list[OuterStepResult]
            Results of the executed outer-loop steps.

        Raises
        ------
        ValueError
            If `plan_schedule` is empty or if `outer_iterations` is not positive.
        """
        if not plan_schedule:
            raise ValueError("`plan_schedule` must contain at least one plan builder.")

        if outer_iterations is None:
            outer_iterations = len(plan_schedule)

        if outer_iterations <= 0:
            raise ValueError("`outer_iterations` must be positive.")

        results: list[OuterStepResult] = []

        logger.info(
            "Starting outer loop for %d steps with %d scheduled plan builders.",
            outer_iterations,
            len(plan_schedule),
        )
        
        if train_before_first_plan and self._outer_iteration == 0:     
            logger.info(
                "Training before first plan execution for %d inner-loop iterations.",
                train_iterations,
            )
            
            if initial_point is None:
                initial_point = [np.random.choice([-1.0, 1.0]) for _ in range(self.ansatz.num_parameters)]
                logger.info(
                    "No initial point provided for training before first plan; using random initialization: %s",
                    initial_point,
                )
            
            if len(initial_point) != self.ansatz.num_parameters:
                raise ValueError(
                    f"Provided `initial_point` has {len(initial_point)} parameters, but the current "
                    f"ansatz has {self.ansatz.num_parameters} parameters. Please provide an `initial_point` "
                    f"with the correct dimensions or set `train_before_first_plan=False` to skip "
                    "this initial training phase."
                )
            
            train_result = self.train_one_time(
                loss_function=loss_function,
                initial_point=initial_point,
                loss_next=loss_next,
                iterations=train_iterations,
                update_parameter_memory=update_parameter_memory,
                trainer_iteration_reset=trainer_iteration_reset,
                record_run_history=record_run_history,
                store_initial_value_in_history=store_initial_value_in_history,
                **train_kwargs,
            )
                            
            self._reset_inner_loop_callback(
                callback_builder,
                **({} if callback_kwargs is None else callback_kwargs)
                ) # Will only reset if callback_builder is not None
            
            if record_parameter_memory:
                self._record_parameter_memory(
                    action="Initial training before first plan",
                    accepted=True,
                    cost=float(train_result.fun),
                )
                
                self._record_accepted_ansatz(
                    action="Initial training before first plan",
                    cost=float(train_result.fun),
                    note="Recorded accepted ansatz after initial training phase before executing any plans."
                )
            
            result = OuterStepResult(
                iteration=self._outer_iteration,
                action="Initial training before first plan",
                accepted=True,
                cost_before=None,
                cost_after=float(train_result.fun),
                delta_cost=None,
                num_parameters_before=self.ansatz.num_parameters,
                num_parameters_after=self.ansatz.num_parameters,
                num_two_qubit_before=len(self._2qbg_positions),
                num_two_qubit_after=len(self._2qbg_positions),
                note="Initial training phase before executing any plans.",
            )
            
            self.outer_step_history.append(result)
            # Advance an outer step in order to keep memory consistent.
            self._outer_iteration += 1 

        for step in range(outer_iterations):
            builder_index = min(step, len(plan_schedule) - 1)
            builder = plan_schedule[builder_index]

            logger.info(
                "Selecting plan builder %d for outer step %d.",
                builder_index,
                self._outer_iteration,
            )

            try:
                plan = builder(self)
            except Exception:
                logger.exception(
                    "Plan builder %d failed while constructing the plan for outer step %d.",
                    builder_index,
                    self._outer_iteration,
                )
                if stop_on_error:
                    raise
                break

            logger.info(
                "Built plan '%s' for outer step %d.",
                plan.display_name,
                self._outer_iteration,
            )

            try:
                result = self.run_outer_step(
                    loss_function=loss_function,
                    plan=plan,
                    train_iterations=train_iterations,
                    initial_point_generator=initial_point_generator,
                    loss_next=loss_next,
                    train_after_plan=train_after_plan,
                    trainer_iteration_reset=trainer_iteration_reset,
                    callback_builder=callback_builder,
                    callback_kwargs=callback_kwargs,
                    update_parameter_memory=update_parameter_memory,
                    reuse_parameter_memory=reuse_parameter_memory,
                    default_value_for_new_params=default_value_for_new_params,
                    record_parameter_memory=record_parameter_memory,
                    record_run_history=record_run_history,
                    store_initial_value_in_history=store_initial_value_in_history,
                    accept_tol=accept_tol,
                    complexity_penalty=complexity_penalty,
                    **train_kwargs,
                )
            except Exception:
                logger.exception(
                    "Outer step %d failed while executing plan '%s'.",
                    self._outer_iteration,
                    plan.display_name,
                )
                if stop_on_error:
                    raise
                break

            results.append(result)

        logger.info(
            "Finished outer loop after %d executed steps.",
            len(results),
        )

        return results

    def evaluate_current_objective(
        self,
        cost: Callable[[np.ndarray, QuantumCircuit], float],
        params: list[float] | np.ndarray | None = None,
    ) -> float:
        """
        Evaluate the objective for the current ansatz.

        Parameters
        ----------
        cost : Callable[[np.ndarray, QuantumCircuit], float]
            Objective function used to score the current ansatz. The function is
            expected to accept the parameter vector together with the ansatz either
            as ``cost(params, ansatz)`` or as ``cost(params, ansatz=ansatz)``.
        params : list[float] | np.ndarray | None, optional
            Parameter vector at which to evaluate the current ansatz. If None, the
            method uses `self.last_params` when it matches the current ansatz size.
            For a parameter-free ansatz, an empty vector is used automatically.

        Returns
        -------
        float
            Objective value of the current ansatz at the chosen parameter vector.

        Raises
        ------
        ValueError
            If no compatible parameter vector is available for the current ansatz.
        """
        current_ansatz = self.ansatz
        num_params = current_ansatz.num_parameters

        if params is None:
            if num_params == 0:
                params_array = np.array([], dtype=float)
            elif len(self.last_params) == num_params:
                params_array = np.asarray(self.last_params, dtype=float)
            else:
                raise ValueError(
                    "No compatible parameter vector is available for the current ansatz. "
                    "Provide `params` explicitly or train the ansatz first."
                )
        else:
            params_array = np.asarray(params, dtype=float)
            if len(params_array) != num_params:
                raise ValueError(
                    f"Got {len(params_array)} parameters for an ansatz with {num_params} parameters."
                )

        try:
            return float(cost(params_array, ansatz=current_ansatz))
        except TypeError:
            return float(cost(params_array, current_ansatz))
        
    def _action_registry(self) -> dict[str, Callable[..., None]]:
        """Return the mapping from action names to bound action handlers."""
        return {
            INSERT_RANDOM_GATE: self._action_insert_random_gate,
            INSERT_GATE: self._action_insert_gate,
            INSERT_BLOCK: self._action_insert_block,
            REMOVE_GATE: self._action_remove_gate,
            SIMPLIFY: self._action_simplify,
            PRUNE_TWO_QUBIT: self._action_prune_two_qubit,
        }
        
    def _action_insert_random_gate(self, **kwargs) -> None:
        """Insert a randomly chosen gate into the current ansatz."""
        self.insert_random()

    def _action_insert_gate(self, **kwargs) -> None:
        """Insert a specified gate at a given circuit index."""
        self.insert_at(
            gate=kwargs["gate"],
            qubits=kwargs["qubits"],
            circ_ind=kwargs["circ_ind"],
        )

    def _action_insert_block(self, **kwargs) -> None:
        """Insert a specified block at a given circuit index."""
        self.insert_block_at(
            block_name=kwargs["block_name"],
            qubits=kwargs["qubits"],
            circ_ind=kwargs["circ_ind"],
        )

    def _action_remove_gate(self, **kwargs) -> None:
        """Remove the gate at the specified circuit index."""
        self.remove_at(circ_ind=kwargs["circ_ind"])

    def _action_simplify(self, **kwargs) -> None:
        """Simplify the current ansatz with transpiler passes."""
        self.simplify_transpiler_passes(
            pass_manager=kwargs.get("pass_manager"),
            repetitions=kwargs.get("repetitions", 2),
            reset_locks_on_ambiguity=kwargs.get("reset_locks_on_ambiguity", True),
        )

    def _action_prune_two_qubit(self, **kwargs) -> None:
        """Attempt to prune one non-locked two-qubit gate."""
        gate_to_remove = kwargs.get("gate_to_remove")
        
        if gate_to_remove is None and "target_occurrence" in kwargs and "target_pair" in kwargs:
            target_occurrence = kwargs["target_occurrence"]
            target_pair = kwargs["target_pair"]
            
            gate_to_remove = self._get_circuit_index_from_pair_occurrence(
                occurrence=target_occurrence,
                pair=target_pair,
            )
            
        if gate_to_remove is None:
            logger.info(
                "Skipping targeted prune because gate with pair %s and occurrence %s "
                "no longer exists in the current ansatz.",
                target_pair,
                target_occurrence,
            )
            return
        
        self.prune_two_qubit_gate_attempt(
            cost=kwargs["cost"],
            gate_to_remove=gate_to_remove,
            temperature=kwargs.get("temperature", 0.08),
            alpha=kwargs.get("alpha", 0.1),
            accept_tol=kwargs.get("accept_tol", 0.2),
        )
        
    def _apply_action(
        self,
        spec: ActionSpec,
        cost: Callable[[np.ndarray, QuantumCircuit], float] | None = None,
    ) -> None:
        """
        Apply one validated atomic action.

        Parameters
        ----------
        spec : ActionSpec
            ActionSpec to apply. Supported actions are:
            - ``"insert_random_gate"``
            - ``"insert_gate"``
            - ``"insert_block"``
            - ``"remove_gate"``
            - ``"simplify"``
            - ``"prune_two_qubit"``
        cost : Callable[[np.ndarray, QuantumCircuit], float] | None, optional
            Objective function required by actions that internally evaluate the
            ansatz, currently ``"prune_two_qubit"``.

        Raises
        ------
        ValueError
            If the action name is unknown or if a required argument is missing.

        Notes
        -----
        This helper keeps the outer-loop driver logic compact by translating a
        small action vocabulary into calls to the corresponding structural-update
        methods of the experiment.
        """
        registry = self._action_registry()
        
        if spec.requires_cost:
            if cost is None:
                raise ValueError(
                    f"Action '{spec.action}' requires a cost callable."
                )
            registry[spec.action](cost=cost, **spec.kwargs)
        else:
            registry[spec.action](**spec.kwargs)
        
    def execute_action_plan(
        self,
        plan: OuterStepPlan,
        cost: Callable[[np.ndarray, QuantumCircuit], float] | None = None,
    ) -> None:
        """
        Execute all atomic actions contained in an outer-step plan.
        
        Parameters
        ----------
        plan : OuterStepPlan
            Structured plan containing a sequence of atomic actions to apply to the
            current ansatz.
        cost : Callable[[np.ndarray, QuantumCircuit], float] | None, optional
            Objective function required by actions that internally evaluate the
            ansatz, currently ``"prune_two_qubit"``.
        
        Raises
        ------
        ValueError
            If any action in the plan is unknown or if a required argument is missing.
        """
        for spec in plan.actions:
            self._apply_action(spec, cost=cost)
    
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
            parameter_memory=self.parameter_memory,
            parameter_memory_history=[
                ParameterMemoryRecord(
                    outer_iteration=record.outer_iteration,
                    action=record.action,
                    accepted=record.accepted,
                    values=record.values,
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

        self.parameter_memory = snapshot.parameter_memory
        self.parameter_memory_history = [
            ParameterMemoryRecord(
                outer_iteration=record.outer_iteration,
                action=record.action,
                accepted=record.accepted,
                values=record.values,
                cost=record.cost,
            )
            for record in snapshot.parameter_memory_history
        ]

        self._outer_iteration = snapshot.outer_iteration

        self.trainer.update_last_evaluation(
            cost=snapshot.last_cost,
            params=np.asarray(snapshot.last_params, dtype=float).copy(),
        )

    def _accept_outer_step(
        self,
        cost_before: float,
        cost_after: float,
        accept_tol: float = 0.0,
        complexity_penalty: Callable[[QuantumCircuit], float] | None = None,
        ansatz_before: QuantumCircuit | None = None,
        ansatz_after: QuantumCircuit | None = None,
    ) -> bool:
        """
        Return whether a proposed outer-loop update should be accepted.

        Parameters
        ----------
        cost_before : float
            Objective value before the structural proposal.
        cost_after : float
            Objective value after the structural proposal and retraining.
        accept_tol : float, optional
            Required improvement threshold. The proposal is accepted if the final
            score is at least `accept_tol` lower than the initial score.
        complexity_penalty : Callable[[QuantumCircuit], float] | None, optional
            Optional penalty function added to the objective in order to discourage
            overly complex ansaetze.
        ansatz_before : QuantumCircuit | None, optional
            Ansatz before the proposal. Required if `complexity_penalty` is used.
        ansatz_after : QuantumCircuit | None, optional
            Ansatz after the proposal. Required if `complexity_penalty` is used.

        Returns
        -------
        bool
            True if the proposal is accepted, False otherwise.
        """
        score_before = float(cost_before)
        score_after = float(cost_after)

        if complexity_penalty is not None:
            if ansatz_before is None or ansatz_after is None:
                raise ValueError(
                    "`ansatz_before` and `ansatz_after` must be provided when "
                    "`complexity_penalty` is used."
                )
            score_before += float(complexity_penalty(ansatz_before))
            score_after += float(complexity_penalty(ansatz_after))
            
        logger.info(
            "Evaluating outer acceptance: score_before=%.10f, score_after=%.10f, accept_tol=%.10f.",
            score_before,
            score_after,
            accept_tol,
        )

        return score_after <= score_before - accept_tol

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
            
        if len(qubits) == 2:
            self._update_locked_gates_on_insert(circ_ind)
            logger.info(f"Updated ansatz. New 2 qubit gate positions are: {self._2qbg_positions}.")
            
        self._sync_after_ansatz_change()

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
        
        if is_2qbg:
            self._update_locked_gates_on_removal(circ_ind)
            logger.info("Updated 2Q positions after removal: %s", self._2qbg_positions)
            
        self._sync_after_ansatz_change()
        
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
        logger.info(f"New number of parameters: {self.ansatz.num_parameters}.")

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
        logger.info(
            "Starting ansatz simplification with transpiler passes for %d repetitions.",
             repetitions
        )
        result = simplify_ansatz(
            circuit=self.adaptive_ansatz.current_ansatz,
            pass_manager=pass_manager,
            repetitions=repetitions,
        )

        self.adaptive_ansatz.current_ansatz = result.circuit
        self._sync_after_ansatz_change()

        if not result.preserve_locked_gates and reset_locks_on_ambiguity:
            logger.info(
                "Resetting locked 2Q gates after simplification because the 2Q map changed."
            )
            self.locked_gates = set()

        return self.adaptive_ansatz.current_ansatz

    def prune_two_qubit_gate_attempt(
        self, 
        cost: Callable, 
        gate_to_remove: int | None = None,
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
        gate_to_remove : int | None, optional
            Specific circuit index of the two-qubit gate to attempt to prune. If None, 
            a gate is chosen automatically from the non-locked two-qubit gates in the ansatz.
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
            gate_to_remove=gate_to_remove
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
        """Return the current optimizer from the trainer."""
        return self.trainer.optimizer

    @property
    def gradient_history(self):
        """Return the history of gradient evaluations from the trainer."""
        return self.trainer.gradient_history

    @property
    def last_cost(self):
        """Return the last accepted cost value from the trainer."""
        return self.trainer.last_cost

    @property
    def last_params(self):
        """Return the last accepted parameter vector from the trainer."""
        return self.trainer.last_params
    
    @property
    def training_run_history(self):
        """Return the stored inner-loop training-run history."""
        return self.trainer.training_run_history

    @property
    def last_training_run_record(self):
        """Return the most recent stored inner-loop training-run record."""
        return self.trainer.last_training_run_record
