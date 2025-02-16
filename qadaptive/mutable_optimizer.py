import numpy as np
import logging, random
from time import time
from typing import Callable, SupportsFloat

from qiskit.circuit import QuantumCircuit
from qiskit_algorithms.optimizers.optimizer import Optimizer, OptimizerResult

from qae.optimization.my_spsa import SPSA
from qadaptive.adaptive_ansatz import AdaptiveAnsatz
from qadaptive.utils import custom_pass_manager, change_circuit_parameters

CALLBACK = Callable[[int, np.ndarray, float, SupportsFloat, bool], None]
TERMINATIONCHECKER = Callable[[int, np.ndarray, float, SupportsFloat, bool], bool]

logger = logging.getLogger(__name__)

class MutableOptimizer:
    """
    An optimizer that dynamically modifies a quantum ansatz while training.

    This class tracks gradients, updates parameters, and applies pruning or growth
    strategies based on optimization performance.
    
    Attributes
    ----------
    ansatz : AdaptiveAnsatz
        The adaptive ansatz to be optimized.
    optimizer : Optimizer
        The classical optimizer object (e.g., SPSA, Adam). Defaults to qae's SPSA if None.
    track_gradients : bool
        Indicates whether gradient history is being tracked.
    gradient_history : list or None
        Stores the gradients at each iteration if `track_gradients` is True; otherwise `None`.
    iteration : int
        The current iteration number in the optimization process.
    callback : CALLBACK or None
        A function called at each iteration step with optimization info.
    termination_checker : TERMINATIONCHECKER or None
        A function executed at the end of each iteration to determine if optimization should terminate.
    _inner_iteration : int
        Internal counter for the current iteration (used internally by the optimizer).
    """

    def __init__(
        self, 
        adaptive_ansatz: AdaptiveAnsatz, 
        optimizer: Optimizer | None = None, 
        track_gradients: bool = True,
        callback: CALLBACK | list[CALLBACK] | None = None,
        termination_checker: TERMINATIONCHECKER | None = None,
        **optimizer_options
        ) -> None:
        """
        Initialize the MutableOptimizer.

        Parameters
        ----------
        ansatz : AdaptiveAnsatz
            The adaptive ansatz to be optimized.
        optimizer : Optimizer, optional
            The classical optimizer object (e.g., SPSA, Adam). Defaults
            to None and qae's SPSA implementation is used.
        track_gradients : bool, optional
            Whether to keep track of gradient history (default: True).
        callback : CALLBACK, list[CALLBACK], optional
            A callback function or list of functions passed information in each iteration step. 
            The function signature has to be: (the number of function evaluations, the parameters,
            the function value, the stepsize, whether the step was accepted).
        termination_checker : TERMINATIONCHECKER
            A callback function executed at the end of each iteration step. The
            function signature has to be: (the parameters, the function value, the number
            of function evaluations, the stepsize, whether the step was accepted). If the callback
            returns True, the optimization is terminated.
        optimizer_options
            Options to be passed to the default optimizer.
            
        Notes
        ----------
        The optimizers provided must have the following methods available:
        ``compute_loss_and_gradient_estimate``, ``process_update``
        If they don't have them, they must be implemented.
        """
        self.adaptive_ansatz = adaptive_ansatz
        self.ansatz: QuantumCircuit = adaptive_ansatz.get_current_ansatz()
        self.optimizer = SPSA(**optimizer_options) if optimizer is None else optimizer
        self.track_gradients = track_gradients
        self.gradient_history = {0 : []} if track_gradients else None
        # Make callback a list
        if callback is not None and not isinstance(callback, list):
            self.callback = [callback]
        else:
            self.callback = callback
        self.termination_checker = termination_checker
        self._times_trained = 0
        # Inner loop and outer loop iteration counters
        self._inner_iteration = 0
        self._outer_iteration = 0
        # Last cost function and last parameters evaluated
        self._last_cost = 0
        self._last_params = []
        # Two qubit gate positions
        self._2qg_positions = self._get_two_qubit_gate_indices()
        # Some 2 qubit gates will be important and thus get locked
        # Since the ansatz will be constantly changing, this is tracked by looking at the
        # n'th two qubit gate. No gate is locked by default.
        self.locked_gates = {i : False for i in range(len(self._2qg_positions))}

    def step(
        self, 
        loss_function: Callable[[np.ndarray], float], 
        x: np.ndarray, 
        loss_next: Callable[[np.ndarray], float] | None = None,
        **kwargs
        ) -> tuple[bool, np.ndarray, float, float]:
        """
        Perform a single optimization step.

        Computes gradients, updates parameters, and modifies the ansatz if needed.
        
        Parameters
        ----------
        loss_function : Callable[[np.ndarray], float]
            The cost function to evaluate the ansatz.
        x : np.ndarray
            The point at which the step is taken.
        loss_next : Callable[[np.ndarray], float], optional
            An optional function to evaluate the objective at the next step.
            
        Notes
        ----------
        In this method, the optimizer attribute is modified in the following ways:
        - Iterators are created if they were not already initialized.
        - Number of cost function evaluations is increased.
        - The number of iterations is increased.
        """
        
        iteration_start = time()
        x = np.asarray(x)
                
        fx_estimate, gradient_estimate = self.optimizer.compute_loss_and_gradient_estimate(
            loss_function, x, **kwargs
            )

        skip, x_next, fx_next = self.optimizer.process_update(
            gradient_estimate, x, fx_estimate, loss_next, iteration_start, self._inner_iteration
            )
        
        current_learn_rate = next(self.optimizer._lr_iterator_copy)
        if not skip:
            self.optimizer.last_iteration += 1
        
        if self.callback:
            for callback_function in self.callback:
            # TODO: Decide callback signature
                callback_function(
                    )
            
        return skip, x_next, fx_next, gradient_estimate, fx_estimate

    def get_gradients(self) -> np.ndarray:
        """
        Get gradients of the cost function with respect to ansatz parameters.

        Returns
        -------
        np.ndarray
            The current estimated gradient values.
        """
        # Placeholder for actual gradient computation
        return np.random.randn(len(self.adaptive_ansatz.param_vector))

    def update_parameters(self, gradients: np.ndarray):
        """
        Update ansatz parameters using the optimizer.

        Parameters
        ----------
        gradients : np.ndarray
            The computed gradients.
        """
        # Placeholder: Apply optimizer update
        pass
    
    def _update_ansatz(self) -> None:
        """Update the ansatz attribute."""
        self.ansatz = self.adaptive_ansatz.get_current_ansatz()
        self._2qg_positions = self._get_two_qubit_gate_indices()
        
    def _get_two_qubit_gate_indices(self) -> list[int]:
        """
        Find the locations of the 2 qubit gates.

        Returns
        -------
        list[int]
            The indices in self.ansatz.data where the 2 qubit gates are.
        """
        indices = []
        for i, gate in enumerate(self.ansatz.data):
            if len(gate.qubits) == 2:
                indices.append(i)
        
        return indices
    
    def lock_gates(self, gate_position: dict[int, bool]) -> None:
        """
        Set the locked state of 2-qubit gates based on the specified positions.

        This method updates the locked state of 2-qubit gates according to the given
        dictionary. Each key in the dictionary represents the position of a gate,
        and the corresponding value indicates whether the gate should be locked (True) or 
        not locked (False).

        Parameters
        ----------
        gate_position : dict[int, bool]
            A dictionary where the keys are the positions of the 2-qubit gates, and the values
            indicate whether each gate should be locked (True) or not locked (False).

        Raises
        ------
        KeyError
            If any key in `gate_position` is not found in `self.locked_gates`.

        Example
        -------
        >>> gate_position = {1: True, 2: False, 3: True}
        >>> obj.lock_gates(gate_position)
        """
        for pos in gate_position.keys():
            if pos not in self.locked_gates:
                raise KeyError(f"Gate position {pos} not found in locked_gates.")
            self.locked_gates[pos] = gate_position[pos]

    def train_one_time(
        self, 
        loss_function: Callable[[np.ndarray], float],
        initial_point: list | np.ndarray | None = None,
        loss_next: Callable[[np.ndarray], float] | None = None,
        iterations: int = 100,
        adaptation_callback: Callable | None = None,
        **kwargs
        ) -> OptimizerResult:
        """
        Train the ansatz for a given number of iterations. This is part of the inner loop.

        Parameters
        ----------
        loss_function : Callable[[np.ndarray], float]
            The cost function to evaluate the ansatz.
        iterations : int
            The number of optimization steps.
            
        Returns
        ----------
        OptimizerResult
            The result of the optimization.
        """
        logger.info(f"Started minimization of loss funcion. Repetitions: {self._times_trained}.")
        
        if self.optimizer.blocking:
            raise NotImplementedError("Training with blocking is not yet implemented.")
        
        if initial_point is None:
            initial_point = np.random.normal(loc=0, scale=1, size=len(self.ansatz.parameters))
        x = np.asarray(initial_point)
        
        # If the iterators have not been set, set them now.
        if self.optimizer.p_iterator is None and self.optimizer.lr_iterator is None:
            self.optimizer._create_iterators(loss_function, initial_point)
        
        # Set up optimizer for epoch mode or iteration mode
        use_epochs = kwargs.get('use_epochs')
        if use_epochs:
            logger.info(f"Starting optimization with initial parameters {x} doing epochs.")
            logger.info("Interpreting max number of iterations as max number of epochs.")
        else:
            logger.info(f"Starting optimization with initial parameters {x}.")
        
        ncpg = kwargs.get('num_circs_per_group')
        ncpb = kwargs.get('num_circs_per_batch')
        if use_epochs:
            # Total number of circuits (complete data set)
            total_circuits = self.optimizer._size_full_batch if self.optimizer._size_full_batch else 12
            if not ncpb:
                ncpb = 3
        logger.info(f"Setting number of circuits per batch to {ncpb}.")
        # use a local variable and while loop to keep track of the number of iterations
        # if the termination checker terminates early
        start = time()
        k = 0
        logger.info("Starting optimization loop")
        while k < iterations:
            k += 1
            current_learn_rate = next(self.optimizer._lr_iterator_copy)
            self._inner_iterationiteration += 1
            iteration_start = time()
            # Compute updates for the whole batched dataset when using epochs
            if use_epochs:
                # Generate indices for all circuits
                indices = np.arange(total_circuits)
                # Shuffle indices for randomness
                np.random.shuffle(indices)
                # Split indices into batches
                batch_indices = [indices[i:i + ncpb] for i in range(0, total_circuits, ncpb)]
                for i, indices in enumerate(batch_indices):
                    logger.info(f"Evaluating batch {i+1} out of {len(batch_indices)}.")
                    skip, x_next, fx_next, gradient_estimate, fx_estimate = self.step(
                        loss_function, x, loss_next, used_circs_indices = indices
                        )
                    if skip:
                        continue
                    # Update values
                    x = x_next
                logger.info(f"Epoch {k}/{iterations} finished in {time() - iteration_start}")
            # Compute updates iteration by iteration
            else:
                skip, x_next, fx_next, gradient_estimate, fx_estimate = self.step(
                    loss_function, x, loss_next, num_circs_per_group = ncpg
                    )
                if skip:
                    continue
                # Update values
                x = x_next
                logger.info(f"Iteration {k}/{iterations} done in {time()-iteration_start}.")
            
            logger.info("Running the optimizer's callback.")
            # Run the optimizer's callback
            if self.optimizer.callback is not None:
                if loss_next is  None:
                    logger.info("Calculating next step for the callback, which takes another function evaluation.")
                    self.optimizer._nfev += 1
                    fx_next = loss_function(x_next)
                else:
                    logger.info("Calculating next step for the callback with custom function.")
                    self.optimizer._nextfev += 1
                    fx_next = loss_next(x_next)
                    
                self.optimizer.callback(
                    self.optimizer._nfev,  # number of function evals
                    x_next,  # next parameters
                    fx_next,  # loss at next parameters
                    np.linalg.norm(gradient_estimate*current_learn_rate),  # size of the update step
                    True, # accepted
                )
            
            # Update gradient history
            if self.track_gradients:
                self.gradient_history[self._times_trained].append(gradient_estimate)

            if self.optimizer.termination_checker is not None:
                fx_check = fx_estimate if fx_next is None else fx_next
                if self.optimizer.termination_checker(
                    self.optimizer._nfev, x_next, fx_check, np.linalg.norm(gradient_estimate*current_learn_rate), True
                ):
                    logger.info(f"terminated optimization at {k}/{iterations} iterations")
                    break

        logger.info("SPSA: Finished in %s", time() - start)
        logger.ingo("Setting inner loop optimization iteration count back to 0.")
        self._inner_iteration = 0
        
        self._times_trained += 1 
        if self.track_gradients:
            self.gradient_history[self._times_trained] = []
            
        result = OptimizerResult()
        result.x = x
        if loss_next is None:
            logger.info("Calculating cost funtion value for final parameters.")
        else:
            logger.info("Calculating custom cost funtion value for final parameters.")
        result.fun = loss_function(x) if loss_next is None else loss_next(x)
        result.nfev = self.optimizer._nfev
        result.nit = k

        return result

    def insert_random(self) -> None:
        """
        Insert a new gate into the adaptive ansatz at the current step.

        Parameters
        ----------
        gate : str
            The type of gate to insert (e.g., 'rx', 'ry', 'cz').
        qubits : list[int]
            The qubits on which to apply the gate.
        params : list[float] or None
            Parameters for the gate (if applicable). Default is None.
        """
        gate_name, qubits, index = self.adaptive_ansatz.add_random_gate()
        logger.info(f"Inserted {gate_name} gate on qubits {qubits} at position {index}.")
        self._update_ansatz()
        
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
        self._update_ansatz()
        
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
        self.adaptive_ansatz.remove_gate_by_index(circ_ind)
        self._update_ansatz()     

    def simplify_unimportant_2qb_gates(
        self, cost: Callable, temperature: float = 0.1, alpha: float = 5.0
        ) -> None:
        """
        Remove 2-qubit gates that do not significantly affect the cost function.

        Parameters
        ----------
        temperature : float, optional
            The temperature factor for Metropolis-like acceptance probability:
            
            .. math::
                p = exp(-\beta \frac{C_{new} - C_{o}}{C_o})
        alpha : float, optional
            Scaling factor for gate locking probability:

            .. math::
                P_{\text{lock}} = 1 - e^{-\alpha \frac{\Delta C}{|C_o|}}
                
        Notes
        ----------
        - If removing the gate **lowers or keeps the cost the same**, it is removed.
        - If the cost increases, the removal is accepted with probability `p`.
        - If the removal is **rejected**, the gate may be locked to prevent further attempts.
        """
        
        if not self._2qg_positions:
            return  # No 2-qubit gates to remove
        
        # Pick a random 2-qubit gate that is NOT locked
        removable_indices = [i - 1 for i, _ in enumerate(self._2qg_positions, start=1) 
                             if not self.locked_gates.get(i, False)]
        if not removable_indices:
            return  # No gates left to consider
        
        gate_index = random.choice(removable_indices)
        gate_to_remove = self._2qg_positions[gate_index]  # Convert index to position in ansatz.data
        
        # Create a trial ansatz with the gate removed
        trial_ansatz = self.ansatz.copy()
        trial_ansatz.data.pop(gate_to_remove)
        
        # Compute the new cost
        trial_cost = cost(self._last_params, trial_ansatz)
        current_cost = self._last_cost
        
        # Compute cost difference
        delta_C = trial_cost - current_cost
        
        # If the cost decreases or stays the same, accept removal
        if delta_C <= 0:
            self.adaptive_ansatz.update_ansatz(trial_ansatz)  # Update ansatz
            self._update_ansatz()
            self._last_cost = trial_cost
            # Update locked_gates: shift remaining indices down
            self.locked_gates = {i: self.locked_gates[i + 1] for i in range(gate_index, len(self.locked_gates))}
            self.locked_gates.pop(len(self.locked_gates), None)  # Remove last entry
            return
        
        # Compute acceptance probability
        beta = 1 / temperature if temperature > 0 else float("inf")  # Avoid division by zero
        acceptance_prob = np.exp(-beta * delta_C / abs(current_cost))
        
        # Accept the removal with probability p
        if np.random.rand() < acceptance_prob:
            self.adaptive_ansatz.update_ansatz(trial_ansatz)  # Update ansatz
            self._update_ansatz()
            self._last_cost = trial_cost
            # Update locked_gates: shift remaining indices down
            self.locked_gates = {i: self.locked_gates[i + 1] for i in range(gate_index, len(self.locked_gates))}
            self.locked_gates.pop(len(self.locked_gates), None)  # Remove last entry
        else:
            # Reject removal: Consider locking the gate with some probability
            lock_prob = 1 - np.exp(-alpha * delta_C / abs(current_cost))
            if np.random.rand() < lock_prob:
                self.locked_gates[gate_index] = True
        
        logger.info(f"Simplified ansatz by removing 2 qubit gate at index {gate_to_remove}.")
        self._update_ansatz()
        
    def simplify_transpiler_passes(self) -> QuantumCircuit:
        """
        Remove conditional operations and Rz rotations at the start of the circuit and joing
        together succesive rotations around the same axis.

        Returns
        ----------
        QuantumCircuit
            The resulting circuit.
        """
        new_circuit, new_vector = change_circuit_parameters(
            custom_pass_manager.run(self.adaptive_ansatz.current_ansatz)
            )
        logger.info("Simplified ansatz by removing doing compilation passes.")
        
        self.adaptive_ansatz.update_ansatz(new_circuit)
        logger.info(f"Updated parameter vector from {self.adaptive_ansatz.param_vector} to {new_vector}.")
        self.adaptive_ansatz.update_parameter_vector(new_vector)
        
        self._update_ansatz()
