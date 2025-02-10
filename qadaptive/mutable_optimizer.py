import numpy as np
import logging
from time import time
from typing import Callable, SupportsFloat

from qiskit.circuit import Parameter, QuantumCircuit
from qiskit_algorithms.optimizers.optimizer import Optimizer, OptimizerResult

from qae.optimization.my_spsa import SPSA
from qadaptive.adaptive_ansatz import AdaptiveAnsatz

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
    _current_iteration : int
        Internal counter for the current iteration (used internally by the optimizer).
    """

    def __init__(
        self, 
        adaptive_ansatz: AdaptiveAnsatz, 
        optimizer: Optimizer | None = None, 
        track_gradients: bool = True,
        callback: CALLBACK | None = None,
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
        callback : CALLBACK
            A callback function passed information in each iteration step. The
            function signature has to be: (the number of function evaluations, the parameters,
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
        self._current_iteration = 0
        self.optimizer = SPSA(**optimizer_options) if optimizer is None else optimizer
        self.track_gradients = track_gradients
        self.gradient_history = {0 : []} if track_gradients else None
        self.iteration = 0
        self.callback = callback
        self.termination_checker = termination_checker
        self._times_trained = 0

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
            gradient_estimate, x, fx_estimate, loss_next, iteration_start, self._current_iteration
            )
        
        current_learn_rate = next(self.optimizer._lr_iterator_copy)
        if not skip:
            self.optimizer.last_iteration += 1
        
        if self.callback:
            self.callback(
                self.optimizer._nextfev, 
                x_next, 
                fx_next, 
                np.linalg.norm(gradient_estimate*current_learn_rate),
                skip
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
        Train the ansatz for a given number of iterations.

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
            self.iteration += 1
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

    def prune_and_grow(self):
        """
        Apply pruning and growing logic based on gradient analysis.
        """
        # Placeholder for pruning & growth decision logic
        pass
