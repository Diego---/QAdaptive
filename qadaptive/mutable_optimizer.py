import numpy as np
from time import time
from typing import Callable, SupportsFloat

from qiskit.circuit import Parameter
from qiskit_algorithms.optimizers.optimizer import Optimizer

from qae.optimization.my_spsa import SPSA
from qadaptive.adaptive_ansatz import AdaptiveAnsatz

CALLBACK = Callable[[int, np.ndarray, float, SupportsFloat, bool], None]
TERMINATIONCHECKER = Callable[[int, np.ndarray, float, SupportsFloat, bool], bool]

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
        ansatz: AdaptiveAnsatz, 
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
        self.ansatz = ansatz
        self._current_iteration = 0
        self.optimizer = SPSA(**optimizer_options) if optimizer is None else optimizer
        self.track_gradients = track_gradients
        self.gradient_history = [] if track_gradients else None
        self.iteration = 0
        self.callback = callback
        self.termination_checker = termination_checker

    def step(
        self, 
        loss_function: Callable[[np.ndarray], float], 
        x: np.ndarray, 
        loss_next: Callable[[np.ndarray], float] | None = None
        ) -> tuple[bool, np.ndarray, float]:
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
            loss_function, x
            )
        
        if self.track_gradients:
            self.gradient_history.append(gradient_estimate)

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
            
        return skip, x_next, fx_next

    def compute_gradients(self) -> np.ndarray:
        """
        Compute gradients of the cost function with respect to ansatz parameters.

        Returns
        -------
        np.ndarray
            The computed gradient values.
        """
        # Placeholder for actual gradient computation
        return np.random.randn(len(self.ansatz.param_vector))

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

    def train(self, iterations: int = 100):
        """
        Train the ansatz for a given number of iterations.

        Parameters
        ----------
        iterations : int
            The number of optimization steps.
        """
        for _ in range(iterations):
            self.step()
            self.iteration += 1

    def prune_and_grow(self):
        """
        Apply pruning and growing logic based on gradient analysis.
        """
        # Placeholder for pruning & growth decision logic
        pass
