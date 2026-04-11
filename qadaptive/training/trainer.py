import logging
import random
from time import time
from typing import Callable, SupportsFloat

import numpy as np
from qiskit import QuantumCircuit
from qiskit_algorithms.optimizers.optimizer import OptimizerResult

from .optimizers import SPSA
from .stepwise_optimizer import StepwiseOptimizer, CALLBACK, TERMINATIONCHECKER

logger = logging.getLogger(__name__)

class InnerLoopTrainer:
    """
    Optimizer-agnostic inner-loop trainer for adaptive ansatz optimization.

    This class orchestrates repeated parameter updates for a fixed ansatz
    structure using any optimizer that implements the project's
    ``StepwiseOptimizer`` interface.

    Attributes
    ----------
    optimizer : StepwiseOptimizer
        Optimizer instance used for training.
    track_gradients : bool
        Whether to store gradient estimates during training.
    callback : list[CALLBACK] | None
        Optional trainer-level callback or callbacks executed after each
        accepted step.
    termination_checker : TERMINATIONCHECKER | None
        Optional trainer-level termination checker.
    gradient_history : dict[int, list[np.ndarray]] | None
        Gradient history indexed by training repetition.
    """

    def __init__(
        self,
        optimizer: StepwiseOptimizer | None = None,
        optimizer_options: dict | None = None,
        track_gradients: bool = True,
        callback: CALLBACK | list[CALLBACK] | None = None,
        termination_checker: TERMINATIONCHECKER | None = None,
    ) -> None:
        """
        Initialize the inner-loop trainer.

        Parameters
        ----------
        optimizer : StepwiseOptimizer | None, optional
            Pre-initialized optimizer instance.
        optimizer_options : dict | None, optional
            Keyword arguments used to initialize a default SPSA optimizer if
            ``optimizer`` is not provided.
        track_gradients : bool, optional
            Whether to store gradient estimates during training.
        callback : CALLBACK | list[CALLBACK] | None, optional
            One callback or a list of callbacks executed after each accepted step.
        termination_checker : TERMINATIONCHECKER | None, optional
            Optional trainer-level termination checker.
        """
        if optimizer is None and optimizer_options is None:
            raise ValueError(
                "Provide either a pre-initialized optimizer or optimizer_options."
            )
            
        if optimizer is not None and not isinstance(optimizer, StepwiseOptimizer):
            raise TypeError(f"Expected StepwiseOptimizer, got {type(optimizer).__name__}")

        self.optimizer = optimizer if optimizer is not None else SPSA(**optimizer_options)
        self.track_gradients = track_gradients

        if callback is None:
            self.callback = None
        elif isinstance(callback, list):
            self.callback = callback
        else:
            self.callback = [callback]

        self.termination_checker = termination_checker

        # Training state
        self.gradient_history = {0: []} if track_gradients else None
        self._times_trained = 0
        self._last_cost = 0.0
        self._last_params = np.array([], dtype=float)
        self._last_num_iterations = 0

    @property
    def last_cost(self) -> float:
        """Return the last recorded objective value."""
        return self._last_cost

    @property
    def last_params(self) -> np.ndarray:
        """Return the last recorded parameter vector."""
        return self._last_params

    def update_last_evaluation(
        self,
        cost: float,
        params: list[float] | np.ndarray | None = None,
    ) -> None:
        """
        Update the cached record of the most recent accepted evaluation.

        Parameters
        ----------
        cost : float
            Objective value associated with the accepted ansatz state.
        params : list[float] | np.ndarray | None, optional
            Parameter vector associated with the accepted ansatz state. If
            provided, it replaces the currently stored parameter vector.
        """
        self._last_cost = float(cost)

        if params is not None:
            self._last_params = np.asarray(params, dtype=float)

    def set_optimizer(
        self,
        optimizer: StepwiseOptimizer | None = None,
        optimizer_options: dict | None = None,
    ) -> None:
        """
        Set or replace the optimizer used by the trainer.

        Parameters
        ----------
        optimizer : StepwiseOptimizer | None, optional
            Pre-initialized optimizer instance.
        optimizer_options : dict | None, optional
            Keyword arguments used to initialize a default SPSA optimizer if
            ``optimizer`` is not provided.

        Raises
        ------
        ValueError
            If both ``optimizer`` and ``optimizer_options`` are None.
        """
        if optimizer is not None:
            self.optimizer = optimizer
        elif optimizer_options is not None:
            self.optimizer = SPSA(**optimizer_options)
        else:
            raise ValueError("Either 'optimizer' or 'optimizer_options' must be provided.")

    def step(
        self,
        ansatz: QuantumCircuit,
        loss_function: Callable[[np.ndarray], float],
        x: np.ndarray,
        loss_next: Callable[[np.ndarray], float] | None = None,
        **kwargs,
    ) -> tuple[bool, np.ndarray, float | None, np.ndarray | None, float | None]:
        """
        Perform one optimization step for the current ansatz.

        Parameters
        ----------
        ansatz : QuantumCircuit
            Current ansatz circuit.
        loss_function : Callable[[np.ndarray], float]
            Objective function to minimize. It must accept the parameter vector
            and the keyword argument ``ansatz``.
        x : np.ndarray
            Current parameter vector.
        loss_next : Callable[[np.ndarray], float] | None, optional
            Optional objective used to evaluate the proposed next point.
        **kwargs
            Additional keyword arguments forwarded to the objective.

        Returns
        -------
        tuple[bool, np.ndarray, float | None, np.ndarray | None, float | None]
            Tuple ``(skip, x_next, fx_next, gradient_estimate, fx_estimate)``.
        """
        if self.optimizer is None:
            raise RuntimeError(
                "The optimizer is not set. Set an optimizer before running a training step."
            )

        x = np.asarray(x, dtype=float)
        # The ansatz kwarg gets used by the loss function, whose signature can be
        # loss(params, ansatz)
        loss_kwargs = {**kwargs, "ansatz": ansatz}

        return self.optimizer.step(
            x,
            loss_function,
            loss_next=loss_next,
            **loss_kwargs,
        )

    def _run_trainer_callbacks(
        self,
        params: np.ndarray,
        fx_value: float,
        accepted: bool,
    ) -> None:
        """
        Execute trainer-level callbacks.

        Parameters
        ----------
        params : np.ndarray
            Current parameter vector.
        fx_value : float
            Objective value associated with ``params``.
        accepted : bool
            Whether the step was accepted.
        """
        if self.callback is None:
            return

        step_size = 0.0 if self.optimizer.last_stepsize is None else self.optimizer.last_stepsize

        for cb in self.callback:
            cb(
                self.optimizer.nfev,
                params,
                fx_value,
                step_size,
                accepted,
            )

    def _run_optimizer_callback_if_present(
        self,
        params: np.ndarray,
        fx_value: float,
        accepted: bool,
    ) -> None:
        """
        Execute optimizer-level callback, if present.

        Parameters
        ----------
        params : np.ndarray
            Current parameter vector.
        fx_value : float
            Objective value associated with ``params``.
        accepted : bool
            Whether the step was accepted.
        """
        if self.optimizer.callback is None:
            return

        step_size = 0.0 if self.optimizer.last_stepsize is None else self.optimizer.last_stepsize

        self.optimizer.callback(
            self.optimizer.nfev,
            params,
            fx_value,
            step_size,
            accepted,
        )

    def train_one_time(
        self,
        ansatz: QuantumCircuit,
        loss_function: Callable[[np.ndarray], float],
        initial_point: list[float] | np.ndarray | None = None,
        loss_next: Callable[[np.ndarray], float] | None = None,
        iterations: int = 100,
        iteration_start: int | None = None,
        **kwargs,
    ) -> OptimizerResult:
        """
        Train a fixed ansatz for a given number of iterations.

        Parameters
        ----------
        ansatz : QuantumCircuit
            Current ansatz circuit.
        loss_function : Callable[[np.ndarray], float]
            Objective function to minimize.
        initial_point : list[float] | np.ndarray | None, optional
            Initial parameter vector. If None, a random +/-1 vector is used.
        loss_next : Callable[[np.ndarray], float] | None, optional
            Optional objective used to evaluate the proposed next point.
        iterations : int, optional
            Number of optimization steps.
        iteration_start : int | None, optional
            Optional starting iteration index passed to the optimizer
            initialization. Useful for continuation runs with schedule-based
            optimizers such as SPSA. If None, it will default to the optimizers
            initial step or the current iteration count if the optimizer has
            already been initialized.
        **kwargs
            Additional keyword arguments forwarded to the objective.

        Returns
        -------
        OptimizerResult
            Result object containing final parameters and objective value.
        """
        if self.optimizer is None:
            raise RuntimeError("No optimizer has been set for the trainer.")

        logger.info("Started minimization. Repetition count: %s.", self._times_trained)

        if initial_point is None:
            initial_point = [random.choice([-1, 1]) for _ in range(ansatz.num_parameters)]
            
        logger.info("Initial point: %s", initial_point)

        x = np.asarray(initial_point, dtype=float)
        loss_kwargs = {**kwargs, "ansatz": ansatz}

        self.optimizer.initialize(
            x,
            loss_function,
            iteration_start=iteration_start,
            **loss_kwargs,
        )

        start = time()
        k = 0

        while k < iterations:
            k += 1
            iteration_begin = time()

            skip, x_next, fx_next, gradient_estimate, fx_estimate = self.step(
                ansatz,
                loss_function,
                x,
                loss_next=loss_next,
                **kwargs,
            )

            if skip:
                logger.info(
                    "Iteration %s/%s rejected in %s.",
                    k,
                    iterations,
                    time() - iteration_begin,
                )
                continue

            x = x_next

            if self.track_gradients and gradient_estimate is not None:
                self.gradient_history[self._times_trained].append(
                    np.asarray(gradient_estimate, dtype=float)
                )

            fx_callback = fx_estimate if fx_next is None else fx_next
            fx_callback = float(fx_callback)

            self._run_optimizer_callback_if_present(x, fx_callback, True)
            self._run_trainer_callbacks(x, fx_callback, True)

            checker = self.termination_checker
            if checker is None:
                checker = self.optimizer.termination_checker

            if checker is not None:
                step_size = 0.0 if self.optimizer.last_stepsize is None else self.optimizer.last_stepsize
                if checker(
                    self.optimizer.nfev,
                    x,
                    fx_callback,
                    step_size,
                    True,
                ):
                    logger.info("Terminated optimization at iteration %s/%s.", k, iterations)
                    break

            logger.info(
                "Iteration %s/%s done in %s.",
                k,
                iterations,
                time() - iteration_begin,
            )

        logger.info("Finished inner-loop optimization in %s seconds.", time() - start)

        self._times_trained += 1

        if self.track_gradients:
            self.gradient_history[self._times_trained] = []

        result = OptimizerResult()
        result.x = x

        if loss_next is None:
            logger.info("Calculating cost function value for final parameters.")
            result.fun = float(loss_function(x, ansatz=ansatz, **kwargs))
        else:
            logger.info("Calculating custom cost function value for final parameters.")
            result.fun = float(loss_next(x, ansatz=ansatz, **kwargs))

        result.nfev = self.optimizer.nfev
        result.nit = k

        self._last_cost = result.fun
        self._last_params = np.asarray(x, dtype=float)
        self._last_num_iterations = k

        return result
