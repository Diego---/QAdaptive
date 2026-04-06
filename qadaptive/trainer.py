import logging
import random
from time import time
from typing import Callable, SupportsFloat

import numpy as np
from qiskit import QuantumCircuit
from qiskit_algorithms.optimizers.optimizer import Optimizer, OptimizerResult

from qae.optimization.my_spsa import SPSA

CALLBACK = Callable[[int, np.ndarray, float, SupportsFloat, bool], None]
TERMINATIONCHECKER = Callable[[int, np.ndarray, float, SupportsFloat, bool], bool]

logger = logging.getLogger(__name__)

class InnerLoopTrainer:
    """
    SPSA-specific inner-loop trainer for adaptive ansatz optimization.

    This class encapsulates the optimizer-facing logic that performs parameter
    updates for a fixed ansatz structure. It is currently designed around the
    custom SPSA implementation used in this project.
    
    Attributes
    ----------
    optimizer : Optimizer
        Optimizer instance used for training. It can be set at initialization 
        or later via `set_optimizer`.
    track_gradients : bool
        Whether to store gradient estimates during training.
    callback : CALLBACK | list[CALLBACK] | None
        One callback or a list of callbacks to be run after each accepted step.
    termination_checker : TERMINATIONCHECKER | None
        Optional termination checker called during training.
    gradient_history : dict[int, list[np.ndarray]] | None
        History of gradient estimates for each training repetition, indexed by the 
        repetition count. Only populated if `track_gradients` is True.

    Notes
    -----
    This trainer currently assumes an optimizer interface with the methods
    `compute_loss_and_gradient_estimate` and `process_update`, as implemented
    by the custom SPSA used in this project.
    """

    def __init__(
        self,
        optimizer: Optimizer | None = None,
        optimizer_options: dict | None = None,
        track_gradients: bool = True,
        callback: CALLBACK | list[CALLBACK] | None = None,
        termination_checker: TERMINATIONCHECKER | None = None,
    ) -> None:
        """
        Initialize an instance of an inner-loop trainer.

        Parameters
        ----------
        optimizer : Optimizer | None, optional
            Pre-initialized optimizer instance. If provided, it is used directly.
        optimizer_options : dict | None, optional
            Keyword arguments used to initialize the custom SPSA optimizer when
            `optimizer` is not provided.
        track_gradients : bool, optional
            Whether to store gradient estimates during training.
        callback : CALLBACK | list[CALLBACK] | None, optional
            One callback or a list of callbacks to be run after each accepted step.
        termination_checker : TERMINATIONCHECKER | None, optional
            Optional termination checker called during training.
        """
        assert (
            optimizer is not None or optimizer_options is not None
        ), "Provide at least a pre-initialized optimizer or options to initialize a new one."
        self.optimizer = optimizer if optimizer is not None else SPSA(**optimizer_options)
        self.track_gradients = track_gradients

        if callback is not None and not isinstance(callback, list):
            self.callback = [callback]
        else:
            self.callback = callback

        self.termination_checker = termination_checker

        # Training state
        self.gradient_history = {0: []} if track_gradients else None
        self._times_trained = 0
        self._inner_iteration = 0
        self._last_cost = 0.0
        self._last_params = np.array([])

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

        This method is intended for situations where the experiment accepts a
        structural change, such as gate pruning or circuit compression, and wants
        to store the corresponding objective value without running a full inner-loop
        training cycle again.

        Parameters
        ----------
        cost : float
            Objective value associated with the accepted ansatz state.
        params : list[float] | np.ndarray | None, optional
            Parameter vector associated with the accepted ansatz state. If provided,
            it replaces the currently stored parameter vector. If None, the existing
            cached parameter vector is left unchanged.

        Notes
        -----
        This helper does not perform any optimization step and does not update
        gradient history, iteration counters, or optimizer-internal statistics. It
        only refreshes the cached ``last_cost`` and, optionally, ``last_params``
        values used by the outer-loop logic.

        Examples
        --------
        After accepting a pruning step, update only the stored cost:

        >>> trainer.update_last_evaluation(cost=trial_cost)

        After accepting a structural change that also changes the parameter vector:

        >>> trainer.update_last_evaluation(cost=new_cost, params=new_params)
        """
        self._last_cost = float(cost)

        if params is not None:
            self._last_params = np.asarray(params, dtype=float)

    def set_optimizer(
        self,
        optimizer: Optimizer | None = None,
        optimizer_options: dict | None = None,
    ) -> None:
        """
        Set or replace the optimizer used by the trainer.

        Parameters
        ----------
        optimizer : Optimizer | None, optional
            Pre-initialized optimizer instance.
        optimizer_options : dict | None, optional
            Keyword arguments used to initialize the custom SPSA optimizer when
            `optimizer` is not provided.

        Raises
        ------
        ValueError
            If both `optimizer` and `optimizer_options` are None.
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
    ) -> tuple[bool, np.ndarray, float | None, np.ndarray, float]:
        """
        Perform one optimization step for the current ansatz.

        Parameters
        ----------
        ansatz : QuantumCircuit
            Current AdaptiveAnsatz ansatz.
        loss_function : Callable[[np.ndarray], float]
            Objective function to minimize. It must accept the parameter vector
            and the keyword argument `ansatz`.
        x : np.ndarray
            Current parameter vector.
        loss_next : Callable[[np.ndarray], float] | None, optional
            Optional objective used to evaluate the next point.
        **kwargs
            Additional keyword arguments forwarded to the loss function.

        Returns
        -------
        tuple[bool, np.ndarray, float | None, np.ndarray, float]
            A tuple containing:
            - whether the step was skipped,
            - the next parameter vector,
            - the next objective value if available,
            - the gradient estimate,
            - the current objective estimate.
        """
        if self.optimizer is None:
            raise RuntimeError(
                "The optimizer is not set. Set an optimizer before running a training step."
            )

        x = np.asarray(x, dtype=float)
        iteration_start = time()
        loss_kwargs = {**kwargs, "ansatz": ansatz}

        fx_estimate, gradient_estimate = self.optimizer.compute_loss_and_gradient_estimate(
            loss_function,
            x,
            **loss_kwargs,
        )

        skip, x_next, fx_next = self.optimizer.process_update(
            gradient_estimate,
            x,
            fx_estimate,
            loss_next,
            iteration_start,
            self._inner_iteration,
        )

        if not skip:
            self.optimizer.last_iteration += 1

        return skip, x_next, fx_next, gradient_estimate, fx_estimate

    def train_one_time(
        self,
        ansatz: QuantumCircuit,
        loss_function: Callable[[np.ndarray], float],
        initial_point: list | np.ndarray | None = None,
        loss_next: Callable[[np.ndarray], float] | None = None,
        iterations: int = 100,
        **kwargs,
    ) -> OptimizerResult:
        """
        Train a fixed ansatz for a given number of iterations.

        Parameters
        ----------
        ansatz : QuantumCircuit
            Current AdaptiveAnsatz circuit.
        loss_function : Callable[[np.ndarray], float]
            Objective function to minimize.
        initial_point : list | np.ndarray | None, optional
            Initial parameter vector. If None, a random +/-1 vector is used.
        loss_next : Callable[[np.ndarray], float] | None, optional
            Optional objective used to evaluate the next point.
        iterations : int, optional
            Number of inner-loop optimization steps.
        **kwargs
            Additional configuration. Currently supports:
            - `use_epochs`
            - `num_circs_per_group`
            - `num_circs_per_batch`

        Returns
        -------
        OptimizerResult
            Result object containing the final point and objective value.

        Notes
        -----
        This implementation is intentionally tailored to the custom SPSA used
        in this project.
        """
        if self.optimizer is None:
            raise RuntimeError("No optimizer has been set for the trainer.")

        logger.info("Started minimization. Repetition count: %s.", self._times_trained)

        if self.optimizer.blocking:
            raise NotImplementedError("Training with blocking is not yet implemented.")

        if initial_point is None:
            initial_point = [random.choice([-1, 1]) for _ in range(ansatz.num_parameters)]
        x = np.asarray(initial_point, dtype=float)

        if self.optimizer.p_iterator is None and self.optimizer.lr_iterator is None:
            self.optimizer._create_iterators(loss_function, initial_point)

        use_epochs = kwargs.get("use_epochs", False)
        ncpg = kwargs.get("num_circs_per_group")
        ncpb = kwargs.get("num_circs_per_batch")

        if use_epochs:
            logger.info("Starting optimization in epoch mode with initial parameters %s.", x)
            total_circuits = (
                self.optimizer._size_full_batch
                if self.optimizer._size_full_batch is not None
                else 12
            )
            if ncpb is None:
                ncpb = 3
        else:
            logger.info("Starting optimization with initial parameters %s.", x)

        logger.info("Setting number of circuits per batch to %s.", ncpb)

        start = time()
        k = 0

        while k < iterations:
            k += 1
            current_learn_rate = next(self.optimizer._lr_iterator_copy)
            self._inner_iteration += 1
            iteration_start = time()
            # Compute updates for the whole batched dataset when using epochs
            if use_epochs:
                indices = np.arange(total_circuits)
                np.random.shuffle(indices)
                batch_indices = [
                    indices[i:i + ncpb] for i in range(0, total_circuits, ncpb)
                ]

                for batch_id, used_indices in enumerate(batch_indices, start=1):
                    logger.info(
                        "Evaluating batch %s out of %s.",
                        batch_id,
                        len(batch_indices),
                    )
                    skip, x_next, fx_next, gradient_estimate, fx_estimate = self.step(
                        ansatz,
                        loss_function,
                        x,
                        loss_next=loss_next,
                        used_circs_indices=used_indices,
                    )
                    if skip:
                        continue
                    x = x_next

                logger.info("Epoch %s/%s finished in %s.", k, iterations, time() - iteration_start)

            else:
                skip, x_next, fx_next, gradient_estimate, fx_estimate = self.step(
                    ansatz,
                    loss_function,
                    x,
                    loss_next=loss_next,
                    num_circs_per_group=ncpg,
                )
                if skip:
                    continue

                x = x_next
                logger.info("Iteration %s/%s done in %s.", k, iterations, time() - iteration_start)

            if self.optimizer.callback is not None:
                if loss_next is None:
                    logger.info(
                        "Calculating next-step value for optimizer callback."
                    )
                    self.optimizer._nfev += 1
                    fx_next = loss_function(x, ansatz=ansatz)
                else:
                    logger.info(
                        "Calculating next-step value with custom objective for optimizer callback."
                    )
                    self.optimizer._nextfev += 1
                    fx_next = loss_next(x, ansatz=ansatz)

                self.optimizer.callback(
                    self.optimizer._nfev,
                    x,  # next parameters
                    fx_next,
                    np.linalg.norm(gradient_estimate * current_learn_rate),
                    True,
                )

            if self.track_gradients:
                self.gradient_history[self._times_trained].append(gradient_estimate)

            checker = self.optimizer.termination_checker or self.termination_checker
            if checker is not None:
                fx_check = fx_estimate if fx_next is None else fx_next
                if checker(
                    self.optimizer._nfev,
                    x,
                    fx_check,
                    np.linalg.norm(gradient_estimate * current_learn_rate),
                    True,
                ):
                    logger.info("Terminated optimization at iteration %s/%s.", k, iterations)
                    break

        logger.info("Finished inner-loop optimization in %s seconds.", time() - start)

        self._inner_iteration = 0
        self._times_trained += 1

        if self.track_gradients:
            self.gradient_history[self._times_trained] = []

        result = OptimizerResult()
        result.x = x

        if loss_next is None:
            logger.info("Calculating cost funtion value for final parameters.")
            result.fun = loss_function(x, ansatz=ansatz)
        else:
            logger.info("Calculating custom cost funtion value for final parameters.")
            result.fun = loss_next(x, ansatz=ansatz)

        result.nfev = self.optimizer._nfev
        result.nit = k

        self._last_cost = result.fun
        self._last_params = np.asarray(x, dtype=float)

        return result
