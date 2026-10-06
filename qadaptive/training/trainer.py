import logging
import random
from time import time
from typing import Callable

import numpy as np
from qiskit import QuantumCircuit
from qiskit_algorithms.optimizers.optimizer import OptimizerResult

from .optimizers import SPSA
from .optimizers.stepwise_optimizer import StepwiseOptimizer, TERMINATIONCHECKER
from .recorder import InnerLoopRecorder

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
    recorder : InnerLoopRecorder
        Persistent recorder that owns all inner-loop history.
    termination_checker : TERMINATIONCHECKER | None
        Optional trainer-level termination checker.
    gradient_history : dict[int, list[np.ndarray]] | None
        Gradient history indexed by training repetition.
    """

    def __init__(
        self,
        optimizer: StepwiseOptimizer | None = None,
        optimizer_options: dict | None = None,
        recorder: InnerLoopRecorder | None = None,
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
        recorder : InnerLoopRecorder | None, optional
            Persistent recorder for all inner-loop runs. If omitted, an empty
            recorder is created automatically.
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
        self.recorder = InnerLoopRecorder() if recorder is None else recorder
        self.termination_checker = termination_checker

        # Training state
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

    @property
    def training_run_history(self):
        """Return the recorder-owned training runs."""
        return self.recorder.runs

    @property
    def last_training_run_record(self):
        """Return the most recently started training run."""
        return self.recorder.last_run

    @property
    def gradient_history(self) -> dict[int, list[np.ndarray]] | None:
        """Return recorder-owned gradients grouped by training run."""
        if not self.recorder.record_gradients:
            return None
        return self.recorder.gradient_history

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
        **kwargs,
    ) -> tuple[bool, np.ndarray, float | None, np.ndarray | None, float | None]:
        """Perform one optimizer step for the current fixed ansatz."""
        if self.optimizer is None:
            raise RuntimeError(
                "The optimizer is not set. Set an optimizer before running a training step."
            )

        x = np.asarray(x, dtype=float)
        loss_kwargs = {**kwargs, "ansatz": ansatz}
        return self.optimizer.step(x, loss_function, **loss_kwargs)

    def train_one_time(
        self,
        ansatz: QuantumCircuit,
        loss_function: Callable[[np.ndarray], float],
        initial_point: list[float] | np.ndarray | None = None,
        evaluation_loss: Callable[[np.ndarray], float] | None = None,
        iterations: int = 100,
        iteration_start: int | None = None,
        initial_value: float | None = None,
        outer_iteration: int | None = None,
        action: str | None = None,
        restart_parameter_schedules: bool = False,
        note: str | None = None,
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
        evaluation_loss : Callable[[np.ndarray], float] | None, optional
            Optional objective evaluated at every accepted updated parameter point. If None, no additional per-step evaluation is performed.
        iterations : int, optional
            Number of optimization steps.
        iteration_start : int | None, optional
            Optional starting iteration index passed to the optimizer
            initialization. Useful for continuation runs with schedule-based
            optimizers such as SPSA. If None, it will default to the optimizers
            initial step or the current iteration count if the optimizer has
            already been initialized.
        initial_value : float | None, optional
            Objective value at the initial point, if already available. This is
            stored without another evaluation. If None and the recorder was
            configured with ``record_initial_value=True``, it is evaluated here.
        outer_iteration : int | None, optional
            Outer-loop iteration associated with this training run.
        action : str | None, optional
            Structural action associated with this training run.
        restart_parameter_schedules : bool, optional
            If True, restart the optimizer's per-parameter schedules after optimizer
            initialization and before the first training step. Defaults to False.
        note : str | None, optional
            Optional run annotation.
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
        
        param_names = [parameter.name for parameter in ansatz.parameters]
        if len(x) != len(param_names):
            raise ValueError(
                "Length of initial_point does not match number of ansatz parameters."
            )

        if initial_value is None and self.recorder.record_initial_value:
            objective_for_initial = loss_function if evaluation_loss is None else evaluation_loss
            try:
                initial_value = float(
                    objective_for_initial(x, ansatz=ansatz, **kwargs)
                )
            except TypeError:
                initial_value = float(objective_for_initial(x, ansatz))

        self.recorder.start_run(
            param_names=param_names,
            initial_point=x,
            initial_value=initial_value,
            outer_iteration=outer_iteration,
            action=action,
            note=note,
        )

        start = time()
        k = 0
        last_evaluation_value: float | None = None

        try:
            self.optimizer.initialize(
                x,
                loss_function,
                iteration_start=iteration_start,
                parameter_names=param_names,
                outer_iteration=outer_iteration,
                **loss_kwargs,
            )
            
            birth_outer_iterations = getattr(
                self.optimizer,
                "parameter_birth_outer_iterations",
                None,
            )

            if birth_outer_iterations:
                self.recorder.set_parameter_birth_outer_iterations(
                    birth_outer_iterations
                )
            
            if restart_parameter_schedules:
                restart_schedules = getattr(
                    self.optimizer,
                    "restart_parameter_schedules",
                    None,
                )

                if restart_schedules is None:
                    raise RuntimeError(
                        "Parameter schedule restart was requested, but the active "
                        "optimizer does not support parameter-dependent schedules."
                    )

                restart_schedules()

            while k < iterations:
                k += 1
                iteration_begin = time()

                x_before = np.asarray(x, dtype=float).copy()
                skip, x_next, fx_next, gradient_estimate, fx_estimate = self.step(
                    ansatz,
                    loss_function,
                    x,
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

                x = np.asarray(x_next, dtype=float)

                evaluation_value = None
                if evaluation_loss is not None:
                    if fx_next is not None and evaluation_loss is loss_function:
                        evaluation_value = float(fx_next)
                    else:
                        evaluation_value = float(
                            evaluation_loss(x, ansatz=ansatz, **kwargs)
                        )
                elif fx_next is not None:
                    evaluation_value = float(fx_next)

                if evaluation_loss is not None and evaluation_value is not None:
                    last_evaluation_value = float(evaluation_value)

                step_size = (
                    0.0
                    if self.optimizer.last_stepsize is None
                    else float(self.optimizer.last_stepsize)
                )

                self.recorder(
                    iteration=k,
                    nfev=self.optimizer.nfev,
                    params=x,
                    optimizer_estimate=float(fx_estimate),
                    evaluation_value=evaluation_value,
                    stepsize=step_size,
                    accepted=True,
                    gradient=gradient_estimate,
                    schedule_steps_used=getattr(
                        self.optimizer,
                        "last_parameter_schedule_steps_used",
                        None,
                    ),
                    learning_rates=getattr(
                        self.optimizer,
                        "last_parameter_learning_rates",
                        None,
                    ),
                    perturbations=getattr(
                        self.optimizer,
                        "last_parameter_perturbations",
                        None,
                    ),
                )

                checker = self.termination_checker
                if checker is None:
                    checker = self.optimizer.termination_checker

                checker_params = x if evaluation_value is not None else x_before
                checker_value = float(evaluation_value) if evaluation_value is not None else float(fx_estimate)

                if checker is not None and checker(
                    self.optimizer.nfev,
                    checker_params,
                    checker_value,
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

            result = OptimizerResult()
            result.x = x

            if evaluation_loss is None:
                logger.info("Calculating objective value for final parameters.")
                result.fun = float(loss_function(x, ansatz=ansatz, **kwargs))
            elif last_evaluation_value is not None:
                logger.info(
                    "Reusing explicit evaluation already recorded at final parameters."
                )
                result.fun = float(last_evaluation_value)
            else:
                logger.info("Calculating evaluation objective for final parameters.")
                result.fun = float(evaluation_loss(x, ansatz=ansatz, **kwargs))

            logger.info("Final objective value: %s", result.fun)

            result.nfev = self.optimizer.nfev
            result.nit = k

            self._last_cost = result.fun
            self._last_params = np.asarray(x, dtype=float)
            self._last_num_iterations = k
            self._times_trained += 1

            self.recorder.finish_run(
                final_params=result.x,
                final_value=result.fun,
            )
            return result
        except BaseException as error:
            self.recorder.abort_run(
                note=f"Aborted after {k} iterations: {type(error).__name__}: {error}"
            )
            raise
