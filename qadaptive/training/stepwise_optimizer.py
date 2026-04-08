from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import SupportsFloat

import numpy as np
from qiskit_algorithms.optimizers.optimizer import Optimizer

CALLBACK = Callable[[int, np.ndarray, float, SupportsFloat, bool], None]
TERMINATIONCHECKER = Callable[[int, np.ndarray, float, SupportsFloat, bool], bool]

logger = logging.getLogger(__name__)

class StepwiseOptimizer(Optimizer, ABC):
    """
    Base class for optimizers that support explicit step-by-step training.

    Concrete subclasses must implement initialization and single-step update
    logic so they can be used by ``InnerLoopTrainer``.
    """

    def __init__(
        self,
        callback: CALLBACK | None = None,
        termination_checker: TERMINATIONCHECKER | None = None,
    ) -> None:
        super().__init__()
        self.callback = callback
        self.termination_checker = termination_checker

        self._nfev: int = 0
        self._nextfev: int = 0
        self._initialized: bool = False
        self._last_stepsize: float | None = None
        self._last_gradient: np.ndarray | None = None
        self._last_fx: float | None = None
        self._iteration: int = 0

    @property
    def nfev(self) -> int:
        """Return the number of objective evaluations."""
        return self._nfev

    @property
    def nextfev(self) -> int:
        """Return the number of next-point objective evaluations."""
        return self._nextfev

    @property
    def iteration(self) -> int:
        """Return the current optimizer iteration."""
        return self._iteration

    @property
    def last_stepsize(self) -> float | None:
        """Return the norm of the most recent accepted update."""
        return self._last_stepsize

    @property
    def last_gradient(self) -> np.ndarray | None:
        """Return the most recent gradient estimate."""
        return self._last_gradient

    @property
    def last_fx(self) -> float | None:
        """Return the most recent function value estimate."""
        return self._last_fx

    @abstractmethod
    def initialize(
        self,
        x0: np.ndarray,
        loss_function: Callable[[np.ndarray], float],
        **kwargs,
    ) -> None:
        """Initialize optimizer state for a new training run."""

    @abstractmethod
    def step(
        self,
        x: np.ndarray,
        loss_function: Callable[[np.ndarray], float],
        loss_next: Callable[[np.ndarray], float] | None = None,
        **kwargs,
    ) -> tuple[bool, np.ndarray, float | None, np.ndarray | None, float | None]:
        """
        Perform one optimization step.

        Returns
        -------
        tuple
            ``(skip, x_next, fx_next, gradient_estimate, fx_estimate)``
        """

    def reset_runtime_state(self, iteration_start: int = 0) -> None:
        """
        Reset generic runtime bookkeeping for a new optimization run.

        This method clears evaluation counters and cached per-step quantities,
        while optionally setting the internal iteration counter to a non-zero
        starting value. This is useful for optimizers with iteration-dependent
        schedules (e.g., SPSA learning rate and perturbation sequences), where
        one may wish to resume optimization with appropriately decayed step sizes.

        Parameters
        ----------
        iteration_start : int, optional
            The iteration index to start from. Defaults to 0.

            - For optimizers with iteration-dependent schedules (e.g., SPSA),
            this value should correspond to the desired position in the
            learning-rate / perturbation sequences.
            - For optimizers without such schedules (e.g., ADAM), this parameter
            has no practical effect beyond bookkeeping.

        Notes
        -----
        This method does not reset optimizer-specific internal state such as
        moment estimates (ADAM) or Hessian approximations (2-SPSA). Subclasses
        should override or extend initialization behavior in ``initialize(...)``
        if such state needs to be reset or preserved.
        """
        logger.info(
            "Resetting optimizer runtime state for new optimization run from iteration %d.",
            iteration_start
            )
        self._nfev = 0
        self._nextfev = 0
        self._initialized = False

        self._last_stepsize = None
        self._last_gradient = None
        self._last_fx = None

        self._iteration = iteration_start
