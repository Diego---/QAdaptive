# This code is part of Qiskit.
#
# (C) Copyright IBM 2019, 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

# This file has been modified from its original version.

"""Stepwise ADAM / AMSGRAD optimizer."""

from __future__ import annotations

import csv
import logging
import os
from collections.abc import Callable
from typing import Any

import numpy as np
from qiskit.utils.deprecation import deprecate_arg
from qiskit_algorithms.optimizers.optimizer import (
    OptimizerResult,
    OptimizerSupportLevel,
    POINT,
)

from ..stepwise_optimizer import StepwiseOptimizer, CALLBACK, TERMINATIONCHECKER

logger = logging.getLogger(__name__)


class ADAM(StepwiseOptimizer):
    """
    Adam and AMSGRAD optimizers with explicit stepwise training support.

    This implementation is designed to work with the project's
    ``StepwiseOptimizer`` interface and ``InnerLoopTrainer``.
    """

    _OPTIONS = [
        "maxiter",
        "tol",
        "lr",
        "beta_1",
        "beta_2",
        "noise_factor",
        "eps",
        "amsgrad",
        "snapshot_dir",
        "blocking",
        "allowed_increase",
        "trust_region",
    ]

    def __init__(
        self,
        maxiter: int = 10000,
        tol: float = 1e-6,
        lr: float = 1e-3,
        beta_1: float = 0.9,
        beta_2: float = 0.99,
        noise_factor: float = 1e-8,
        eps: float = 1e-10,
        amsgrad: bool = False,
        snapshot_dir: str | None = None,
        blocking: bool = False,
        allowed_increase: float | None = None,
        trust_region: bool = False,
        callback: CALLBACK | None = None,
        termination_checker: TERMINATIONCHECKER | None = None,
    ) -> None:
        """
        Parameters
        ----------
        maxiter : int, optional
            Maximum number of iterations when using ``minimize``.
        tol : float, optional
            Norm tolerance on the parameter update for termination in ``minimize``.
        lr : float, optional
            Base learning rate.
        beta_1 : float, optional
            Exponential decay factor for the first moment.
        beta_2 : float, optional
            Exponential decay factor for the second moment.
        noise_factor : float, optional
            Small stabilizer added to the denominator.
        eps : float, optional
            Finite-difference step size used when no analytic gradient is supplied.
        amsgrad : bool, optional
            Whether to use AMSGRAD instead of standard ADAM.
        snapshot_dir : str | None, optional
            Optional directory in which optimizer state is appended after each step.
        blocking : bool, optional
            If True, reject steps whose next-point objective increases by more than
            ``allowed_increase``.
        allowed_increase : float | None, optional
            Maximum allowed increase for blocking mode. If None and blocking is enabled,
            the effective threshold is taken as 0.0.
        trust_region : bool, optional
            If True, normalize the raw gradient to norm at most 1 before applying
            the ADAM moment update.
        callback : CALLBACK | None, optional
            Optional callback. Retained for compatibility with the current base class.
        termination_checker : TERMINATIONCHECKER | None, optional
            Optional termination checker. Retained for compatibility with the current
            base class.
        """
        super().__init__(callback=callback, termination_checker=termination_checker)

        for key, value in list(locals().items()):
            if key in self._OPTIONS:
                self._options[key] = value

        self._maxiter = maxiter
        self._tol = tol
        self._lr = lr
        self._beta_1 = beta_1
        self._beta_2 = beta_2
        self._noise_factor = noise_factor
        self._eps = eps
        self._amsgrad = amsgrad
        self._snapshot_dir = snapshot_dir
        self.blocking = blocking
        self.allowed_increase = allowed_increase
        self.trust_region = trust_region

        # ADAM-specific runtime state
        self._t = 0
        self._m = np.zeros(1, dtype=float)
        self._v = np.zeros(1, dtype=float)
        self._last_update = np.zeros(1, dtype=float)

        if self._amsgrad:
            self._v_eff = np.zeros(1, dtype=float)

        if self._snapshot_dir:
            self._initialize_snapshot_file()

    @property
    def settings(self) -> dict[str, Any]:
        """
        Return optimizer settings.

        Returns
        -------
        dict[str, Any]
            Dictionary of optimizer configuration values.
        """
        return {
            "maxiter": self._maxiter,
            "tol": self._tol,
            "lr": self._lr,
            "beta_1": self._beta_1,
            "beta_2": self._beta_2,
            "noise_factor": self._noise_factor,
            "eps": self._eps,
            "amsgrad": self._amsgrad,
            "snapshot_dir": self._snapshot_dir,
            "blocking": self.blocking,
            "allowed_increase": self.allowed_increase,
            "trust_region": self.trust_region,
            "callback": self.callback,
            "termination_checker": self.termination_checker,
        }

    def get_support_level(self) -> dict[str, OptimizerSupportLevel]:
        """
        Return support levels for optimizer features.

        Returns
        -------
        dict[str, OptimizerSupportLevel]
            Support levels for gradient, bounds, and initial point.
        """
        return {
            "gradient": OptimizerSupportLevel.supported,
            "bounds": OptimizerSupportLevel.ignored,
            "initial_point": OptimizerSupportLevel.supported,
        }

    def set_learning_rate(self, value: float) -> None:
        """
        Set the ADAM learning rate.

        Parameters
        ----------
        value : float
            New learning rate.
        """
        self._lr = value

    def set_allowed_increase(self, value: float | None) -> None:
        """
        Set the blocking threshold.

        Parameters
        ----------
        value : float | None
            Allowed objective increase used when ``blocking=True``.
        """
        self.allowed_increase = value

    def _initialize_snapshot_file(self) -> None:
        """Create or overwrite the optimizer snapshot CSV file."""
        os.makedirs(self._snapshot_dir, exist_ok=True)
        filepath = os.path.join(self._snapshot_dir, "adam_params.csv")

        with open(filepath, mode="w", newline="") as csv_file:
            if self._amsgrad:
                fieldnames = ["v", "v_eff", "m", "t"]
            else:
                fieldnames = ["v", "m", "t"]

            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()

    def save_params(self, snapshot_dir: str) -> None:
        """
        Append the current optimizer state to ``adam_params.csv``.

        Parameters
        ----------
        snapshot_dir : str
            Directory in which the snapshot file is stored.
        """
        filepath = os.path.join(snapshot_dir, "adam_params.csv")

        if self._amsgrad:
            with open(filepath, mode="a", newline="") as csv_file:
                writer = csv.DictWriter(csv_file, fieldnames=["v", "v_eff", "m", "t"])
                writer.writerow(
                    {
                        "v": self._v.tolist(),
                        "v_eff": self._v_eff.tolist(),
                        "m": self._m.tolist(),
                        "t": self._t,
                    }
                )
        else:
            with open(filepath, mode="a", newline="") as csv_file:
                writer = csv.DictWriter(csv_file, fieldnames=["v", "m", "t"])
                writer.writerow(
                    {
                        "v": self._v.tolist(),
                        "m": self._m.tolist(),
                        "t": self._t,
                    }
                )

    def load_params(self, load_dir: str) -> None:
        """
        Load optimizer state from ``adam_params.csv``.

        Parameters
        ----------
        load_dir : str
            Directory containing the snapshot file.
        """
        filepath = os.path.join(load_dir, "adam_params.csv")

        with open(filepath, newline="") as csv_file:
            reader = csv.DictReader(csv_file)
            last_row = None
            for row in reader:
                last_row = row

        if last_row is None:
            raise ValueError(f"No optimizer state found in '{filepath}'.")

        self._v = np.fromstring(last_row["v"].strip("[]"), dtype=float, sep=",")
        self._m = np.fromstring(last_row["m"].strip("[]"), dtype=float, sep=",")
        self._t = int(last_row["t"])

        if self._amsgrad:
            self._v_eff = np.fromstring(
                last_row["v_eff"].strip("[]"),
                dtype=float,
                sep=",",
            )

        self._last_update = np.zeros_like(self._m, dtype=float)

    def _ensure_state_shape(self, x: np.ndarray) -> None:
        """
        Resize runtime state arrays to match the parameter dimension.

        Parameters
        ----------
        x : np.ndarray
            Current parameter vector.
        """
        if self._m.shape != x.shape:
            self._m = np.zeros_like(x, dtype=float)
            self._v = np.zeros_like(x, dtype=float)
            self._last_update = np.zeros_like(x, dtype=float)

            if self._amsgrad:
                self._v_eff = np.zeros_like(x, dtype=float)

    def _numeric_gradient(
        self,
        loss: Callable[[np.ndarray], float],
        x: np.ndarray,
        **kwargs,
    ) -> np.ndarray:
        """
        Compute a central-difference numerical gradient.

        Parameters
        ----------
        loss : Callable[[np.ndarray], float]
            Objective function.
        x : np.ndarray
            Point at which to estimate the gradient.
        **kwargs
            Additional keyword arguments forwarded to the objective.

        Returns
        -------
        np.ndarray
            Numerical gradient estimate.
        """
        grad = np.zeros_like(x, dtype=float)

        if x.size == 0:
            return grad

        for i in range(x.size):
            step = np.zeros_like(x, dtype=float)
            step[i] = self._eps

            f_plus = float(loss(x + step, **kwargs))
            f_minus = float(loss(x - step, **kwargs))
            grad[i] = (f_plus - f_minus) / (2.0 * self._eps)

        self._nfev += 2 * x.size
        return grad

    def compute_loss_and_gradient_estimate(
        self,
        loss: Callable[[np.ndarray], float],
        x: np.ndarray,
        iteration: int | None = None,
        **kwargs,
    ) -> tuple[float, np.ndarray]:
        """
        Compute the current loss and gradient estimate.

        Parameters
        ----------
        loss : Callable[[np.ndarray], float]
            Objective function.
        x : np.ndarray
            Current parameter vector.
        iteration : int | None, optional
            Unused. Present for symmetry with SPSA.
        **kwargs
            Additional keyword arguments forwarded to the objective.
            If ``jac`` is provided, it is used as the analytic gradient function.

        Returns
        -------
        tuple[float, np.ndarray]
            Current function value and gradient estimate.
        """
        del iteration

        x = np.asarray(x, dtype=float)
        self._ensure_state_shape(x)

        grad_kwargs = dict(kwargs)
        jac = grad_kwargs.pop("jac", None)

        fx = float(loss(x, **grad_kwargs))
        self._nfev += 1

        if jac is None:
            gradient = self._numeric_gradient(loss, x, **grad_kwargs)
        else:
            gradient = np.asarray(jac(x, **grad_kwargs), dtype=float)

        logger.debug("ADAM function estimate: %s", fx)
        logger.debug("ADAM gradient estimate: %s", gradient)

        return fx, gradient

    def process_update(
        self,
        gradient_estimate: np.ndarray,
        x: np.ndarray,
        fx: float,
        fun: Callable[[np.ndarray], float],
        fun_next: Callable[[np.ndarray], float] | None,
        iteration_start: float = 0.0,
        iteration: int = 0,
        **kwargs,
    ) -> tuple[bool, np.ndarray, float | None]:
        """
        Process one ADAM parameter update.

        Parameters
        ----------
        gradient_estimate : np.ndarray
            Gradient estimate at the current point.
        x : np.ndarray
            Current parameter vector.
        fx : float
            Current function value.
        fun : Callable[[np.ndarray], float]
            Objective function.
        fun_next : Callable[[np.ndarray], float] | None
            Optional objective used to evaluate the proposed next point.
        iteration_start : float, optional
            Iteration start time used only for logging.
        iteration : int, optional
            Global iteration index used only for logging.
        **kwargs
            Additional keyword arguments forwarded to ``fun`` or ``fun_next``.

        Returns
        -------
        tuple[bool, np.ndarray, float | None]
            Tuple ``(skip, x_next, fx_next)``.
        """
        del iteration_start

        x = np.asarray(x, dtype=float)
        gradient_estimate = np.asarray(gradient_estimate, dtype=float)
        self._ensure_state_shape(x)

        grad = gradient_estimate.copy()

        if self.trust_region:
            norm = np.linalg.norm(grad)
            if norm > 1.0:
                grad = grad / norm

        self._t += 1

        self._m = self._beta_1 * self._m + (1.0 - self._beta_1) * grad
        self._v = self._beta_2 * self._v + (1.0 - self._beta_2) * (grad**2)

        bias_corrected_lr = (
            self._lr
            * np.sqrt(1.0 - self._beta_2**self._t)
            / (1.0 - self._beta_1**self._t)
        )

        if self._amsgrad:
            self._v_eff = np.maximum(self._v_eff, self._v)
            denom = np.sqrt(self._v_eff) + self._noise_factor
        else:
            denom = np.sqrt(self._v) + self._noise_factor

        update = bias_corrected_lr * self._m / denom
        x_next = x - update
        self._last_update = update

        logger.debug("ADAM iteration %d", iteration)
        logger.debug("ADAM effective learning rate: %s", bias_corrected_lr)
        logger.debug("ADAM first moment m: %s", self._m)
        logger.debug("ADAM second moment v: %s", self._v)
        logger.debug("ADAM update vector: %s", update)
        logger.debug("ADAM step norm: %s", np.linalg.norm(update))
        logger.debug("ADAM next parameters: %s", x_next)

        if self._snapshot_dir is not None:
            self.save_params(self._snapshot_dir)

        fx_next = None

        if self.blocking:
            eval_fun = fun if fun_next is None else fun_next
            fx_next = float(eval_fun(x_next, **kwargs))

            if fun_next is None:
                self._nfev += 1
            else:
                self._nextfev += 1

            allowed = 0.0 if self.allowed_increase is None else self.allowed_increase
            if fx_next > fx + allowed:
                logger.info(
                    "ADAM rejected iteration %d because fx_next=%s exceeded fx+allowed=%s.",
                    iteration,
                    fx_next,
                    fx + allowed,
                )
                return True, x, fx

        return False, x_next, fx_next

    def initialize(
        self,
        x0: np.ndarray,
        loss_function: Callable[[np.ndarray], float],
        iteration_start: int | None = None,
        **kwargs,
    ) -> None:
        """
        Initialize optimizer state for a new run.

        Parameters
        ----------
        x0 : np.ndarray
            Initial parameter vector.
        loss_function : Callable[[np.ndarray], float]
            Objective function. Included for interface consistency.
        iteration_start : int | None, optional
            Starting iteration index for bookkeeping. For ADAM this does not
            affect the moment reset, which is always fresh on initialization.
        **kwargs
            Unused. Present for interface compatibility.
        """
        del loss_function, kwargs

        x0 = np.asarray(x0, dtype=float)
        start = 0 if iteration_start is None else iteration_start

        self.reset_runtime_state(iteration_start=start)

        self._t = 0
        self._m = np.zeros_like(x0, dtype=float)
        self._v = np.zeros_like(x0, dtype=float)
        self._last_update = np.zeros_like(x0, dtype=float)

        if self._amsgrad:
            self._v_eff = np.zeros_like(x0, dtype=float)

        self._initialized = True

        logger.info(
            "Initialized ADAM optimizer for %d parameters (iteration_start=%d).",
            x0.size,
            start,
        )

    def step(
        self,
        x: np.ndarray,
        loss_function: Callable[[np.ndarray], float],
        loss_next: Callable[[np.ndarray], float] | None = None,
        **kwargs,
    ) -> tuple[bool, np.ndarray, float | None, np.ndarray | None, float | None]:
        """
        Perform one ADAM optimization step.

        Parameters
        ----------
        x : np.ndarray
            Current parameter vector.
        loss_function : Callable[[np.ndarray], float]
            Objective function.
        loss_next : Callable[[np.ndarray], float] | None, optional
            Optional objective used to evaluate the proposed next point.
        **kwargs
            Additional keyword arguments forwarded to the objective.

        Returns
        -------
        tuple[bool, np.ndarray, float | None, np.ndarray | None, float | None]
            Tuple ``(skip, x_next, fx_next, gradient_estimate, fx_estimate)``.
        """
        if not self._initialized:
            self.initialize(np.asarray(x, dtype=float), loss_function, **kwargs)

        x = np.asarray(x, dtype=float)
        next_iteration = self.iteration + 1

        fx_estimate, gradient_estimate = self.compute_loss_and_gradient_estimate(
            loss_function,
            x,
            iteration=next_iteration,
            **kwargs,
        )

        skip, x_next, fx_next = self.process_update(
            gradient_estimate,
            x,
            fx_estimate,
            loss_function,
            loss_next,
            iteration=next_iteration,
            **kwargs,
        )

        self._iteration = next_iteration
        self._last_gradient = np.asarray(gradient_estimate, dtype=float)
        self._last_fx = float(fx_estimate)
        self._last_stepsize = None if skip else float(np.linalg.norm(x_next - x))

        return skip, x_next, fx_next, gradient_estimate, fx_estimate

    @deprecate_arg(
        "objective_function",
        new_alias="fun",
        since="0.19.0",
        package_name="qiskit-terra",
    )
    @deprecate_arg(
        "initial_point",
        new_alias="fun",
        since="0.19.0",
        package_name="qiskit-terra",
    )
    @deprecate_arg(
        "gradient_function",
        new_alias="jac",
        since="0.19.0",
        package_name="qiskit-terra",
    )
    def minimize(
        self,
        fun: Callable[[POINT], float],
        x0: POINT,
        jac: Callable[[POINT], POINT] | None = None,
        bounds: list[tuple[float, float]] | None = None,
        objective_function: Callable[[np.ndarray], float] | None = None,
        initial_point: np.ndarray | None = None,
        gradient_function: Callable[[np.ndarray], float] | None = None,
    ) -> OptimizerResult:
        """
        Minimize a scalar objective using the standard Qiskit optimizer interface.

        Parameters
        ----------
        fun : Callable[[POINT], float]
            Objective function.
        x0 : POINT
            Initial point.
        jac : Callable[[POINT], POINT] | None, optional
            Analytic gradient function.
        bounds : list[tuple[float, float]] | None, optional
            Ignored.
        objective_function : Callable[[np.ndarray], float] | None, optional
            Deprecated alias.
        initial_point : np.ndarray | None, optional
            Deprecated alias.
        gradient_function : Callable[[np.ndarray], float] | None, optional
            Deprecated alias.

        Returns
        -------
        OptimizerResult
            Optimization result.
        """
        del bounds, objective_function, initial_point, gradient_function

        x = np.asarray(x0, dtype=float)
        self.initialize(x, fun)

        k = 0
        while k < self._maxiter:
            k += 1

            skip, x_next, fx_next, gradient_estimate, fx_estimate = self.step(
                x,
                fun,
                loss_next=None,
                jac=jac,
            )

            if skip:
                continue

            if np.linalg.norm(x_next - x) < self._tol:
                x = x_next
                break

            x = x_next

            if self.callback is not None:
                fx_cb = fx_estimate if fx_next is None else fx_next
                self.callback(
                    self.nfev,
                    x,
                    float(fx_cb),
                    0.0 if self.last_stepsize is None else self.last_stepsize,
                    True,
                )

            if self.termination_checker is not None:
                fx_check = fx_estimate if fx_next is None else fx_next
                if self.termination_checker(
                    self.nfev,
                    x,
                    float(fx_check),
                    0.0 if self.last_stepsize is None else self.last_stepsize,
                    True,
                ):
                    break

        result = OptimizerResult()
        result.x = x
        result.fun = float(fun(x))
        self._nfev += 1
        result.nfev = self.nfev
        result.nit = k

        return result
