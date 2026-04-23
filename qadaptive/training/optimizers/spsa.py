from __future__ import annotations

from collections import deque
from collections.abc import Callable, Iterator
from numbers import Real
from time import time
from typing import Any
import itertools
import logging
import warnings

import numpy as np
import scipy

from qiskit_algorithms.optimizers.optimizer import (
    OptimizerResult,
    OptimizerSupportLevel,
    POINT,
)
from qiskit_algorithms.utils import algorithm_globals

from .stepwise_optimizer import StepwiseOptimizer, CALLBACK, TERMINATIONCHECKER

logger = logging.getLogger(__name__)

class SPSA(StepwiseOptimizer):
    """
    Simultaneous Perturbation Stochastic Approximation (SPSA) optimizer.

    This implementation supports both first-order SPSA and second-order SPSA
    (2-SPSA), and exposes a stepwise optimizer interface via ``initialize``
    and ``step``.
    """

    def __init__(
        self,
        maxiter: int = 100,
        blocking: bool = False,
        allowed_increase: float | None = None,
        trust_region: bool = False,
        learning_rate: float | np.ndarray | Callable[..., Iterator[float]] | None = None,
        perturbation: float | np.ndarray | Callable[..., Iterator[float]] | None = None,
        start_point: int = 0,
        last_avg: int = 1,
        resamplings: int | dict[int, int] = 1,
        perturbation_dims: int | None = None,
        second_order: bool = False,
        regularization: float | None = None,
        hessian_delay: int = 0,
        lse_solver: Callable[[np.ndarray, np.ndarray], np.ndarray] | None = None,
        initial_hessian: np.ndarray | None = None,
        callback: CALLBACK | None = None,
        termination_checker: TERMINATIONCHECKER | None = None,
    ) -> None:
        """
        Parameters
        ----------
        maxiter : int, optional
            Maximum number of iterations.
        blocking : bool, optional
            If True, only accept updates that improve the loss up to
            ``allowed_increase``.
        allowed_increase : float | None, optional
            Maximum allowed increase in loss when ``blocking=True``.
        trust_region : bool, optional
            If True, constrain the norm of the update direction to at most 1.
        learning_rate : float | np.ndarray | Callable[..., Iterator[float]] | None, optional
            Learning-rate schedule. If None together with ``perturbation=None``,
            SPSA performs internal calibration.
        perturbation : float | np.ndarray | Callable[..., Iterator[float]] | None, optional
            Perturbation schedule.
        start_point : int, optional
            Default schedule offset for a fresh initialization.
        last_avg : int, optional
            Number of final iterates to average when using ``minimize``.
        resamplings : int | dict[int, int], optional
            Number of SPSA resamplings per iteration. If a dictionary, it is
            interpreted as ``{iteration: num_resamplings}``.
        perturbation_dims : int | None, optional
            Number of perturbed dimensions. If None, all dimensions are perturbed.
        second_order : bool, optional
            If True, use 2-SPSA.
        regularization : float | None, optional
            Regularization for the 2-SPSA Hessian preconditioner.
        hessian_delay : int, optional
            Number of initial iterations during which the Hessian is estimated
            but not yet used as a preconditioner.
        lse_solver : Callable[[np.ndarray, np.ndarray], np.ndarray] | None, optional
            Linear solver used to apply the inverse Hessian in 2-SPSA.
        initial_hessian : np.ndarray | None, optional
            Initial Hessian estimate for 2-SPSA.
        callback : CALLBACK | None, optional
            Optional callback.
        termination_checker : TERMINATIONCHECKER | None, optional
            Optional termination checker.
        """
        super().__init__(callback=callback, termination_checker=termination_checker)

        self.maxiter = maxiter
        self.blocking = blocking
        self.allowed_increase = allowed_increase
        self.trust_region = trust_region

        self.learning_rate = learning_rate
        self.perturbation = perturbation
        self.start_point = start_point

        self.last_avg = last_avg
        self.resamplings = resamplings
        self.perturbation_dims = perturbation_dims

        self.second_order = second_order
        self.regularization = 0.01 if regularization is None else regularization
        self.hessian_delay = hessian_delay
        self.lse_solver = lse_solver
        self.initial_hessian = initial_hessian

        # iterator state
        self.lr_iterator: Iterator[float] | None = None
        self._lr_iterator_copy: Iterator[float] | None = None
        self.p_iterator: Iterator[float] | None = None

        # 2-SPSA state
        self._smoothed_hessian: np.ndarray | None = None
        self._active_lse_solver: Callable[[np.ndarray, np.ndarray], np.ndarray] | None = None
        self._steps_in_run: int = 0

        # bookkeeping
        self._hyperparameters: dict[str, float | None] = {
            "a": None,
            "alpha": None,
            "stability_constant": None,
            "c": None,
            "gamma": None,
        }

    @property
    def settings(self) -> dict[str, Any]:
        """
        Return optimizer settings.

        Returns
        -------
        dict[str, Any]
            Dictionary of optimizer configuration values.
        """
        if callable(self.learning_rate):
            iterator = self.learning_rate(n_start=0)
            learning_rate = np.array([next(iterator) for _ in range(self.maxiter)])
        else:
            learning_rate = self.learning_rate

        if callable(self.perturbation):
            iterator = self.perturbation(n_start=0)
            perturbation = np.array([next(iterator) for _ in range(self.maxiter)])
        else:
            perturbation = self.perturbation

        return {
            "maxiter": self.maxiter,
            "blocking": self.blocking,
            "allowed_increase": self.allowed_increase,
            "trust_region": self.trust_region,
            "learning_rate": learning_rate,
            "perturbation": perturbation,
            "start_point": self.start_point,
            "last_avg": self.last_avg,
            "resamplings": self.resamplings,
            "perturbation_dims": self.perturbation_dims,
            "second_order": self.second_order,
            "regularization": self.regularization,
            "hessian_delay": self.hessian_delay,
            "lse_solver": self.lse_solver,
            "initial_hessian": self.initial_hessian,
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
            "gradient": OptimizerSupportLevel.ignored,
            "bounds": OptimizerSupportLevel.ignored,
            "initial_point": OptimizerSupportLevel.required,
        }

    def set_learning_rate(
        self,
        value: float | np.ndarray | Callable[..., Iterator[float]],
    ) -> None:
        """
        Set the learning-rate schedule.

        Parameters
        ----------
        value : float | np.ndarray | Callable[..., Iterator[float]]
            Learning-rate specification.
        """
        self.learning_rate = value

    def set_perturbation(
        self,
        value: float | np.ndarray | Callable[..., Iterator[float]],
    ) -> None:
        """
        Set the perturbation schedule.

        Parameters
        ----------
        value : float | np.ndarray | Callable[..., Iterator[float]]
            Perturbation specification.
        """
        self.perturbation = value

    def set_init_hessian(self, value: np.ndarray | None) -> None:
        """
        Set the initial Hessian estimate for 2-SPSA.

        Parameters
        ----------
        value : np.ndarray | None
            Initial Hessian estimate.
        """
        self.initial_hessian = value

    def set_allowed_increase(self, value: float | None) -> None:
        """
        Set the allowed increase used by blocking mode.

        Parameters
        ----------
        value : float | None
            Allowed increase in objective value.
        """
        self.allowed_increase = value

    def get_hyperparameters(self) -> dict[str, float | None]:
        """
        Return the SPSA power-series hyperparameters.

        Returns
        -------
        dict[str, float | None]
            Dictionary with keys ``a``, ``alpha``, ``stability_constant``,
            ``c``, and ``gamma``.
        """
        return dict(self._hyperparameters)

    def set_power_series_hyperparameters(
        self,
        a: float,
        alpha: float,
        c: float,
        gamma: float,
        stability_constant: float = 0.0,
    ) -> None:
        """
        Set SPSA learning-rate and perturbation schedules as power series.

        Parameters
        ----------
        a : float
            Initial learning-rate scaling factor.
        alpha : float
            Learning-rate decay exponent.
        c : float
            Initial perturbation scaling factor.
        gamma : float
            Perturbation decay exponent.
        stability_constant : float, optional
            Stability constant in the learning-rate schedule.
        """

        def learning_rate(n_start: int = 0) -> Iterator[float]:
            return powerseries(a, alpha, offset=stability_constant, n_start=n_start)

        def perturbation(n_start: int = 0) -> Iterator[float]:
            return powerseries(c, gamma, n_start=n_start)

        self.set_learning_rate(learning_rate)
        self.set_perturbation(perturbation)
        self._hyperparameters = {
            "a": a,
            "alpha": alpha,
            "stability_constant": stability_constant,
            "c": c,
            "gamma": gamma,
        }

    @staticmethod
    def calibrate(
        loss: Callable[[np.ndarray], float],
        initial_point: np.ndarray,
        c: float = 0.2,
        stability_constant: float = 0.0,
        target_magnitude: float | None = None,
        alpha: float = 0.602,
        gamma: float = 0.101,
        modelspace: bool = False,
        max_evals_grouped: int = 1,
        **kwargs,
    ) -> tuple[
        Callable[[int], Iterator[float]],
        Callable[[int], Iterator[float]],
        dict[str, float],
    ]:
        r"""
        Calibrate SPSA learning-rate and perturbation power series.

        The power series are

        .. math::

            a_k = a / (A + k + 1)^\alpha, \quad c_k = c / (k + 1)^\gamma

        Parameters
        ----------
        loss : Callable[[np.ndarray], float]
            Loss function.
        initial_point : np.ndarray
            Initial point.
        c : float, optional
            Initial perturbation magnitude.
        stability_constant : float, optional
            Stability constant for the learning-rate sequence.
        target_magnitude : float | None, optional
            Target magnitude of the initial update. If None, uses ``2*pi/10``.
        alpha : float, optional
            Learning-rate decay exponent.
        gamma : float, optional
            Perturbation decay exponent.
        modelspace : bool, optional
            Whether to calibrate in model space.
        max_evals_grouped : int, optional
            Maximum grouped evaluations supported by the loss.
        **kwargs
            Additional keyword arguments forwarded to the loss function.

        Returns
        -------
        tuple[Callable[[int], Iterator[float]], Callable[[int], Iterator[float]], dict[str, float]]
            Learning-rate factory, perturbation factory, and hyperparameter dictionary.
        """
        logger.info("SPSA: starting calibration of learning rate and perturbation.")

        if target_magnitude is None:
            target_magnitude = 2 * np.pi / 10

        dim = len(initial_point)
        steps = 25
        points: list[np.ndarray] = []

        for _ in range(steps):
            pert = bernoulli_perturbation(dim)
            points.extend([initial_point + c * pert, initial_point - c * pert])

        losses = _batch_evaluate(loss, points, max_evals_grouped, **kwargs)

        avg_magnitudes = 0.0
        for i in range(steps):
            delta = losses[2 * i] - losses[2 * i + 1]
            avg_magnitudes += abs(delta / (2 * c))

        avg_magnitudes /= steps

        if modelspace:
            a = target_magnitude / (avg_magnitudes**2)
        else:
            a = target_magnitude / avg_magnitudes

        if a < 1e-10:
            warnings.warn(f"Calibration failed, using {target_magnitude} for `a`.")
            a = target_magnitude

        def learning_rate(n_start: int = 0) -> Iterator[float]:
            return powerseries(a, alpha, offset=stability_constant, n_start=n_start)

        def perturbation(n_start: int = 0) -> Iterator[float]:
            return powerseries(c, gamma, n_start=n_start)

        hyperparams = {
            "a": a,
            "alpha": alpha,
            "stability_constant": stability_constant,
            "c": c,
            "gamma": gamma,
        }

        logger.info(
            "SPSA calibration finished: a=%s, alpha=%s, A=%s, c=%s, gamma=%s",
            a,
            alpha,
            stability_constant,
            c,
            gamma,
        )

        return learning_rate, perturbation, hyperparams

    @staticmethod
    def estimate_stddev(
        loss: Callable[[np.ndarray], float],
        initial_point: np.ndarray,
        avg: int = 25,
        max_evals_grouped: int = 1,
        **kwargs,
    ) -> float:
        """
        Estimate the standard deviation of the loss function at a point.

        Parameters
        ----------
        loss : Callable[[np.ndarray], float]
            Loss function.
        initial_point : np.ndarray
            Point at which to estimate the standard deviation.
        avg : int, optional
            Number of repeated evaluations.
        max_evals_grouped : int, optional
            Maximum grouped evaluations supported by the loss.
        **kwargs
            Additional keyword arguments forwarded to the loss function.

        Returns
        -------
        float
            Estimated standard deviation.
        """
        losses = _batch_evaluate(loss, avg * [initial_point], max_evals_grouped, **kwargs)
        return float(np.std(losses))

    def _create_iterators(
        self,
        fun: Callable[[np.ndarray], float] | None = None,
        x0: np.ndarray | list[float] | None = None,
        n_start: int = 0,
        **kwargs,
    ) -> None:
        """
        Create learning-rate and perturbation iterators aligned to a given start index.

        Parameters
        ----------
        fun : Callable[[np.ndarray], float] | None, optional
            Loss function, required if calibration is needed.
        x0 : np.ndarray | list[float] | None, optional
            Initial point, required if calibration is needed.
        n_start : int, optional
            Starting iteration index for both schedules.
        **kwargs
            Additional keyword arguments forwarded to calibration.
        """
        if self.learning_rate is None and self.perturbation is None:
            if fun is None or x0 is None:
                raise ValueError(
                    "Loss function and initial point are required for SPSA calibration."
                )

            max_grouped = getattr(self, "_max_evals_grouped", 1)
            get_eta, get_eps, hyperparams = self.calibrate(
                fun,
                np.asarray(x0, dtype=float),
                max_evals_grouped=max_grouped,
                **kwargs,
            )
            self.learning_rate = get_eta
            self.perturbation = get_eps
            self._hyperparameters = hyperparams
        else:
            get_eta, get_eps = _validate_pert_and_learningrate(
                self.perturbation,
                self.learning_rate,
            )

        logger.info(
            "Creating SPSA iterators starting at index %d.", n_start
        )
        eta = get_eta(n_start=n_start)
        eps = get_eps(n_start=n_start)
        self.lr_iterator, self._lr_iterator_copy = itertools.tee(eta)
        self.p_iterator = eps

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
            Objective function.
        iteration_start : int | None, optional
            Schedule offset for the new run. If None, ``self.start_point`` is used.
        **kwargs
            Additional keyword arguments forwarded to calibration and objective evaluation.
        """
        logger.info("Initializing SPSA optimizer.")
        x0 = np.asarray(x0, dtype=float)

        if iteration_start is None:
            iteration_start = self._iteration if self._initialized else self.start_point


        self.reset_runtime_state(iteration_start=iteration_start)
        self._steps_in_run = 0

        self._create_iterators(
            fun=loss_function,
            x0=x0,
            n_start=iteration_start,
            **kwargs,
        )

        self._active_lse_solver = np.linalg.solve if self.lse_solver is None else self.lse_solver

        if self.second_order:
            if self.initial_hessian is None:
                self._smoothed_hessian = np.eye(x0.size)
            else:
                self._smoothed_hessian = np.asarray(self.initial_hessian, dtype=float)
        else:
            self._smoothed_hessian = None

        if self.blocking:
            self._last_fx = float(loss_function(x0, **kwargs))
            self._nfev += 1

            if self.allowed_increase is None:
                max_grouped = getattr(self, "_max_evals_grouped", 1)
                self.allowed_increase = 2 * self.estimate_stddev(
                    loss_function,
                    x0,
                    max_evals_grouped=max_grouped,
                    **kwargs,
                )
        else:
            self._last_fx = None

        self._initialized = True

    def _point_sample(
        self,
        loss: Callable[[np.ndarray], float],
        x: np.ndarray,
        eps: float,
        delta1: np.ndarray,
        delta2: np.ndarray | None = None,
        **kwargs,
    ) -> tuple[float, np.ndarray, np.ndarray | None]:
        """
        Compute a single SPSA sample of the function value, gradient, and optional Hessian.

        Parameters
        ----------
        loss : Callable[[np.ndarray], float]
            Loss function.
        x : np.ndarray
            Current parameter vector.
        eps : float
            Perturbation magnitude.
        delta1 : np.ndarray
            First perturbation direction.
        delta2 : np.ndarray | None, optional
            Second perturbation direction used in 2-SPSA.

        Returns
        -------
        tuple[float, np.ndarray, np.ndarray | None]
            Mean value estimate, gradient sample, and optional Hessian sample.
        """
        points = [x + eps * delta1, x - eps * delta1]
        self._nfev += 2
        
        logger.info("Perturbation with strength %s in directions %s.", eps, delta1)
        logger.info("Evaluating with perturbed parameters: %s in this resampling.", points)

        if self.second_order:
            if delta2 is None:
                raise ValueError("delta2 must be provided when second_order=True.")
            points += [x + eps * (delta1 + delta2), x + eps * (-delta1 + delta2)]
            self._nfev += 2
            logger.info(
                "Evaluating with additional parameters: %s due to second order being set.",
                points[-2:]
                )


        max_grouped = getattr(self, "_max_evals_grouped", 1)
        values = _batch_evaluate(loss, points, max_grouped, **kwargs)

        plus = values[0]
        minus = values[1]
        gradient_sample = (plus - minus) / (2 * eps) * delta1

        hessian_sample = None
        if self.second_order:
            diff = (values[2] - plus) - (values[3] - minus)
            diff /= 2 * eps**2
            rank_one = np.outer(delta1, delta2)
            hessian_sample = diff * (rank_one + rank_one.T) / 2

        return float(np.mean(values)), gradient_sample, hessian_sample

    def _point_estimate(
        self,
        loss: Callable[[np.ndarray], float],
        x: np.ndarray,
        eps: float,
        num_samples: int,
        **kwargs,
    ) -> tuple[float, np.ndarray, np.ndarray]:
        """
        Estimate the function value, gradient, and Hessian at a point.

        Parameters
        ----------
        loss : Callable[[np.ndarray], float]
            Loss function.
        x : np.ndarray
            Current parameter vector.
        eps : float
            Perturbation magnitude.
        num_samples : int
            Number of resamplings.
        **kwargs
            Additional keyword arguments forwarded to the loss function.

        Returns
        -------
        tuple[float, np.ndarray, np.ndarray]
            Value estimate, gradient estimate, and Hessian estimate.
        """
        value_estimate = 0.0
        gradient_estimate = np.zeros(x.size)
        hessian_estimate = np.zeros((x.size, x.size))

        deltas1 = [
            bernoulli_perturbation(x.size, self.perturbation_dims)
            for _ in range(num_samples)
        ]

        deltas2 = (
            [
                bernoulli_perturbation(x.size, self.perturbation_dims)
                for _ in range(num_samples)
            ]
            if self.second_order
            else [None] * num_samples
        )

        for i in range(num_samples):
            value_sample, gradient_sample, hessian_sample = self._point_sample(
                loss,
                x,
                eps,
                deltas1[i],
                deltas2[i],
                **kwargs,
            )
            value_estimate += value_sample
            gradient_estimate += gradient_sample

            if self.second_order and hessian_sample is not None:
                hessian_estimate += hessian_sample

        return (
            value_estimate / num_samples,
            gradient_estimate / num_samples,
            hessian_estimate / num_samples,
        )

    def compute_loss_and_gradient_estimate(
        self,
        loss: Callable[[np.ndarray], float],
        x: np.ndarray,
        iteration: int | None = None,
        lse_solver: Callable[[np.ndarray, np.ndarray], np.ndarray] | None = None,
        **kwargs,
    ) -> tuple[float, np.ndarray]:
        """
        Compute the SPSA estimate of the function value and gradient.

        Parameters
        ----------
        loss : Callable[[np.ndarray], float]
            Loss function.
        x : np.ndarray
            Current parameter vector.
        iteration : int | None, optional
            Absolute iteration index for schedule-dependent behavior. If None,
            ``self.iteration + 1`` is used.
        lse_solver : Callable[[np.ndarray, np.ndarray], np.ndarray] | None, optional
            Linear solver to use for the 2-SPSA preconditioner. If None, uses
            the active solver from initialization.
        **kwargs
            Additional keyword arguments forwarded to the loss function.

        Returns
        -------
        tuple[float, np.ndarray]
            Function value estimate and gradient estimate.
        """
        if iteration is None:
            iteration = self.iteration + 1

        if isinstance(self.resamplings, dict):
            num_samples = self.resamplings.get(iteration, 1)
        else:
            num_samples = self.resamplings

        if self.p_iterator is None or self.lr_iterator is None:
            self._create_iterators(fun=loss, x0=x, n_start=max(iteration - 1, 0), **kwargs)

        eps = next(self.p_iterator)
        fx_estimate, gradient, hessian = self._point_estimate(
            loss,
            np.asarray(x, dtype=float),
            eps,
            num_samples,
            **kwargs,
        )

        active_solver = self._active_lse_solver if lse_solver is None else lse_solver

        if self.second_order:
            local_step = self._steps_in_run + 1
            if self._smoothed_hessian is None:
                self._smoothed_hessian = np.asarray(hessian, dtype=float)
            else:
                self._smoothed_hessian = (
                    (local_step - 1) / local_step * self._smoothed_hessian
                    + 1 / local_step * hessian
                )

            if local_step > self.hessian_delay:
                spd_hessian = _make_spd(self._smoothed_hessian, self.regularization)
                gradient = np.real(active_solver(spd_hessian, gradient))

        return fx_estimate, gradient

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
        Process a proposed SPSA update.

        Parameters
        ----------
        gradient_estimate : np.ndarray
            Gradient estimate.
        x : np.ndarray
            Current parameter vector.
        fx : float
            Current function value estimate.
        fun : Callable[[np.ndarray], float]
            Objective function.
        fun_next : Callable[[np.ndarray], float] | None
            Optional objective used to evaluate the proposed next point.
        iteration_start : float, optional
            Timestamp used for logging.
        iteration : int, optional
            Current iteration number.
        **kwargs
            Additional keyword arguments forwarded to the objective.

        Returns
        -------
        tuple[bool, np.ndarray, float | None]
            Whether the step should be skipped, the proposed next point,
            and the objective value at that point if evaluated.
        """
        grad = np.asarray(gradient_estimate, dtype=float)

        if self.trust_region:
            norm = np.linalg.norm(grad)
            if norm > 1:
                grad = grad / norm
                
        logger.info("Gradient was estimated as %s", grad)

        if self.lr_iterator is None:
            self._create_iterators(fun=fun, x0=x, n_start=max(iteration - 1, 0), **kwargs)

        learn_rate = next(self.lr_iterator)
        logger.debug("Learning rate for this iteration is %s", learn_rate)
        
        update = grad * learn_rate
        x_next = np.asarray(x, dtype=float) - update
        fx_next = None
        
        logger.debug("Proposed next point is: %s", x_next)

        if self.blocking:
            if fun_next is None:
                logger.info("Calculating cost function at next point for blocking check.")
                self._nfev += 1
                fx_next = float(fun(x_next, **kwargs))
            else:
                logger.info(
                    "Calculating cost function at next point with custom function for blocking check."
                    )
                self._nextfev += 1
                fx_next = float(fun_next(x_next, **kwargs))

            if fx + self.allowed_increase <= fx_next:
                if self.callback is not None:
                    self.callback(
                        self._nfev,
                        x_next,
                        fx_next,
                        float(np.linalg.norm(update)),
                        False,
                    )

                logger.info(
                    "Iteration %s/%s rejected in %s.",
                    iteration,
                    self.maxiter,
                    time() - iteration_start if iteration_start else 0.0,
                )
                return True, np.asarray(x, dtype=float), fx

        return False, x_next, fx_next

    def step(
        self,
        x: np.ndarray,
        loss_function: Callable[[np.ndarray], float],
        loss_next: Callable[[np.ndarray], float] | None = None,
        **kwargs,
    ) -> tuple[bool, np.ndarray, float | None, np.ndarray | None, float | None]:
        """
        Perform one SPSA optimization step.

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
            self.initialize(np.asarray(x, dtype=float), loss_function, iteration_start=self.iteration, **kwargs)

        x = np.asarray(x, dtype=float)
        next_iteration = self.iteration + 1
        iter_start = time()

        fx_estimate, gradient_estimate = self.compute_loss_and_gradient_estimate(
            loss_function,
            x,
            iteration=next_iteration,
            lse_solver=self._active_lse_solver,
            **kwargs,
        )

        skip, x_next, fx_next = self.process_update(
            gradient_estimate,
            x,
            fx_estimate,
            loss_function,
            loss_next,
            iteration_start=iter_start,
            iteration=next_iteration,
            **kwargs,
        )

        self._iteration = next_iteration
        self._steps_in_run += 1
        self._last_gradient = np.asarray(gradient_estimate, dtype=float)
        self._last_fx = float(fx_estimate)
        self._last_stepsize = None if skip else float(np.linalg.norm(x_next - x))

        return skip, x_next, fx_next, gradient_estimate, fx_estimate

    def minimize(
        self,
        fun: Callable[[POINT], float],
        x0: POINT,
        fun_next: Callable[[POINT], float] | None = None,
        jac: Callable[[POINT], POINT] | None = None,
        bounds: list[tuple[float, float]] | None = None,
        **kwargs,
    ) -> OptimizerResult:
        """
        Minimize the scalar function.

        Parameters
        ----------
        fun : Callable[[POINT], float]
            Objective function.
        x0 : POINT
            Initial point.
        fun_next : Callable[[POINT], float] | None, optional
            Optional objective for evaluating the proposed next point.
        jac : Callable[[POINT], POINT] | None, optional
            Ignored. Present for Qiskit optimizer compatibility.
        bounds : list[tuple[float, float]] | None, optional
            Ignored.
        **kwargs
            Additional keyword arguments forwarded to the objective.

        Returns
        -------
        OptimizerResult
            Optimization result.
        """
        del jac, bounds

        x = np.asarray(x0, dtype=float)
        self.initialize(x, fun, self.iteration, **kwargs)
        start_time = time()

        last_steps = deque([x])

        while self._steps_in_run < self.maxiter:
            skip, x_next, fx_next, gradient_estimate, fx_estimate = self.step(
                x,
                fun,
                loss_next=fun_next,
                **kwargs,
            )

            if skip:
                continue

            x = x_next

            if self.callback is not None:
                fx_cb: float
                if fx_next is None:
                    if fun_next is None:
                        self._nfev += 1
                        fx_cb = float(fun(x, **kwargs))
                    else:
                        self._nextfev += 1
                        fx_cb = float(fun_next(x, **kwargs))
                else:
                    fx_cb = float(fx_next)

                self.callback(
                    self._nfev,
                    x,
                    fx_cb,
                    0.0 if self.last_stepsize is None else self.last_stepsize,
                    True,
                )

            if self.last_avg > 1:
                last_steps.append(x)
                if len(last_steps) > self.last_avg:
                    last_steps.popleft()

            if self.termination_checker is not None:
                fx_check = fx_estimate if fx_next is None else fx_next
                if self.termination_checker(
                    self._nfev,
                    x,
                    float(fx_check),
                    0.0 if self.last_stepsize is None else self.last_stepsize,
                    True,
                ):
                    logger.info(
                        "SPSA terminated early at local iteration %s/%s.",
                        self._steps_in_run,
                        self.maxiter,
                    )
                    break
                
            logger.debug(
                "Iteration %s/%s done in %s.", self._steps_in_run, self.maxiter + 1, time() - start_time
                )

        if self.last_avg > 1:
            x = np.mean(last_steps, axis=0)

        result = OptimizerResult()
        result.x = x

        if fun_next is None:
            self._nfev += 1
            result.fun = float(fun(x, **kwargs))
        else:
            self._nextfev += 1
            result.fun = float(fun_next(x, **kwargs))

        result.nfev = self._nfev
        result.nit = self._steps_in_run

        logger.info("SPSA finished in %s seconds.", time() - start_time)
        return result


def bernoulli_perturbation(
    dim: int,
    perturbation_dims: int | None = None,
) -> np.ndarray:
    """
    Generate a Bernoulli perturbation vector.

    Parameters
    ----------
    dim : int
        Dimension of the parameter vector.
    perturbation_dims : int | None, optional
        Number of perturbed dimensions. If None, all dimensions are perturbed.

    Returns
    -------
    np.ndarray
        Perturbation vector with entries in ``{-1, 0, +1}``.
    """
    if perturbation_dims is None:
        return 1 - 2 * algorithm_globals.random.binomial(1, 0.5, size=dim)

    pert = 1 - 2 * algorithm_globals.random.binomial(1, 0.5, size=perturbation_dims)
    indices = algorithm_globals.random.choice(
        list(range(dim)),
        size=perturbation_dims,
        replace=False,
    )
    result = np.zeros(dim)
    result[indices] = pert
    return result


def powerseries(
    eta: float = 0.01,
    power: float = 2.0,
    offset: float = 0.0,
    n_start: int = 0,
) -> Iterator[float]:
    """
    Yield a power-law-decaying sequence.

    Parameters
    ----------
    eta : float, optional
        Initial scale factor.
    power : float, optional
        Decay exponent.
    offset : float, optional
        Offset added to the iteration count.
    n_start : int, optional
        Starting iteration index.

    Yields
    ------
    float
        Next value in the sequence.
    """
    n = n_start + 1
    while True:
        yield eta / ((n + offset) ** power)
        n += 1


def constant(eta: float = 0.01) -> Iterator[float]:
    """
    Yield a constant sequence.

    Parameters
    ----------
    eta : float, optional
        Constant value.

    Yields
    ------
    float
        Constant value.
    """
    while True:
        yield eta


def _batch_evaluate(
    function: Callable,
    points: list[Any] | np.ndarray,
    max_evals_grouped: int,
    unpack_points: bool = False,
    **kwargs,
) -> list[Any]:
    """
    Evaluate a function on all points with optional grouped execution.

    Parameters
    ----------
    function : Callable
        Function to evaluate.
    points : list[Any] | np.ndarray
        Points at which to evaluate the function.
    max_evals_grouped : int
        Maximum number of grouped evaluations.
    unpack_points : bool, optional
        Whether each point should be unpacked as a tuple.
    **kwargs
        Additional keyword arguments forwarded to the function.

    Returns
    -------
    list[Any]
        List of function values.
    """
    if max_evals_grouped is None or max_evals_grouped == 1:
        results = []
        for point in points:
            if unpack_points:
                results.append(function(*point, **kwargs))
            else:
                results.append(function(point, **kwargs))
        return results

    num_points = len(points)
    num_batches = num_points // max_evals_grouped
    if num_points % max_evals_grouped != 0:
        num_batches += 1

    batched_points = np.array_split(np.asarray(points, dtype=object), num_batches)

    results: list[Any] = []
    for batch in batched_points:
        if unpack_points:
            repacked = _repack_points(batch)
            results.extend(_as_list(function(*repacked, **kwargs)))
        else:
            results.extend(_as_list(function(batch, **kwargs)))

    return results


def _as_list(obj: Any) -> list[Any]:
    """
    Convert an array-like output to a Python list.

    Parameters
    ----------
    obj : Any
        Object to convert.

    Returns
    -------
    list[Any]
        Output converted to a list if needed.
    """
    return obj.tolist() if isinstance(obj, np.ndarray) else list(obj)


def _repack_points(points: list[tuple[Any, ...]] | np.ndarray):
    """
    Repack a list of tuples of points into a tuple of lists.

    Parameters
    ----------
    points : list[tuple[Any, ...]] | np.ndarray
        Input points.

    Returns
    -------
    generator
        Generator yielding grouped components.
    """
    num_sets = len(points[0])
    return ([x[i] for x in points] for i in range(num_sets))


def _make_spd(matrix: np.ndarray, bias: float = 0.01) -> np.ndarray:
    """
    Construct a symmetric positive-definite matrix from a square matrix.

    Parameters
    ----------
    matrix : np.ndarray
        Input matrix.
    bias : float, optional
        Positive diagonal bias.

    Returns
    -------
    np.ndarray
        Symmetric positive-definite matrix.
    """
    identity = np.eye(matrix.shape[0])
    psd = scipy.linalg.sqrtm(matrix.dot(matrix))
    return np.real(psd) + bias * identity


def _validate_pert_and_learningrate(
    perturbation: float | np.ndarray | Callable[..., Iterator[float]] | None,
    learning_rate: float | np.ndarray | Callable[..., Iterator[float]] | None,
) -> tuple[
    Callable[[int], Iterator[float]],
    Callable[[int], Iterator[float]],
]:
    """
    Validate and normalize perturbation and learning-rate specifications.

    Parameters
    ----------
    perturbation : float | np.ndarray | Callable[..., Iterator[float]] | None
        Perturbation specification.
    learning_rate : float | np.ndarray | Callable[..., Iterator[float]] | None
        Learning-rate specification.

    Returns
    -------
    tuple[Callable[[int], Iterator[float]], Callable[[int], Iterator[float]]]
        Generator factories for learning rate and perturbation.

    Raises
    ------
    ValueError
        If exactly one of the two schedules is provided.
    """
    if learning_rate is None or perturbation is None:
        raise ValueError(
            "If one of learning rate or perturbation is set, both must be set."
        )

    if isinstance(perturbation, Real):

        def get_eps(n_start: int = 0) -> Iterator[float]:
            del n_start
            return constant(float(perturbation))

    elif isinstance(perturbation, (list, np.ndarray)):

        def get_eps(n_start: int = 0) -> Iterator[float]:
            return itertools.islice(iter(perturbation), n_start, None)

    else:
        get_eps = perturbation

    if isinstance(learning_rate, Real):

        def get_eta(n_start: int = 0) -> Iterator[float]:
            del n_start
            return constant(float(learning_rate))

    elif isinstance(learning_rate, (list, np.ndarray)):

        def get_eta(n_start: int = 0) -> Iterator[float]:
            return itertools.islice(iter(learning_rate), n_start, None)

    else:
        get_eta = learning_rate

    return get_eta, get_eps
