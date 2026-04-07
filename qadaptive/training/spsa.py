# This code is part of a Qiskit project.
#
# (C) Copyright IBM 2018, 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

# This File has been modified from it's original version.

"""Simultaneous Perturbation Stochastic Approximation (SPSA) optimizer.

This implementation allows both, standard first-order as well as second-order SPSA.
"""
from __future__ import annotations

from collections import deque
from collections.abc import Iterator
from typing import Callable, Any, SupportsFloat
import logging
import warnings
from time import time
import random
import itertools

import scipy
import numpy as np

from qiskit_algorithms.utils import algorithm_globals

from qiskit_algorithms.optimizers.optimizer import Optimizer, OptimizerSupportLevel, OptimizerResult, POINT

# number of function evaluations, parameters, loss, stepsize, accepted
CALLBACK = Callable[[int, np.ndarray, float, SupportsFloat, bool], None]
TERMINATIONCHECKER = Callable[[int, np.ndarray, float, SupportsFloat, bool], bool]

logger = logging.getLogger(__name__)

class SPSA(Optimizer):
    """Simultaneous Perturbation Stochastic Approximation (SPSA) optimizer.

    SPSA [1] is an gradient descent method for optimizing systems with multiple unknown parameters.
    As an optimization method, it is appropriately suited to large-scale population models,
    adaptive modeling, and simulation optimization.

    .. seealso::

        Many examples are presented at the `SPSA Web site <http://www.jhuapl.edu/SPSA>`__.

    The main feature of SPSA is the stochastic gradient approximation, which requires only two
    measurements of the objective function, regardless of the dimension of the optimization
    problem.

    Additionally to standard, first-order SPSA, where only gradient information is used, this
    implementation also allows second-order SPSA (2-SPSA) [2]. In 2-SPSA we also estimate the
    Hessian of the loss with a stochastic approximation and multiply the gradient with the
    inverse Hessian to take local curvature into account and improve convergence.
    Notably this Hessian estimate requires only a constant number of function evaluations
    unlike an exact evaluation of the Hessian, which scales quadratically in the number of
    function evaluations.

    .. note::

        SPSA can be used in the presence of noise, and it is therefore indicated in situations
        involving measurement uncertainty on a quantum computation when finding a minimum.
        If you are executing a variational algorithm using a Quantum ASseMbly Language (QASM)
        simulator or a real device, SPSA would be the most recommended choice among the optimizers
        provided here.

    The optimization process can includes a calibration phase if neither the ``learning_rate`` nor
    ``perturbation`` is provided, which requires additional functional evaluations.
    (Note that either both or none must be set.) For further details on the automatic calibration,
    please refer to the supplementary information section IV. of [3].

    .. note::

        This component has some function that is normally random. If you want to reproduce behavior
        then you should set the random number generator seed in the algorithm_globals
        (``qiskit_algorithms.utils.algorithm_globals.random_seed = seed``).


    Examples:

        This short example runs SPSA for the ground state calculation of the ``Z ^ Z``
        observable where the ansatz is a ``PauliTwoDesign`` circuit.

        .. code-block:: python

            import numpy as np
            from qiskit_algorithms.optimizers import SPSA
            from qiskit.circuit.library import PauliTwoDesign
            from qiskit.quantum_info import Pauli

            ansatz = PauliTwoDesign(2, reps=1, seed=2)
            observable = Pauli("Z") ^ Pauli("Z")
            initial_point = np.random.random(ansatz.num_parameters)

            def loss(x):
                bound = ansatz.bind_parameters(x)
                return np.real((StateFn(observable, is_measurement=True) @ StateFn(bound)).eval())

            spsa = SPSA(maxiter=300)
            result = spsa.optimize(ansatz.num_parameters, loss, initial_point=initial_point)

        To use the Hessian information, i.e. 2-SPSA, you can add `second_order=True` to the
        initializer of the `SPSA` class, the rest of the code remains the same.

        .. code-block:: python

            two_spsa = SPSA(maxiter=300, second_order=True)
            result = two_spsa.optimize(ansatz.num_parameters, loss, initial_point=initial_point)

        The `termination_checker` can be used to implement a custom termination criterion.

        .. code-block:: python

            import numpy as np
            from qiskit_algorithms.optimizers import SPSA

            def objective(x):
                return np.linalg.norm(x) + .04*np.random.rand(1)

            class TerminationChecker:

                def __init__(self, N : int):
                    self.N = N
                    self.values = []

                def __call__(self, nfev, parameters, value, stepsize, accepted) -> bool:
                    self.values.append(value)

                    if len(self.values) > self.N:
                        last_values = self.values[-self.N:]
                        pp = np.polyfit(range(self.N), last_values, 1)
                        slope = pp[0] / self.N

                        if slope > 0:
                            return True
                    return False

            spsa = SPSA(maxiter=200, termination_checker=TerminationChecker(10))
            parameters, value, niter = spsa.optimize(2, objective, initial_point=[0.5, 0.5])
            print(f'SPSA completed after {niter} iterations')


    References:

        [1]: J. C. Spall (1998). An Overview of the Simultaneous Perturbation Method for Efficient
        Optimization, Johns Hopkins APL Technical Digest, 19(4), 482–492.
        `Online at jhuapl.edu. <https://www.jhuapl.edu/SPSA/PDF-SPSA/Spall_An_Overview.PDF>`_

        [2]: J. C. Spall (1997). Accelerated second-order stochastic optimization using only
        function measurements, Proceedings of the 36th IEEE Conference on Decision and Control,
        1417-1424 vol.2. `Online at IEEE.org. <https://ieeexplore.ieee.org/document/657661>`_

        [3]: A. Kandala et al. (2017). Hardware-efficient Variational Quantum Eigensolver for
        Small Molecules and Quantum Magnets. Nature 549, pages242–246(2017).
        `arXiv:1704.05018v2 <https://arxiv.org/pdf/1704.05018v2.pdf#section*.11>`_

    """

    def __init__(
        self,
        maxiter: int = 100,
        blocking: bool = False,
        allowed_increase: float | None = None,
        trust_region: bool = False,
        learning_rate: float | np.ndarray | Callable[[], Iterator] | None = None,
        perturbation: float | np.ndarray | Callable[[], Iterator] | None = None,
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
        size_full_batch: int | None = None
    ) -> None:
        r"""
        Args:
            maxiter: The maximum number of iterations. Note that this is not the maximal number
                of function evaluations.
            blocking: If True, only accepts updates that improve the loss (up to some allowed
                increase, see next argument).
            allowed_increase: If ``blocking`` is ``True``, this argument determines by how much
                the loss can increase with the proposed parameters and still be accepted.
                If ``None``, the allowed increases is calibrated automatically to be twice the
                approximated standard deviation of the loss function.
            trust_region: If ``True``, restricts the norm of the update step to be :math:`\leq 1`.
            learning_rate: The update step is the learning rate is multiplied with the gradient.
                If the learning rate is a float, it remains constant over the course of the
                optimization. If a NumPy array, the :math:`i`-th element is the learning rate for
                the :math:`i`-th iteration. It can also be a callable returning an iterator which
                yields the learning rates for each optimization step.
                If ``learning_rate`` is set ``perturbation`` must also be provided.
            perturbation: Specifies the magnitude of the perturbation for the finite difference
                approximation of the gradients. See ``learning_rate`` for the supported types.
                If ``perturbation`` is set ``learning_rate`` must also be provided.
            start_point: Specifies what self.last_iteration should be set to. Used to start the
                perturbation and learning rate iterators at a latter point.
            last_avg: Return the average of the ``last_avg`` parameters instead of just the
                last parameter values.
            resamplings: The number of times the gradient (and Hessian) is sampled using a random
                direction to construct a gradient estimate. Per default the gradient is estimated
                using only one random direction. If an integer, all iterations use the same number
                of resamplings. If a dictionary, this is interpreted as
                ``{iteration: number of resamplings per iteration}``.
            perturbation_dims: The number of perturbed dimensions. Per default, all dimensions
                are perturbed, but a smaller, fixed number can be perturbed. If set, the perturbed
                dimensions are chosen uniformly at random.
            second_order: If True, use 2-SPSA instead of SPSA. In 2-SPSA, the Hessian is estimated
                additionally to the gradient, and the gradient is preconditioned with the inverse
                of the Hessian to improve convergence.
            regularization: To ensure the preconditioner is symmetric and positive definite, the
                identity times a small coefficient is added to it. This generator yields that
                coefficient.
            hessian_delay: Start multiplying the gradient with the inverse Hessian only after a
                certain number of iterations. The Hessian is still evaluated and therefore this
                argument can be useful to first get a stable average over the last iterations before
                using it as preconditioner.
            lse_solver: The method to solve for the inverse of the Hessian. Per default an
                exact LSE solver is used, but can e.g. be overwritten by a minimization routine.
            initial_hessian: The initial guess for the Hessian. By default the identity matrix
                is used.
            callback: A callback function passed information in each iteration step. The
                information is, in this order: the number of function evaluations, the parameters,
                the function value, the stepsize, whether the step was accepted.
            termination_checker: A callback function executed at the end of each iteration step. The
                arguments are, in this order: the parameters, the function value, the number
                of function evaluations, the stepsize, whether the step was accepted. If the callback
                returns True, the optimization is terminated.
                To prevent additional evaluations of the objective method, if the objective has not yet
                been evaluated, the objective is estimated by taking the mean of the objective
                evaluations used in the estimate of the gradient.
            size_full_batch: Optional. Number of circuits in a full batch (usually 12 in this application).
                Defaults to None.


        Raises:
            ValueError: If ``learning_rate`` or ``perturbation`` is an array with less elements
                than the number of iterations.


        """
        super().__init__()

        # general optimizer arguments
        self.maxiter = maxiter
        self.trust_region = trust_region
        self.callback = callback
        self.termination_checker = termination_checker
        self._size_full_batch = size_full_batch

        # if learning rate and perturbation are arrays, check they are sufficiently long
        for attr, name in zip([learning_rate, perturbation], ["learning_rate", "perturbation"]):
            if isinstance(attr, (list, np.ndarray)):
                if len(attr) < maxiter:
                    raise ValueError(f"Length of {name} is smaller than maxiter ({maxiter}).")

        self.learning_rate = learning_rate
        self.perturbation = perturbation
        self.lr_iterator = None
        self._lr_iterator_copy = None
        self.p_iterator = None
        self.last_iteration = start_point

        # SPSA specific arguments
        self.blocking = blocking
        self.allowed_increase = allowed_increase
        self.last_avg = last_avg
        self.resamplings = resamplings
        self.perturbation_dims = perturbation_dims

        # 2-SPSA specific arguments
        if regularization is None:
            regularization = 0.01

        self.second_order = second_order
        self.hessian_delay = hessian_delay
        self.lse_solver = lse_solver
        self.regularization = regularization
        self.initial_hessian = initial_hessian

        # runtime arguments
        self._nfev: int  = 0  # the number of function evaluations
        self._nextfev: int = 0 # the number of evaluations of the function for next value
        self._smoothed_hessian: np.ndarray | None = None  # smoothed average of the Hessians
        
        self._hyperparameters = {
            "a": None,
            "alpha": None,
            "stability_constant": None,
            "c": None,
            "gamma": None,
        }

    def set_learning_rate(self, value):
        self.learning_rate = value

    def set_perturbation(self, value):
        self.perturbation = value

    def set_init_hessian(self, value):
        self.initial_hessian = value

    def set_allowed_increase(self, value):
        self.allowed_increase = value
        
    def get_hyperparameters(self) -> dict[str, float | None]:
        """
        Return the SPSA power-series hyperparameters.

        Returns
        -------
        dict
            Dictionary with keys:
            - 'a'
            - 'alpha'
            - 'stability_constant'
            - 'c'
            - 'gamma'

        Notes
        -----
        These values are only guaranteed to be available if the optimizer
        was calibrated internally or if they were explicitly set through
        a dedicated SPSA power-series setter.
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
        Set SPSA learning-rate and perturbation schedules as power series and
        store the associated hyperparameters.

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
            Stability constant A in the learning-rate schedule.
        """
        def learning_rate(n_start: int = 0):
            return powerseries(a, alpha, offset=stability_constant, n_start=n_start)

        def perturbation(n_start: int = 0):
            return powerseries(c, gamma, n_start=n_start)

        self.learning_rate = learning_rate
        self.perturbation = perturbation
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
        stability_constant: float = 0,
        target_magnitude: float | None = None,  # 2 pi / 10
        alpha: float = 0.602,
        gamma: float = 0.101,
        modelspace: bool = False,
        max_evals_grouped: int = 1,
    ) -> tuple[Callable, Callable]:
        r"""Calibrate SPSA parameters with a power series as learning rate and perturbation coeffs.

        The power series are:

        .. math::

            a_k = \frac{a}{(A + k + 1)^\alpha}, c_k = \frac{c}{(k + 1)^\gamma}

        Args:
            loss: The loss function.
            initial_point: The initial guess of the iteration.
            c: The initial perturbation magnitude.
            stability_constant: The value of `A`.
            target_magnitude: The target magnitude for the first update step, defaults to
                :math:`2\pi / 10`.
            alpha: The exponent of the learning rate power series.
            gamma: The exponent of the perturbation power series.
            modelspace: Whether the target magnitude is the difference of parameter values
                or function values (= model space).
            max_evals_grouped: The number of grouped evaluations supported by the loss function.
                Defaults to 1, i.e. no grouping.

        Returns:
            tuple(generator, generator): A tuple of power series generators, the first one for the
                learning rate and the second one for the perturbation.
        """
        logger.info("SPSA: Starting calibration of learning rate and perturbation.")
        if target_magnitude is None:
            target_magnitude = 2 * np.pi / 10

        dim = len(initial_point)

        # compute the average magnitude of the first step
        steps = 25
        points = []
        for i in range(steps):
            # compute the random direction
            pert = bernoulli_perturbation(dim)
            points += [initial_point + c * pert, initial_point - c * pert]

        losses = _batch_evaluate(loss, points, max_evals_grouped)

        avg_magnitudes = 0.0
        for i in range(steps):
            logger.info("Calculating average magnitudes for learning rate power series.")
            delta = losses[2 * i] - losses[2 * i + 1]
            avg_magnitudes += np.abs(delta / (2 * c))

        avg_magnitudes /= steps

        if modelspace:
            a = target_magnitude / (avg_magnitudes**2)
        else:
            a = target_magnitude / avg_magnitudes

        # compute the rescaling factor for correct first learning rate
        if a < 1e-10:
            warnings.warn(f"Calibration failed, using {target_magnitude} for `a`")
            a = target_magnitude

        logger.info("Finished calibration:")
        logger.info(
            " -- Learning rate: a / ((A + n) ^ alpha) with a = %s, A = %s, alpha = %s",
            a,
            stability_constant,
            alpha,
        )
        logger.info(" -- Perturbation: c / (n ^ gamma) with c = %s, gamma = %s", c, gamma)

        # set up the power series iterator
        def learning_rate(n_start: int = 1):
            return powerseries(a, alpha, stability_constant, n_start)

        def perturbation(n_start: int = 1):
            return powerseries(c, gamma, n_start=n_start)

        return learning_rate, perturbation

    @staticmethod
    def estimate_stddev(
        loss: Callable[[np.ndarray], float],
        initial_point: np.ndarray,
        avg: int = 25,
        max_evals_grouped: int = 1,
    ) -> float:
        """Estimate the standard deviation of the loss function."""
        losses = _batch_evaluate(loss, avg * [initial_point], max_evals_grouped)
        return np.std(losses)

    @property
    def settings(self) -> dict[str, Any]:
        # if learning rate or perturbation are custom iterators expand them
        if callable(self.learning_rate):
            iterator = self.learning_rate()
            learning_rate = np.array([next(iterator) for _ in range(self.maxiter)])
        else:
            learning_rate = self.learning_rate

        if callable(self.perturbation):
            iterator = self.perturbation()
            perturbation = np.array([next(iterator) for _ in range(self.maxiter)])
        else:
            perturbation = self.perturbation

        return {
            "maxiter": self.maxiter,
            "learning_rate": learning_rate,
            "perturbation": perturbation,
            "trust_region": self.trust_region,
            "blocking": self.blocking,
            "allowed_increase": self.allowed_increase,
            "resamplings": self.resamplings,
            "perturbation_dims": self.perturbation_dims,
            "second_order": self.second_order,
            "hessian_delay": self.hessian_delay,
            "regularization": self.regularization,
            "lse_solver": self.lse_solver,
            "initial_hessian": self.initial_hessian,
            "callback": self.callback,
            "termination_checker": self.termination_checker,
        }
    
    def _point_sample(self, loss, x, eps, delta1, delta2, **kwargs):
        """A single sample of the gradient at position ``x`` in direction ``delta``."""
        # points to evaluate
        points = [x + eps * delta1, x - eps * delta1]
        logging.info(f"Perturbation with strength {eps} in directions {delta1}.")
        logging.info(f"Evaluating with perturbed parameters: {points} in this resampling.")
        self._nfev += 2

        if self.second_order:
            points += [x + eps * (delta1 + delta2), x + eps * (-delta1 + delta2)]
            logging.info(f"Evaluating with additional parameters: {points[:-2]} due to second order being set.")
            self._nfev += 2

        ncpg = kwargs.get('num_circs_per_group')
        uci = kwargs.get('used_circs_indices')
        # batch evaluate the points (if possible)
        values = _batch_evaluate(loss, points, self._max_evals_grouped, 
                                 num_circs_per_group = ncpg, used_circs_indices = uci)
        plus = values[0]
        minus = values[1]
        gradient_sample = (plus - minus) / (2 * eps) * delta1

        hessian_sample = None
        if self.second_order:
            diff = (values[2] - plus) - (values[3] - minus)
            diff /= 2 * eps**2

            rank_one = np.outer(delta1, delta2)
            hessian_sample = diff * (rank_one + rank_one.T) / 2

        return np.mean(values), gradient_sample, hessian_sample

    def _point_estimate(self, loss, x, eps, num_samples, **kwargs):
        """The gradient estimate at point x."""
        # set up variables to store averages
        value_estimate = 0
        gradient_estimate = np.zeros(x.size)
        hessian_estimate = np.zeros((x.size, x.size))
        
        ncpg = kwargs.get('num_circs_per_group')
        uci = kwargs.get('used_circs_indices')

        # iterate over the directions
        deltas1 = [
            bernoulli_perturbation(x.size, self.perturbation_dims) for _ in range(num_samples)
        ]

        if self.second_order:
            deltas2 = [
                bernoulli_perturbation(x.size, self.perturbation_dims) for _ in range(num_samples)
            ]
        else:
            deltas2 = None

        for i in range(num_samples):
            delta1 = deltas1[i]
            delta2 = deltas2[i] if self.second_order else None

            value_sample, gradient_sample, hessian_sample = self._point_sample(
                loss, x, eps, delta1, delta2, num_circs_per_group=ncpg,
                used_circs_indices = uci
            )
            value_estimate += value_sample
            gradient_estimate += gradient_sample

            if self.second_order:
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
        iteration: int = 0,
        lse_solver: Callable[[np.ndarray, np.ndarray], np.ndarray] = None,
        **kwargs
    ) -> tuple[float, np.ndarray]:
        """
        Compute an estimate of the function value (loss) and the gradient at a given point x
        using the Simultaneous Perturbation Stochastic Approximation (SPSA) method.

        Parameters
        ----------
        loss : Callable[[np.ndarray], float]
            The loss function to be minimized.
        x : np.ndarray
            The current set of parameters.
        iteration : int, optional
            The current iteration number. Defaults to 0.
        lse_solver : Callable[[np.ndarray, np.ndarray], np.ndarray], optional
            A solver function to compute the inverse Hessian multiplication in 2-SPSA.
            Defaults to None.
        kwargs
        """
        # compute the perturbations
        if isinstance(self.resamplings, dict):
            num_samples = self.resamplings.get(iteration, 1)
        else:
            num_samples = self.resamplings
            
        ncpg = kwargs.get('num_circs_per_group')
        uci = kwargs.get('used_circs_indices')

        if self.p_iterator is None:
            assert self.lr_iterator is None, "Learn rate iterator was set without setting perturbation."
            self._create_iterators()

        # accumulate the number of samples
        fx_estimate, gradient, hessian = self._point_estimate(loss, x, next(self.p_iterator), 
                                                        num_samples, 
                                                        num_circs_per_group = ncpg, 
                                                        used_circs_indices = uci)

        # precondition gradient with inverse Hessian, if specified
        if self.second_order:
            smoothed = iteration / (iteration + 1) * self._smoothed_hessian + 1 / (iteration + 1) * hessian
            self._smoothed_hessian = smoothed

            if iteration > self.hessian_delay:
                spd_hessian = _make_spd(smoothed, self.regularization)

                # solve for the gradient update
                gradient = np.real(lse_solver(spd_hessian, gradient))

        return fx_estimate, gradient
    
    def process_update(
        self,
        gradient_estimate: np.ndarray,
        x: np.ndarray,
        fx: float,
        fun: Callable[[np.ndarray], float],
        fun_next: Callable[[np.ndarray], float] | None,
        iteration_start: float = 0,
        iteration: int = 0,
        ) -> tuple[bool, np.ndarray, float]:
        """
        Process an update step in the optimization, applying trust region constraints,
        blocking mechanisms, and function evaluations as needed.

        Parameters
        ----------
        gradient_estimate : np.ndarray
            The computed gradient step.
        x : np.ndarray
            The current parameter values.
        fx : float
            The current function value.
        fun : Callable[[np.ndarray], float]
            The objective function.
        fun_next : Callable[[np.ndarray], float] | None
            An optional function to evaluate the objective at the next step.
        iteration_start : float
            The timestamp when the iteration started.
        iteration : int
            The current iteration number.

        Returns
        -------
        tuple[bool, np.ndarray, float]
            A tuple containing a boolean indicating whether to skip the update,
            the updated parameter values, and the updated function value.
        """
    
    # trust region
        if self.trust_region:
            norm = np.linalg.norm(gradient_estimate)
            if norm > 1:  # stop from dividing by 0
                gradient_estimate = gradient_estimate / norm

        logger.info(f"Gradient was estimated to be: {gradient_estimate}.")
        if self.lr_iterator is None:
            assert self.p_iterator is None, "Perturbation iterator was set without setting learn rate."
            self._create_iterators()
        learn_rate = next(self.lr_iterator)
        logger.info(f"Learn rate at this point is {learn_rate}")

        # compute next parameter value
        update = gradient_estimate * learn_rate
        x_next = x - update
        fx_next = None

        logger.info(f"Next set of parameters is: {x_next}.")

        # blocking
        if self.blocking:
            if fun_next is None:
                logger.info("Calculating vlaue of cost function at next point.")
                self._nfev += 1
                fx_next = fun(x_next)
            else:
                logger.info("Calculating vlaue of cost function at next point with custom function.")
                self._nextfev += 1
                fx_next = fun_next(x_next)

            if fx + self.allowed_increase <= fx_next:  # accept only if loss improved
                if self.callback is not None:
                    self.callback(
                        self._nfev,  # number of function evals
                        x_next,  # next parameters
                        fx_next,  # loss at next parameters
                        np.linalg.norm(update),  # size of the update step
                        False, # not accepted
                    )  

                logger.info(
                    "Iteration %s/%s rejected in %s.",
                    iteration,
                    self.maxiter + 1,
                    time() - iteration_start,
                )
                # Continue outer loop
                return True, x, fx
                
        return False, x_next, fx_next
    
    def minimize(
        self,
        fun: Callable[[POINT], float],
        x0: POINT,
        fun_next: Callable[[POINT], float] | None = None,
        jac: Callable[[POINT], POINT] | None = None,
        bounds: list[tuple[float, float]] | None = None,
        **kwargs
    ) -> OptimizerResult:
        logger.info("Started minimization of loss funcion.")
        # If the iterators have not been set, set them now.
        if self.p_iterator is None and self.lr_iterator is None:
            self._create_iterators(fun, x0)

        if self.lse_solver is None:
            logger.info("Setting default linear solver.")
            lse_solver = np.linalg.solve
        else:
            logger.info("Setting custom solver.")
            lse_solver = self.lse_solver

        # prepare some initials
        logger.info("Initializing Hessian.")
        x = np.asarray(x0)
        if self.initial_hessian is None:
            logger.info("Using default identity Hessian.")
            self._smoothed_hessian = np.identity(x.size)
        else:
            logger.info("Using provided Hessian.")
            self._smoothed_hessian = self.initial_hessian

        self._nfev = 0
        if fun_next:
            self._nextfev = 0

        # if blocking is enabled we need to keep track of the function values
        if self.blocking:
            logger.info("Evaluating function at initial point for blocking option.")
            fx = fun(x)  # pylint: disable=invalid-name

            self._nfev += 1
            if self.allowed_increase is None:
                logger.info("Calculating allowed increase with standard deviation.")
                self.allowed_increase = 2 * self.estimate_stddev(
                    fun, x, max_evals_grouped=self._max_evals_grouped
                )
                logger.info(f"Allowed increase is: {self.allowed_increase}")
                self.set_allowed_increase(self.allowed_increase)
        else:
            fx = None

        use_epochs = kwargs.get('use_epochs')
        
        if use_epochs:
            logger.info(f"SPSA: Starting optimization with initial parameters {x0} doing epochs.")
            logger.info("Interpreting max number of iterations as max number of epochs.")
        else:
            logger.info(f"SPSA: Starting optimization with initial parameters {x0}.")
        start = time()

        # keep track of the last few steps to return their average
        last_steps = deque([x])

        ncpg = kwargs.get('num_circs_per_group')
        ncpb = kwargs.get('num_circs_per_batch')
        if use_epochs:
            # Total number of circuits (complete data set)
            total_circuits = self._size_full_batch if self._size_full_batch else 12
            if not ncpb:
                ncpb = 3
        logger.info(f"Setting number of circuits per batch to {ncpb}.")
        # use a local variable and while loop to keep track of the number of iterations
        # if the termination checker terminates early
        k = 0
        logger.info("Starting optimization loop")
        while k < self.maxiter:
            k += 1
            self.last_iteration += 1
            current_learn_rate = next(self._lr_iterator_copy)
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
                    fx_estimate, gradient_estimate = self.compute_loss_and_gradient_estimate(
                        fun, x, k, lse_solver, used_circs_indices = indices
                        )
                    # Calculate next set of parameters and, if blocking is set, the value of cost function at that point.
                    # If blocking option is set, decide whether the parameters are accepted.
                    skip, x_next, fx_next = self.process_update(
                        gradient_estimate, x, fx, fun, fun_next, iteration_start, k
                        )
                    if skip:
                        continue
                    # Update values
                    x = x_next
                    fx = fx_next
                logger.info(f"Epoch {k}/{self.maxiter} finished in {time() - iteration_start}")
            # Compute updates iteration by iteration
            else:
                fx_estimate, gradient_estimate = self.compute_loss_and_gradient_estimate(
                    fun, x, k, lse_solver, num_circs_per_group = ncpg
                    )
                skip, x_next, fx_next = self.process_update(
                    gradient_estimate, x, fx, fun, fun_next, iteration_start, k
                    )
                if skip:
                    continue
                # Update values
                x = x_next
                fx = fx_next
                logger.info("Iteration %s/%s done in %s.", k, self.maxiter + 1, time() - iteration_start)
                
            if self.callback is not None:
                # if we didn't evaluate the function yet, do it now
                if not self.blocking:
                    if fun_next is None:
                        logger.info("Calculating next step for the callback, which takes another function evaluation.")
                        self._nfev += 1
                        fx_next = fun(x_next)
                    else:
                        logger.info("Calculating next step for the callback with custom function.")
                        self._nextfev += 1
                        fx_next = fun_next(x_next)

                self.callback(
                    self._nfev,  # number of function evals
                    x_next,  # next parameters
                    fx_next,  # loss at next parameters
                    np.linalg.norm(gradient_estimate*current_learn_rate),  # size of the update step
                    True, # accepted
                )
                
            # update the list of the last ``last_avg`` parameters
            if self.last_avg > 1:
                last_steps.append(x_next)
                if len(last_steps) > self.last_avg:
                    last_steps.popleft()

            if self.termination_checker is not None:
                fx_check = fx_estimate if fx_next is None else fx_next
                if self.termination_checker(
                    self._nfev, x_next, fx_check, np.linalg.norm(gradient_estimate*current_learn_rate), True
                ):
                    logger.info(f"terminated optimization at {k}/{self.maxiter} iterations")
                    break

        logger.info("SPSA: Finished in %s", time() - start)

        if self.last_avg > 1:
            x = np.mean(last_steps, axis=0)

        result = OptimizerResult()
        result.x = x
        if fun_next is None:
            logger.info("Calculating cost funtion value for final parameters.")
        else:
            logger.info("Calculating custom cost funtion value for final parameters.")
        result.fun = fun(x) if fun_next is None else fun_next(x)
        result.nfev = self._nfev
        result.nit = k

        return result
    
    def get_support_level(self):
        """Get the support level dictionary."""
        return {
            "gradient": OptimizerSupportLevel.ignored,
            "bounds": OptimizerSupportLevel.ignored,
            "initial_point": OptimizerSupportLevel.required,
        }
        
    def _create_iterators(self, fun: Callable | None = None, x0: list | np.ndarray = None):
        """Create learn rate and perturbation iterators."""
        # ensure learning rate and perturbation are correctly set: either none or both
        # this happens only here because for the calibration the loss function is required
        if self.learning_rate is None and self.perturbation is None:
            logger.info("Entered calibration step")
            get_eta, get_eps = self.calibrate(fun, x0, max_evals_grouped=self._max_evals_grouped)
            logger.info("Setting learning rate and perturbation to use in case of interruption of current run.") 
            self.set_learning_rate(get_eta)
            self.set_perturbation(get_eps)
        else:
            logger.info("Skipped calibration and entered validation of existing learning rate and perturbation.")
            get_eta, get_eps = _validate_pert_and_learningrate(
                self.perturbation, self.learning_rate
            )
        logger.info(f"Creating learn rate and perturbation iterators starting from {self.last_iteration}.")
        # eta = Learning rate itarator eps = Perturbation strength iterator
        eta, eps = get_eta(n_start = self.last_iteration), get_eps(n_start = self.last_iteration)
        logger.info("Setting learn rate and perturbation iterator attributes.")
        self.lr_iterator, self._lr_iterator_copy = itertools.tee(eta)
        self.p_iterator = eps


def bernoulli_perturbation(dim, perturbation_dims=None):
    """Get a Bernoulli random perturbation."""
    if perturbation_dims is None:
        return 1 - 2 * algorithm_globals.random.binomial(1, 0.5, size=dim)

    pert = 1 - 2 * algorithm_globals.random.binomial(1, 0.5, size=perturbation_dims)
    indices = algorithm_globals.random.choice(
        list(range(dim)), size=perturbation_dims, replace=False
    )
    result = np.zeros(dim)
    result[indices] = pert

    return result

def powerseries(eta=0.01, power=2, offset=0, n_start = 0):
    """Yield a series decreasing by a power law."""

    n = n_start+1
    while True:
        yield eta / ((n + offset) ** power)
        n += 1


def constant(eta=0.01):
    """Yield a constant series."""

    while True:
        yield eta


def _batch_evaluate(function, points, max_evals_grouped, unpack_points=False, **kwargs):
    """Evaluate a function on all points with batches of max_evals_grouped.

    The points are a list of inputs, as ``[in1, in2, in3, ...]``. If the individual
    inputs are tuples (because the function takes multiple inputs), set ``unpack_points`` to ``True``.
    """

    # if the function cannot handle lists of points as input, cover this case immediately
    if max_evals_grouped is None or max_evals_grouped == 1:
        # support functions with multiple arguments where the points are given in a tuple
        function_batch = []
        # Number of points used to sample the gradient in current "resampling"
        num_steps = len(points)
        ncpg = kwargs.get('num_circs_per_group')
        uci = kwargs.get('used_circs_indices')
        for i, point in enumerate(points):
            logger.info(f"Evalutation {i+1}/{num_steps} for current sampling.")
            # The following is used when using "mini-batches" for the training of the QAE.
            if ncpg:
                # A "group" is, for example, {|0_err_n>, |1_err_n>, |+_err_n>}, that is,.
                # each starting state with the same error (or no error). The least amount 
                # of circuits ran is 3 and the most 12 (all starting states and all errors).
                num_circs_per_group = min(ncpg, 4)
                logger.info(f"Using {num_circs_per_group} per group.")
                inds = random.sample([0,1,2,3], num_circs_per_group)
                # The circuits used are defined in a list. That list
                # goes like [|0>, |0_err0>, |0_err1>, |0_err2>, |1>, |1_err0>,...]
                # so we choose the first num_circs_per_group indices for the 
                # starting state |0>, then +4 for the |1> states, and then +8
                # for the |+> state.
                inds += [ind + 4 for ind in inds]
                inds += [ind + 8 for ind in inds[0:num_circs_per_group]]
                function_batch.append(function(*point, used_circs_indices = inds)) if isinstance(point, tuple) else function_batch.append(function(point, used_circs_indices = inds))
            elif not (uci is None):
                # Here we specify which circuits will be used. Used for epochs training.
                logger.info(f"Used indices are: {uci}.")
                function_batch.append(function(*point, used_circs_indices = uci)) if isinstance(point, tuple) else function_batch.append(function(point, used_circs_indices = uci))
            else:
                function_batch.append(function(*point)) if isinstance(point, tuple) else function_batch.append(function(point))
        return function_batch

    num_points = len(points)

    # get the number of batches
    num_batches = num_points // max_evals_grouped
    if num_points % max_evals_grouped != 0:
        num_batches += 1

    # split the points
    batched_points = np.array_split(np.asarray(points), num_batches)

    results = []
    for batch in batched_points:
        if unpack_points:
            batch = _repack_points(batch)
            results += _as_list(function(*batch))
        else:
            results += _as_list(function(batch))

    return results


def _as_list(obj):
    """Convert a list or numpy array into a list."""
    return obj.tolist() if isinstance(obj, np.ndarray) else obj


def _repack_points(points):
    """Turn a list of tuples of points into a tuple of lists of points.
    E.g. turns
        [(a1, a2, a3), (b1, b2, b3)]
    into
        ([a1, b1], [a2, b2], [a3, b3])
    where all elements are np.ndarray.
    """
    num_sets = len(points[0])  # length of (a1, a2, a3)
    return ([x[i] for x in points] for i in range(num_sets))


def _make_spd(matrix, bias=0.01):
    identity = np.identity(matrix.shape[0])
    psd = scipy.linalg.sqrtm(matrix.dot(matrix))
    return psd + bias * identity


def _validate_pert_and_learningrate(perturbation, learning_rate):
    import itertools
    if learning_rate is None or perturbation is None:
        raise ValueError("If one of learning rate or perturbation is set, both must be set.")

    if isinstance(perturbation, float):

        def get_eps(n_start = 0):
            return constant(perturbation)

    elif isinstance(perturbation, (list, np.ndarray)):

        def get_eps(n_start = 0):
            return itertools.islice(perturbation, n_start, None)

    else:
        get_eps = perturbation

    if isinstance(learning_rate, float):

        def get_eta(n_start = 0):
            return constant(learning_rate)

    elif isinstance(learning_rate, (list, np.ndarray)):

        def get_eta(n_start = 0):
            return itertools.islice(learning_rate, n_start, None)

    else:
        get_eta = learning_rate

    return get_eta, get_eps
