import json
import datetime
import logging
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import clear_output
from collections.abc import Callable, Sequence
from typing import Literal
from uncertainties.core import AffineScalarFunc

logger = logging.getLogger(__name__)


def _nominal_value(value: float | AffineScalarFunc) -> float:
    """
    Return the nominal float value.

    Parameters
    ----------
    value : float or AffineScalarFunc
        Scalar value that may carry uncertainty metadata.

    Returns
    -------
    float
        Nominal value as a Python float.
    """
    if isinstance(value, AffineScalarFunc):
        return float(value.nominal_value)
    return float(value)


def _std_value(value: float | AffineScalarFunc) -> float:
    """
    Return the standard deviation if available.

    Parameters
    ----------
    value : float or AffineScalarFunc
        Scalar value that may carry uncertainty metadata.

    Returns
    -------
    float
        Standard deviation if present, otherwise 0.0.
    """
    if isinstance(value, AffineScalarFunc):
        return float(value.std_dev)
    return 0.0


def _linear_slope(values: Sequence[float]) -> float:
    """
    Estimate the slope of a sequence using a linear fit.

    Parameters
    ----------
    values : Sequence[float]
        Window of scalar values.

    Returns
    -------
    float
        Slope of the best-fit line.
    """
    y = np.asarray(values, dtype=float)
    if y.size < 2:
        return 0.0
    x = np.arange(y.size, dtype=float)
    return float(np.polyfit(x, y, deg=1)[0])


def _window_improvement(values: Sequence[float], mode: Literal["min", "max"]) -> float:
    """
    Compute the best improvement over a window.

    Parameters
    ----------
    values : Sequence[float]
        Window of objective values.
    mode : {"min", "max"}
        Whether the objective is being minimized or maximized.

    Returns
    -------
    float
        Best improvement over the window, expressed as a positive number
        when progress was made.
    """
    y = np.asarray(values, dtype=float)
    if y.size == 0:
        return 0.0

    if mode == "min":
        return float(y[0] - np.min(y))
    return float(np.max(y) - y[0])


def create_live_plot_callback(
    counts: list[int], 
    values: list[float], 
    params: list[list[float]], 
    stepsize: list[float], 
    json_file_path: str | None = None,
    store_data: bool = False,
    extra_eval_freq: int | None = None,
    cost_extra: Callable | None = None,
    values_extra: list[float] | None = None,
    gradient_norms: list[float] | None = None,
    plot: bool = True,
    use_epoch: bool = False
) -> Callable[[int, list[float], float, float, bool], None]:
    """
    Create a callback function that stores intermediate results, updates plots, and optionally writes data to a JSON file.
    
    Parameters
    ----------
    counts : list[int]
        The list where iteration counts will be stored.
    values : list[float]
        The list where mean values (e.g., cost function values) will be stored.
    params : list[list[float]]
        The list where parameter vectors will be stored.
    stepsize : list[float]
        The list where step sizes will be stored.
    json_file_path : str
        The path to the JSON file where data will be saved.
    store_data : bool, optional
        Whether to store the data in the json file provided. Defaults to false.
    extra_eval_freq : int, optional
        Number of iterations that pass between evaluations of an extra cost function.
        Used for evaluations in Hardware of the next point.
    cost_extra : Callable, optional
        The extra cost function to be evaluated every extra_eval_freq iterations.
    values_extra : list[float], optional
        The array to store the values from the extra cost function.
    gradient_norms : list[float] or None, optional
        Storage for gradient norms, if gradient information is provided.
    plot : bool, optional
        Whether to do a live plot. Defaults to True.
    use_epoch : bool, optional
        Whether the callback is calculated on epoch end (True) or on iteration end (False). Defaults to False.
    
    Returns
    -------
    Callable[..., None]
        Callback function. It accepts the usual five positional arguments

        `(eval_count, parameters, mean, stp_size, accepted)`

        and may also accept an optional keyword argument `gradient`.
    """
    
    if extra_eval_freq is not None:
        assert cost_extra is not None and values_extra is not None, (
            "Must provide extra cost function and array in which to store extra values."
        )
        
    if store_data and json_file_path is None:
        raise ValueError("`json_file_path` must be provided when `store_data=True`.")
    
    def _record_gradient(gradient: Sequence[float] | np.ndarray | None) -> None:
        """
        Record the norm of a gradient estimate.

        Parameters
        ----------
        gradient : Sequence[float] or np.ndarray or None
            Gradient-like object. If None, nothing is recorded.
        """
        if gradient is None or gradient_norms is None:
            return

        grad = np.asarray(gradient, dtype=float)
        gradient_norms.append(float(np.linalg.norm(grad)))

    # Initialize the function with default behavior
    def store_intermediate_result_plot_live(
        eval_count: int, 
        parameters: list[float], 
        mean: float, 
        stp_size: float, 
        accepted: bool,
        gradient: Sequence[float] | np.ndarray | None = None
        ) -> None:
        """
        Store and visualize one callback record.

        Parameters
        ----------
        eval_count : int
            Iteration or epoch counter.
        parameters : list[float]
            Current parameter vector.
        mean : float
            Current objective value.
        stp_size : float
            Current optimizer step size.
        accepted : bool
            Whether the candidate step was accepted.
        gradient : Sequence[float] or np.ndarray or None, optional
            Gradient estimate associated with this callback event.
        """
        if plot:
            clear_output(wait=True)  # Clears the previous output in the notebook
        
        counts.append(int(eval_count))
        values.append(float(mean))
        params.append(list(parameters))
        stepsize.append(float(stp_size))
        _record_gradient(gradient)
        
        if extra_eval_freq is not None and len(counts) % extra_eval_freq == 0:
            logger.info("Extra cost evaluation with provided function at parameters: %s", parameters)
            value_extra = cost_extra(parameters)
            logger.info("Value of extra evaluation was: %s", value_extra)
            values_extra.append(value_extra)
            
        progress_str = "Epoch" if use_epoch else "Iteration"
        
        # Store data in the file if global `store_data` flag is True
        if store_data:
            data = {
                progress_str: int(eval_count),
                "Objective": float(mean),
                "Accepted": bool(accepted),
                "Step size": float(stp_size),
                "Params": tuple(parameters),
                "Time": str(datetime.datetime.now()),
            }
            if gradient_norms is not None and gradient_norms:
                data["Gradient norm"] = gradient_norms[-1]
            if values_extra is not None and values_extra:
                data["Extra objective"] = _nominal_value(values_extra[-1])

            with open(json_file_path, "a", encoding="utf-8") as json_file:
                json.dump(data, json_file)
                json_file.write("\n")

        if plot:
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.set_title("Cost Evolution")
            ax.set_xlabel(progress_str)
            ax.set_ylabel("Objective")
            ax.plot(counts, values, "b.", label="Objective")

            if extra_eval_freq is not None and values_extra is not None and len(values_extra) > 0:
                x_extra = counts[extra_eval_freq - 1 :: extra_eval_freq][: len(values_extra)]
                y_extra = [_nominal_value(v) for v in values_extra]
                yerr_extra = [_std_value(v) for v in values_extra]

                ax.errorbar(
                    x_extra,
                    y_extra,
                    yerr=yerr_extra,
                    fmt="r.",
                    capsize=3,
                    label="Extra evals",
                )

            ax.legend(loc="best")
            plt.show()

    setattr(store_intermediate_result_plot_live, "record_gradient", _record_gradient)
    return store_intermediate_result_plot_live

def create_callback_args(
    json_file_path: str | None = None,
    store_data: bool = False,
    extra_eval_freq: int | None = None,
    cost_extra: Callable | None = None,
    plot: bool = True,
    use_epoch: bool = False,
    track_gradient_norm: bool = False,
) -> tuple[list[int], list[float], list[list[float]], list[float], dict]:
    """
    Generate the arguments required by `create_live_plot_callback`.

    Parameters
    ----------
    json_file_path : str or None, optional
        Path to the JSONL file for storing callback data.
    store_data : bool, optional
        Whether to store callback data.
    extra_eval_freq : int or None, optional
        Frequency of extra cost evaluations.
    cost_extra : Callable or None, optional
        Extra cost function to evaluate.
    plot : bool, optional
        Whether to update plots.
    use_epoch : bool, optional
        Whether to label progress in epochs rather than iterations.
    track_gradient_norm : bool, optional
        Whether to allocate storage for gradient norms.

    Returns
    -------
    tuple
        Tuple containing `(counts, values, params, stepsize, kwargs_dict)`.
        The returned kwargs dictionary can be unpacked into
        `create_live_plot_callback(...)`.
    """
    counts: list[int] = []
    values: list[float] = []
    params: list[list[float]] = []
    stepsize: list[float] = []
    values_extra = [] if extra_eval_freq is not None else None
    gradient_norms = [] if track_gradient_norm else None

    kwargs = {
        "json_file_path": json_file_path,
        "store_data": store_data,
        "extra_eval_freq": extra_eval_freq,
        "cost_extra": cost_extra,
        "values_extra": values_extra,
        "gradient_norms": gradient_norms,
        "plot": plot,
        "use_epoch": use_epoch,
    }

    return counts, values, params, stepsize, kwargs


class TerminationChecker:
    """
    Termination checker for iterative optimization.

    This class supports four independent stopping criteria:

    1. Reaching a target objective value.
    2. Plateau detection: the objective slope is small and best improvement
       over a recent window is negligible.
    3. Noisy wandering: the recent window has high variance but little net
       progress.
    4. Gradient flattening: recent gradient norms remain small for long enough
       and objective improvement over the same window is negligible.

    Attributes
    ----------
    values : list[float]
        History of objective values.
    grad_norms : list[float]
        History of gradient norms.
    last_reason : str or None
        Reason associated with the most recent termination event.
    """

    def __init__(
        self,
        mode: Literal["min", "max"] = "min",
        target_value: float | None = None,
        target_tol: float = 1e-3,
        plateau_window: int | None = None,
        plateau_slope_tol: float | None = None,
        plateau_improvement_tol: float | None = None,
        noisy_window: int | None = None,
        noisy_std_tol: float | None = None,
        noisy_slope_tol: float | None = None,
        noisy_improvement_tol: float | None = None,
        gradient_window: int | None = None,
        gradient_norm_tol: float | None = None,
        gradient_norm_std_tol: float | None = None,
        gradient_improvement_tol: float | None = None,
        verbose: bool = True,
    ) -> None:
        """
        Initialize the TerminationChecker instance.

        Parameters
        ----------
        mode : {"min", "max"}, optional
            Whether the objective is being minimized or maximized.
        target_value : float or None, optional
            Target objective value.
        target_tol : float, optional
            Tolerance used when checking proximity to `target_value`.
        plateau_window : int or None, optional
            Window size used for plateau detection.
        plateau_slope_tol : float or None, optional
            Threshold on the absolute slope of the objective over the plateau window.
        plateau_improvement_tol : float or None, optional
            Threshold on the best improvement over the plateau window.
        noisy_window : int or None, optional
            Window size used for noisy wandering detection.
        noisy_std_tol : float or None, optional
            Threshold on the standard deviation over the noisy window.
        noisy_slope_tol : float or None, optional
            Threshold on the absolute slope over the noisy window.
        noisy_improvement_tol : float or None, optional
            Threshold on the best improvement over the noisy window.
        gradient_window : int or None, optional
            Window size used for gradient-flattening detection.
        gradient_norm_tol : float or None, optional
            Threshold on the median gradient norm over the gradient window.
        gradient_norm_std_tol : float or None, optional
            Threshold on the standard deviation of gradient norms over the gradient window.
        gradient_improvement_tol : float or None, optional
            Threshold on the best objective improvement over the gradient window.
        verbose : bool, optional
            Whether to print stop messages in addition to logging them.
        """

        if mode not in {"min", "max"}:
            raise ValueError("`mode` must be either 'min' or 'max'.")

        for name, window in (
            ("plateau_window", plateau_window),
            ("noisy_window", noisy_window),
            ("gradient_window", gradient_window),
        ):
            if window is not None and window < 2:
                raise ValueError(f"`{name}` must be >= 2.")
        
        for name, tol in (
            ("target_tol", target_tol),
            ("plateau_slope_tol", plateau_slope_tol),
            ("plateau_improvement_tol", plateau_improvement_tol),
            ("noisy_std_tol", noisy_std_tol),
            ("noisy_slope_tol", noisy_slope_tol),
            ("noisy_improvement_tol", noisy_improvement_tol),
            ("gradient_norm_tol", gradient_norm_tol),
            ("gradient_norm_std_tol", gradient_norm_std_tol),
            ("gradient_improvement_tol", gradient_improvement_tol),
        ):
            if tol is not None and tol < 0:
                raise ValueError(f"`{name}` must be non-negative.")
            
        self.mode = mode
        self.target_value = target_value
        self.target_tol = target_tol
        
        self.plateau_window = plateau_window
        self.plateau_slope_tol = plateau_slope_tol
        self.plateau_improvement_tol = plateau_improvement_tol

        self.noisy_window = noisy_window
        self.noisy_std_tol = noisy_std_tol
        self.noisy_slope_tol = noisy_slope_tol
        self.noisy_improvement_tol = noisy_improvement_tol

        self.gradient_window = gradient_window
        self.gradient_norm_tol = gradient_norm_tol
        self.gradient_norm_std_tol = gradient_norm_std_tol
        self.gradient_improvement_tol = gradient_improvement_tol

        self.verbose = verbose

        self.values: list[float] = []
        self.grad_norms: list[float] = []
        self.last_reason: str | None = None

    @staticmethod
    def _window_slope(values: list[float]) -> float:
        """Return the slope of a linear fit over the supplied window."""
        y = np.asarray(values, dtype=float)
        x = np.arange(len(y), dtype=float)
        return float(np.polyfit(x, y, 1)[0])

    def _window_improvement(self, values: list[float]) -> float:
        """
        Return best improvement over the window as a positive number.

        For minimization:
            first_value - min(window)

        For maximization:
            max(window) - first_value
        """
        y = np.asarray(values, dtype=float)
        if self.mode == "min":
            return float(y[0] - np.min(y))
        return float(np.max(y) - y[0])

    def update_gradient(self, gradient: list[float] | np.ndarray | None) -> None:
        """
        Record gradient norm for gradient-flattening checks.

        Parameters
        ----------
        gradient : list[float] | np.ndarray | None
            Gradient estimate. If None, nothing is recorded.
        """
        if gradient is None:
            return

        grad = np.asarray(gradient, dtype=float)
        self.grad_norms.append(float(np.linalg.norm(grad)))

    def _stop(self, reason: str, message: str) -> bool:
        self.last_reason = reason
        logger.info(message)
        if self.verbose:
            print(message)
        return True

    def __call__(
        self,
        nfev: int,
        parameters: list[float],
        value: float,
        stepsize: float,
        accepted: bool,
    ) -> bool:
        """
        Check whether optimization should terminate.

        Parameters
        ----------
        nfev : int
            Number of function evaluations.
        parameters : list[float]
            Current parameter vector.
        value : float
            Current objective value.
        stepsize : float
            Current step size.
        accepted : bool
            Whether the current step was accepted.

        Returns
        -------
        bool
            True if a stopping criterion is met, False otherwise.
        """
        del nfev, parameters, stepsize, accepted

        value = float(value)
        self.values.append(value)
        self.last_reason = None

        # Target-value convergence
        if self.target_value is not None:
            if abs(self.target_value - value) < self.tol:
                return self._stop(
                    "target_reached",
                    f"Reached target value within tolerance: {self.target_value}",
                )

        # Plateau detection
        if (
            self.plateau_window is not None
            and self.plateau_slope_tol is not None
            and self.plateau_improvement_tol is not None
            and len(self.values) >= self.plateau_window
        ):
            recent = self.values[-self.plateau_window :]
            slope = self._window_slope(recent)
            improvement = self._window_improvement(recent)

            if abs(slope) < self.plateau_slope_tol and improvement < self.plateau_improvement_tol:
                return self._stop(
                    "plateau",
                    (
                        "Detected plateau: "
                        f"slope={slope:.6g}, "
                        f"best_improvement={improvement:.6g}, "
                        f"window={self.plateau_window}"
                    ),
                )

        # Noisy wandering detection
        if (
            self.noisy_window is not None
            and self.noisy_std_tol is not None
            and self.noisy_slope_tol is not None
            and self.noisy_improvement_tol is not None
            and len(self.values) >= self.noisy_window
        ):
            recent = self.values[-self.noisy_window :]
            std_dev = float(np.std(recent))
            slope = self._window_slope(recent)
            improvement = self._window_improvement(recent)

            if (
                std_dev > self.noisy_std_tol
                and abs(slope) < self.noisy_slope_tol
                and improvement < self.noisy_improvement_tol
            ):
                return self._stop(
                    "noisy_wandering",
                    (
                        "Detected noisy wandering: "
                        f"std={std_dev:.6g}, "
                        f"slope={slope:.6g}, "
                        f"best_improvement={improvement:.6g}, "
                        f"window={self.noisy_window}"
                    ),
                )

        # Gradient flattening detection
        if (
            self.gradient_window is not None
            and self.gradient_norm_tol is not None
            and len(self.grad_norms) >= self.gradient_window
            and len(self.values) >= self.gradient_window
        ):
            recent_grad_norms = np.asarray(
                self.grad_norms[-self.gradient_window :], dtype=float
            )
            recent_values = self.values[-self.gradient_window :]

            grad_median = float(np.median(recent_grad_norms))
            grad_std = float(np.std(recent_grad_norms))
            improvement = self._window_improvement(recent_values)

            flat_enough = grad_median < self.gradient_norm_tol
            stable_enough = (
                True
                if self.gradient_norm_std_tol is None
                else grad_std < self.gradient_norm_std_tol
            )
            little_progress = (
                True
                if self.gradient_improvement_tol is None
                else improvement < self.gradient_improvement_tol
            )

            if flat_enough and stable_enough and little_progress:
                return self._stop(
                    "gradient_flattening",
                    (
                        "Detected gradient flattening: "
                        f"median_grad_norm={grad_median:.6g}, "
                        f"std_grad_norm={grad_std:.6g}, "
                        f"best_improvement={improvement:.6g}, "
                        f"window={self.gradient_window}"
                    ),
                )

        return False

    def reset(self) -> None:
        """Reset stored histories and last stop reason."""
        logger.info("Resetting termination checker for new optimization run.")
        self.values.clear()
        self.grad_norms.clear()
        self.last_reason = None
