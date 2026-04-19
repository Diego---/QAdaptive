import json
import datetime
import logging
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import clear_output
from collections.abc import Callable
from uncertainties.core import AffineScalarFunc

logger = logging.getLogger(__name__)

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
    plot : bool, optional
        Whether to do a live plot. Defaults to True.
    use_epoch : bool, optional
        Whether the callback is calculated on epoch end (True) or on iteration end (False). Defaults to False.

    cost_extra : Callable, optional
    values_extra : list[float], optional
    
    Returns
    -------
    callback : Callable
        A function that accepts the iteration count, parameters, mean value, step size, and acceptance status,
        and stores this information.
    """
    
    if extra_eval_freq is not None:
        assert cost_extra is not None and values_extra is not None, "Must provide extra cost function and array in which to store extra values."
    
    # Initialize the function with default behavior
    def store_intermediate_result_plot_live(eval_count: int, parameters: list[float], mean: float, stp_size: float, accepted: bool):
        if plot:
            clear_output(wait=True)  # Clears the previous output in the notebook
        
        counts.append(eval_count)
        values.append(mean)
        params.append(parameters)
        stepsize.append(stp_size)
        
        if extra_eval_freq is not None:
            if len(counts) % extra_eval_freq == 0:
                print("Extra cost evaluation with provided function.")
                logger.info(f"Extra cost evaluation with provided function at parameters: {parameters}.")
                value_extra = cost_extra(parameters)
                logger.info(f"Value of extra evaluations was: {value_extra}")
        
                values_extra.append(value_extra)
            
        progress_str = "Epoch" if use_epoch else "Iteration"
        
        # Store data in the file if global `store_data` flag is True
        if store_data:
            with open(json_file_path, 'a') as json_file:
                data = {
                    progress_str: len(values),
                    "Fidelity": -mean,
                    "Params": tuple(parameters),
                    "Time": str(datetime.datetime.now())
                }
                if not extra_eval_freq is None:
                    data_extra = {"Fidelity Extra": values_extra[-1] if values_extra else 0}
                    data.update(data_extra)
                    
                json.dump(data, json_file)
                json_file.write('\n')
        
        if plot:
            # Real-time plot
            plt.title("Cost Evolution")
            plt.xlabel(progress_str)
            plt.ylabel(r'$F$')
            plt.plot(range(len(values)), values, "b.")
            if extra_eval_freq is not None and len(values_extra) > 0:
                x_extra = [(2 * i) + 1 for i in range(len(values_extra))]
                # Extract nominal values and error bars
                y_extra = [v.nominal_value if isinstance(v, AffineScalarFunc) else v for v in values_extra]
                yerr_extra = [v.std_dev if isinstance(v, AffineScalarFunc) else 0 for v in values_extra]
                plt.errorbar(x_extra, y_extra, yerr=yerr_extra, fmt="r.", capsize=4, label="Hardware evals")
                # Make sure these are NumPy arrays
                x_extra = np.array(x_extra, dtype=float)
                y_extra = np.array(y_extra, dtype=float)
                yerr_extra = np.array(yerr_extra, dtype=float)
                # Plot
                plt.errorbar(x_extra, y_extra, yerr=yerr_extra, fmt="r.", capsize=2, label="Hardware evals")
            plt.show()

    return store_intermediate_result_plot_live

def create_callback_args(
    json_file_path: str | None = None,
    store_data: bool = False,
    extra_eval_freq: int | None = None,
    cost_extra: Callable | None = None,
    plot: bool = True,
    use_epoch: bool = False
) -> tuple[list[int], list[float], list[list[float]], list[float], dict]:
    """
    Generate the arguments for create_live_plot_callback.
    
    Parameters
    ----------
    json_file_path : str, optional
        Path to JSON file for storing data.
    store_data : bool, optional
        Whether to store data. Defaults to False.
    extra_eval_freq : int, optional
        Frequency of extra cost evaluations.
    cost_extra : Callable, optional
        Extra cost function to evaluate.
    plot : bool, optional
        Whether to plot results. Defaults to True.
    use_epoch : bool, optional
        Whether to use epoch tracking. Defaults to False.
    
    Returns
    -------
    tuple
        A tuple containing (counts, values, params, stepsize, kwargs_dict) ready for create_live_plot_callback.
    """
    counts = []
    values = []
    params = []
    stepsize = []
    values_extra = [] if extra_eval_freq is not None else None
    
    kwargs = {
        "json_file_path": json_file_path,
        "store_data": store_data,
        "extra_eval_freq": extra_eval_freq,
        "cost_extra": cost_extra,
        "values_extra": values_extra,
        "plot": plot,
        "use_epoch": use_epoch
    }
    
    return counts, values, params, stepsize, kwargs

class TerminationChecker:
    """
    A termination checker class for optimization algorithms.
    
    This class allows checking whether the optimization should terminate based on either a target value
    or stagnation (flat evolution) over a number of iterations. Stagnation checking is optional and can be enabled
    by providing a `stagnation_tol` and `number_past_iterations`.

    Parameters
    ----------
    target_value : float | None, optional
        The target value the optimization aims to reach. The optimization terminates when the value is within
        the given tolerance of the target value.
    tol : float
        The tolerance level for the target value, typically for convergence.
    stagnation_tol : float | None, optional
        The tolerance level for stagnation checking. The optimization terminates if the value does not
        change beyond this tolerance over the last `number_past_iterations` iterations. Default is None (no stagnation check).
    number_past_iterations : int | None, optional
        The number of past iterations to consider when checking for stagnation. Default is None (no stagnation check).

    Attributes
    ----------
    target_value : float
        The target value the optimization aims to reach.
    tol : float
        The tolerance level for the target value.
    stagnation_tol : float | None
        The tolerance level for stagnation checking, or None if no stagnation check is enabled.
    number_past_iterations : int | None
        The number of past iterations to consider for stagnation, or None if no stagnation check is enabled.
    values : list[float]
        The list of function values over iterations, used for stagnation checking.
    """

    def __init__(
        self, 
        target_value: float | None = None, 
        tol: float = 0.001, 
        stagnation_tol: float = 0.001, 
        number_past_iterations: int | None = None,
        noisy_oscillation_tol: float | None = None,
        number_past_iterations_oscillation: int | None = None,
        ):
        """
        Initialize the TerminationChecker instance.

        Parameters
        ----------
        target_value : float, optional
            The target value the optimization aims to reach.
        tol : float, optional
            The tolerance for convergence to the target value.
        stagnation_tol : float, optional
            The threshold for stagnation checking.
        number_past_iterations : int, optional
            The number of past iterations to track for stagnation checking.
        noisy_oscillation_tol : float, optional
            The threshold for a noisy non-convergence checking.
        number_past_iterations_oscillation : int, optional
            The number of past iterations to track for non-convergence checking.
        """
        self.target_value = target_value # Default to None (no convergence to target check)
        self.tol = tol
        self.stagnation_tol = stagnation_tol  # Default to None (no stagnation check)
        self.number_past_iterations = number_past_iterations
        self.noisy_oscillation_tol = noisy_oscillation_tol
        self.number_past_iterations_oscillation = number_past_iterations_oscillation
        self.values: list[float] = []

    def __call__(self, nfev: int, parameters: list[float], value: float, stepsize: float, accepted: bool) -> bool:
        """
        Check whether the optimization should terminate based on current optimization step.

        Parameters
        ----------
        nfev : int
            The number of function evaluations.
        parameters : list[float]
            The current parameters of the optimization.
        value : float
            The current value of the objective function.
        stepsize : float
            The current step size used in the optimization.
        accepted : bool
            Whether the current iteration was accepted.

        Returns
        -------
        bool
            True if termination criteria are met (either convergence or stagnation), False otherwise.
        """
        self.values.append(value)
        
        # Check if the target value is reached within tolerance
        if self.target_value is not None:
            if abs(self.target_value - value) < self.tol:
                logger.info(f"Reached target value within tolerance: {self.target_value}")
                print(f"Reached target value within tolerance: {self.target_value}")
                return True
        
        # If stagnation check is enabled, calculate the average of the last `number_past_iterations` values
        if self.stagnation_tol is not None and self.number_past_iterations is not None:
            stagnation_tol_std = self.stagnation_tol * 0.5
            if len(self.values) >= 0:
                last_values = self.values[-self.number_past_iterations:]
                last_few_av = np.mean(last_values)
                std_dev = np.std(last_values)
            
            # Check for stagnation over the last `number_past_iterations` values
            if len(self.values) > self.number_past_iterations and abs(value - last_few_av) < self.stagnation_tol:
                if std_dev < stagnation_tol_std:
                    logger.info("Stagnating Optimization. Average of last few iterations is " + 
                                f"{last_few_av}, standard deviation of last few values is: {std_dev}")
                    logger.info(f"Current value is {value}")
                    print("Stagnating Optimization. Average of last few iterations is " + 
                          f"{last_few_av}, standard deviation of last few values is: {std_dev}")
                    print(f"Current value is {value}")
                    return True
                
        # Check for non-converging noisy oscillations
        if self.noisy_oscillation_tol is not None and self.number_past_iterations_oscillation is not None:
            if len(self.values) > self.number_past_iterations_oscillation:
                recent = self.values[-self.number_past_iterations_oscillation:]
                std_dev = np.std(recent)
                mean_diff = recent[-1] - recent[0]  # Trend: positive = getting worse

                if std_dev > self.noisy_oscillation_tol:
                    logger.info("Detected noisy optimization with no convergence.")
                    logger.info(f"Standard deviation: {std_dev}, trend (Δ): {mean_diff}")
                    print("Detected noisy optimization with no convergence.")
                    print(f"Standard deviation: {std_dev}, trend (Δ): {mean_diff}")
                    return True        

        return False
    
    def reset(self) -> None:
        """
        Reset the termination checker by clearing the stored values.
        This can be used to start a new optimization run without creating a new instance.
        """
        logger.info("Resetting termination checker for new optimization run.")
        self.values.clear()
