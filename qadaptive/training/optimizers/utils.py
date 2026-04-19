from typing import Callable

def create_callback_args(
    json_file_path: str | None = None,
    store_data: bool = False,
    extra_eval_freq: int | None = None,
    cost_extra: Callable | None = None,
    plot: bool = True,
    use_epoch: bool = False
) -> dict:
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
    dict
        A dictionary with the arguments for the callback builder.
    """
    counts = []
    values = []
    params = []
    stepsize = []
    values_extra = [] if extra_eval_freq is not None else None
    
    args = {
        "counts": counts,
        "values": values,
        "params": params,
        "stepsize": stepsize,
        "json_file_path": json_file_path,
        "store_data": store_data,
        "extra_eval_freq": extra_eval_freq,
        "cost_extra": cost_extra,
        "values_extra": values_extra,
        "plot": plot,
        "use_epoch": use_epoch
    }
    
    return args
