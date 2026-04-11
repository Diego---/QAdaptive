from __future__ import annotations
from typing import TYPE_CHECKING

from dataclasses import dataclass
from collections.abc import Sequence

import numpy as np

if TYPE_CHECKING:
    from qadaptive.outer.mutable_ansatz_experiment import MutableAnsatzExperiment

@dataclass
class TrainingRunTrace:
    """
    Container for a single inner-loop training segment within an outer-loop run.

    This structure represents the data produced during one training phase
    (i.e., between two structural modifications of the ansatz). It provides
    a consistent interface for plotting and analysis across multiple outer steps.
    
    Attributes
    ----------
    run_index : int
        Sequential index of the training run (0-based).
    outer_iteration : int
        Index of the outer-loop iteration associated with this training segment.
    action : str
        Description of the structural modification applied before this training
        segment (e.g., "uniform_growth", "prune_sweep").
    accepted : bool
        Whether the structural modification associated with this segment was
        accepted according to the outer-loop acceptance criterion.
    start : int
        Global index (inclusive) of the first inner-loop iteration in this segment.
    stop : int
        Global index (exclusive) of the last inner-loop iteration in this segment.
    param_names : list[str]
        List of parameter names active during this training segment.
        The order matches the columns of `params`.
    params : np.ndarray
        Parameter values recorded during the training segment, with shape
        ``(num_iterations, num_parameters)``.
    values : np.ndarray
        Objective values recorded during the training segment, with shape
        ``(num_iterations,)``.
    stepsizes : np.ndarray
        Step sizes recorded during the training segment, with shape
        ``(num_iterations,)``.
    cost_after : float | None, optional
        Final objective value associated with this segment, typically taken
        from the corresponding outer-step result.
    note : str | None, optional
        Optional note associated with the segment, for example information
        about automatic acceptance or rollback.

    Notes
    -----
    This object is intended as the canonical intermediate representation for
    plotting utilities. Plotting functions should consume `TrainingRunTrace`
    objects rather than raw callback lists whenever possible.
    """
    run_index: int
    outer_iteration: int
    action: str
    accepted: bool
    start: int
    stop: int
    param_names: list[str]
    params: np.ndarray
    values: np.ndarray
    stepsizes: np.ndarray
    cost_after: float | None = None
    note: str | None = None


def _deduplicate_initial_training_record(parameter_memory_history: Sequence) -> list:
    """
    Remove duplicated initial-training records from parameter-memory history.

    Parameters
    ----------
    parameter_memory_history : Sequence
        Sequence of parameter-memory history records, typically
        `experiment.parameter_memory_history`.

    Returns
    -------
    list
        Cleaned list of records in which the duplicated
        ``"Initial training before first plan"`` entry, if present, appears
        only once.

    Notes
    -----
    Some experiment versions may append the initial-training parameter-memory
    record twice. This helper removes the second occurrence so that the number
    of parameter-memory records matches the number of reconstructed training
    segments.
    """
    cleaned = []
    seen_initial = False

    for record in parameter_memory_history:
        is_initial = (
            getattr(record, "outer_iteration", None) == 0
            and getattr(record, "action", None) == "Initial training before first plan"
        )

        if is_initial and seen_initial:
            continue

        cleaned.append(record)

        if is_initial:
            seen_initial = True

    return cleaned


def _validate_trace_inputs(
    outer_step_history: Sequence,
    parameter_memory_history: Sequence,
    params_history: Sequence[Sequence[float]] | Sequence[np.ndarray],
    values_history: Sequence[float],
    stepsize_history: Sequence[float],
    train_iterations: int,
) -> None:
    """
    Validate raw inputs used to reconstruct training traces.

    Parameters
    ----------
    outer_step_history : Sequence
        Sequence of outer-step result records.
    parameter_memory_history : Sequence
        Sequence of parameter-memory history records.
    params_history : Sequence[Sequence[float]] or Sequence[np.ndarray]
        Flat callback history of parameter vectors across all inner-loop runs.
    values_history : Sequence[float]
        Flat callback history of objective values across all inner-loop runs.
    stepsize_history : Sequence[float]
        Flat callback history of step sizes across all inner-loop runs.
    train_iterations : int
        Number of inner-loop iterations expected per training run.

    Raises
    ------
    ValueError
        If the input histories are inconsistent in length or shape.
    """
    if train_iterations <= 0:
        raise ValueError("`train_iterations` must be positive.")

    if len(outer_step_history) != len(parameter_memory_history):
        raise ValueError(
            f"Mismatch between outer_step_history ({len(outer_step_history)}) "
            f"and parameter_memory_history ({len(parameter_memory_history)})."
        )

    expected_points = len(outer_step_history) * train_iterations

    if len(params_history) != len(values_history) or len(values_history) != len(stepsize_history):
        raise ValueError(
            "Callback histories must have the same length for parameters, "
            "objective values, and step sizes."
        )

    if len(params_history) < expected_points:
        raise ValueError(
            f"Callback history is too short: got {len(params_history)} entries, "
            f"expected at least {expected_points} from "
            f"{len(outer_step_history)} runs and {train_iterations} iterations per run."
        )


def build_training_run_traces(
    experiment: "MutableAnsatzExperiment",
    params_history: Sequence[Sequence[float]] | Sequence[np.ndarray],
    values_history: Sequence[float],
    stepsize_history: Sequence[float],
    train_iterations: int,
    deduplicate_initial_record: bool = True,
) -> list[TrainingRunTrace]:
    """
    Reconstruct structured training traces from flat callback histories.

    This function converts the raw histories accumulated by a live callback
    into a list of `TrainingRunTrace` objects, one for each inner-loop
    training segment in the experiment. Each segment is associated with the
    corresponding outer-step metadata and parameter names active during that
    run.

    Parameters
    ----------
    experiment : MutableAnsatzExperiment
        Experiment object containing `outer_step_history` and
        `parameter_memory_history`.
    params_history : Sequence[Sequence[float]] or Sequence[np.ndarray]
        Flat callback history of parameter vectors across all inner-loop runs.
        Each entry corresponds to one recorded optimizer step.
    values_history : Sequence[float]
        Flat callback history of objective values across all inner-loop runs.
    stepsize_history : Sequence[float]
        Flat callback history of step sizes across all inner-loop runs.
    train_iterations : int
        Number of inner-loop iterations per training run. This is used to slice
        the flat callback histories into per-run segments.
    deduplicate_initial_record : bool, optional
        Whether to remove a duplicated initial-training parameter-memory record
        before reconstruction. Default is True.

    Returns
    -------
    list[TrainingRunTrace]
        Ordered list of reconstructed training segments.

    Raises
    ------
    ValueError
        If the callback histories are inconsistent with the number of outer
        steps, with `train_iterations`, or with the recorded parameter names.

    Notes
    -----
    The reconstruction assumes that:
    - the ansatz structure is fixed during each inner-loop training run,
    - parameter order within a run is the order stored in the corresponding
      parameter-memory record,
    - callback histories are concatenated in chronological order across runs.

    The returned traces are intended to serve as the main input to plotting
    utilities such as:
    - `plot_cost_with_outer_boundaries`
    - `plot_parameter_lifelines`
    - `plot_parameter_heatmap`
    """
    outer_step_history = list(experiment.outer_step_history)
    parameter_memory_history = list(experiment.parameter_memory_history)

    if deduplicate_initial_record:
        parameter_memory_history = _deduplicate_initial_training_record(
            parameter_memory_history
        )

    _validate_trace_inputs(
        outer_step_history=outer_step_history,
        parameter_memory_history=parameter_memory_history,
        params_history=params_history,
        values_history=values_history,
        stepsize_history=stepsize_history,
        train_iterations=train_iterations,
    )

    traces: list[TrainingRunTrace] = []

    for run_index, (outer_result, memory_record) in enumerate(
        zip(outer_step_history, parameter_memory_history)
    ):
        start = run_index * train_iterations
        stop = start + train_iterations

        param_names = list(memory_record.values.parameter_names)
        param_matrix = np.asarray(params_history[start:stop], dtype=float)
        value_vector = np.asarray(values_history[start:stop], dtype=float)
        stepsize_vector = np.asarray(stepsize_history[start:stop], dtype=float)

        if param_matrix.ndim != 2:
            raise ValueError(
                f"Run {run_index} parameter history must be a 2D array, "
                f"got shape {param_matrix.shape}."
            )

        if param_matrix.shape[0] != train_iterations:
            raise ValueError(
                f"Run {run_index} contains {param_matrix.shape[0]} recorded iterations, "
                f"expected {train_iterations}."
            )

        if param_matrix.shape[1] != len(param_names):
            raise ValueError(
                f"Run {run_index} has {param_matrix.shape[1]} parameter columns in the "
                f"callback history, but {len(param_names)} parameter names were recorded "
                "in parameter_memory_history."
            )

        traces.append(
            TrainingRunTrace(
                run_index=run_index,
                outer_iteration=outer_result.iteration,
                action=outer_result.action,
                accepted=outer_result.accepted,
                start=start,
                stop=stop,
                param_names=param_names,
                params=param_matrix,
                values=value_vector,
                stepsizes=stepsize_vector,
                cost_after=outer_result.cost_after,
                note=outer_result.note,
            )
        )

    return traces
