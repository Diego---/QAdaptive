from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Sequence

import numpy as np

from qadaptive.outer.outer_loop import OuterStepResult
from qadaptive.training.history import TrainingRunRecord


@dataclass
class TrainingRunTrace:
    """
    Plot-ready representation of one inner-loop training run.

    This structure stores dense arrays aligned to a global plotting axis.
    It is built from a `TrainingRunRecord` and optional outer-loop metadata.

    Attributes
    ----------
    run_index : int
        Sequential index of the training run (0-based).
    outer_iteration : int
        Index of the associated outer-loop iteration.
    action : str
        Label describing the structural modification associated with the run.
    accepted : bool
        Whether the associated outer-loop proposal was accepted.
    start : int
        Global plotting index (inclusive) of the first point in this run.
    stop : int
        Global plotting index (exclusive) of the last point in this run.
    param_names : list[str]
        Ordered parameter names corresponding to the columns of `params`.
    params : np.ndarray
        Parameter values for the run, with shape ``(num_points, num_parameters)``.
        If `include_initial=True` in the builder, the first row is the initial point.
    values : np.ndarray
        Objective values for the run, with shape ``(num_points,)``.
        If `include_initial=True`, the first entry is the initial objective value
        or `missing_initial_value` if it was not stored.
    stepsizes : np.ndarray
        Step sizes for the run, with shape ``(num_points,)``.
        If `include_initial=True`, the first entry is `np.nan`.
    cost_after : float | None, optional
        Final objective value associated with the run.
    note : str | None, optional
        Optional annotation for the run.
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


def _iteration_param_matrix(record: TrainingRunRecord) -> np.ndarray:
    """
    Return the recorded iteration parameter vectors as a dense matrix.

    Parameters
    ----------
    record : TrainingRunRecord
        Training-run record from which to extract iteration parameters.

    Returns
    -------
    np.ndarray
        Array of shape ``(num_iterations, num_parameters)``.

    Raises
    ------
    ValueError
        If an iteration parameter vector does not match the recorded parameter count.
    """
    num_params = len(record.param_names)

    if len(record.iterations) == 0:
        return np.empty((0, num_params), dtype=float)

    rows = []
    for iteration in record.iterations:
        row = np.asarray(iteration.params, dtype=float)
        if row.shape != (num_params,):
            raise ValueError(
                f"Iteration parameter vector has shape {row.shape}, "
                f"expected ({num_params},)."
            )
        rows.append(row)

    return np.vstack(rows)


def _iteration_value_vector(record: TrainingRunRecord) -> np.ndarray:
    """
    Return the recorded per-iteration objective values as a vector.

    Parameters
    ----------
    record : TrainingRunRecord
        Training-run record from which to extract objective values.

    Returns
    -------
    np.ndarray
        Objective values with shape ``(num_iterations,)``.
    """
    return np.asarray([float(iteration.value) for iteration in record.iterations], dtype=float)


def _iteration_stepsize_vector(record: TrainingRunRecord) -> np.ndarray:
    """
    Return the recorded per-iteration step sizes as a vector.

    Parameters
    ----------
    record : TrainingRunRecord
        Training-run record from which to extract step sizes.

    Returns
    -------
    np.ndarray
        Step sizes with shape ``(num_iterations,)``.
    """
    return np.asarray([float(iteration.stepsize) for iteration in record.iterations], dtype=float)


def _resolve_outer_metadata(
    record: TrainingRunRecord,
    run_index: int,
    outer_step_history: Sequence[OuterStepResult] | None,
) -> tuple[int, str, bool, float | None, str | None]:
    """
    Resolve outer-loop metadata for one training run.

    Parameters
    ----------
    record : TrainingRunRecord
        Training-run record.
    run_index : int
        Index of the training run.
    outer_step_history : Sequence[OuterStepResult] | None
        Optional outer-loop history aligned with the training runs.

    Returns
    -------
    tuple[int, str, bool, float | None, str | None]
        Tuple ``(outer_iteration, action, accepted, cost_after, note)``.
    """
    if outer_step_history is not None:
        outer_result = outer_step_history[run_index]
        return (
            int(outer_result.iteration),
            str(outer_result.action),
            bool(outer_result.accepted),
            None if outer_result.cost_after is None else float(outer_result.cost_after),
            outer_result.note,
        )

    outer_iteration = getattr(record, "outer_iteration", run_index)
    action = getattr(record, "action", None)
    accepted = getattr(record, "accepted_outer_step", True)
    note = getattr(record, "note", None)
    cost_after = getattr(record, "final_value", None)

    if action is None:
        action = f"run_{run_index}"

    if accepted is None:
        accepted = True

    return (
        int(outer_iteration),
        str(action),
        bool(accepted),
        None if cost_after is None else float(cost_after),
        note,
    )


def build_training_run_traces(
    records: Sequence[TrainingRunRecord],
    outer_step_history: Sequence[OuterStepResult] | None = None,
    include_initial: bool = True,
    missing_initial_value: float = np.nan,
) -> list[TrainingRunTrace]:
    """
    Build plotting traces from stored training-run records.

    Parameters
    ----------
    records : Sequence[TrainingRunRecord]
        Stored training-run records, typically from
        `MutableAnsatzExperiment.training_run_history`.
    outer_step_history : Sequence[OuterStepResult] | None, optional
        Optional outer-step metadata aligned with `records`. If provided, it
        must have the same length as `records`.
    include_initial : bool, optional
        If True, prepend each run's initial point to the plotted trace.
        The first plotted x-position of the first run is then 0.
        If False, only recorded optimizer iterations are plotted, and the first
        plotted x-position of the first run is 1.
    missing_initial_value : float, optional
        Placeholder value used when `include_initial=True` but a run has no stored
        initial objective value. Defaults to `np.nan`.

    Returns
    -------
    list[TrainingRunTrace]
        Ordered list of plot-ready training traces.

    Raises
    ------
    ValueError
        If `outer_step_history` length does not match `records`, if initial
        points have incompatible sizes, or if a run would contain no plotted points.

    Notes
    -----
    This builder creates a global plotting axis by concatenating training runs.
    If `include_initial=True`, each run occupies one additional x-position for
    its initial point.
    """
    records = list(records)

    if len(records) == 0:
        return []

    if outer_step_history is not None and len(outer_step_history) != len(records):
        raise ValueError(
            f"Got {len(outer_step_history)} outer-step records for {len(records)} training runs."
        )

    traces: list[TrainingRunTrace] = []
    cursor = 0 if include_initial else 1

    for run_index, record in enumerate(records):
        param_names = list(record.param_names)
        num_params = len(param_names)

        initial_point = np.asarray(record.initial_point, dtype=float)
        if initial_point.shape != (num_params,):
            raise ValueError(
                f"Run {run_index} initial point has shape {initial_point.shape}, "
                f"expected ({num_params},)."
            )

        iter_params = _iteration_param_matrix(record)
        iter_values = _iteration_value_vector(record)
        iter_stepsizes = _iteration_stepsize_vector(record)

        if include_initial:
            if num_params == 0:
                params = np.empty((1 + len(record.iterations), 0), dtype=float)
            else:
                params = np.vstack([initial_point.reshape(1, -1), iter_params])

            initial_value = (
                float(record.initial_value)
                if record.initial_value is not None
                else float(missing_initial_value)
            )
            values = np.concatenate(([initial_value], iter_values))
            stepsizes = np.concatenate(([np.nan], iter_stepsizes))
        else:
            params = iter_params
            values = iter_values
            stepsizes = iter_stepsizes

        if values.size == 0:
            raise ValueError(
                f"Run {run_index} has no plotted points. "
                "Use include_initial=True or record at least one accepted iteration."
            )

        start = cursor
        stop = start + len(values)

        outer_iteration, action, accepted, cost_after, note = _resolve_outer_metadata(
            record=record,
            run_index=run_index,
            outer_step_history=outer_step_history,
        )

        traces.append(
            TrainingRunTrace(
                run_index=run_index,
                outer_iteration=outer_iteration,
                action=action,
                accepted=accepted,
                start=start,
                stop=stop,
                param_names=param_names,
                params=params,
                values=values,
                stepsizes=stepsizes,
                cost_after=cost_after,
                note=note,
            )
        )

        cursor = stop

    return traces
