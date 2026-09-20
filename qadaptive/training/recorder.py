from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from .history import IterationRecord, TrainingRunRecord


def _nominal_and_std(value: Any) -> tuple[float, float | None]:
    """Convert a scalar, including an uncertainty scalar, to value and error."""
    nominal = getattr(value, "nominal_value", value)
    std = getattr(value, "std_dev", None)
    return float(nominal), None if std is None else float(std)


class InnerLoopRecorder:
    """Record all inner-loop runs belonging to one adaptive experiment.

    The recorder is created once and passed to :class:`InnerLoopTrainer`. The
    trainer controls run boundaries and records every accepted optimizer step;
    the outer loop later attaches proposal acceptance metadata to the same run.

    Parameters
    ----------
    record_initial_value : bool, optional
        Whether the trainer should evaluate and store the objective at each
        run's initial point. This adds one objective evaluation per run.
    record_gradients : bool, optional
        Whether gradient vectors should be retained. Defaults to ``True``.
    extra_objective : callable or None, optional
        Additional objective evaluated periodically at the recorded parameters.
        It must accept one NumPy parameter vector.
    extra_evaluation_frequency : int or None, optional
        Evaluate ``extra_objective`` every N recorded optimizer steps.
    jsonl_path : str, Path, or None, optional
        Append lifecycle events to this JSON Lines file as the run progresses.
        This is useful for preserving partial hardware runs after interruption.
    """

    def __init__(
        self,
        *,
        record_initial_value: bool = False,
        record_gradients: bool = True,
        extra_objective: Callable[[np.ndarray], Any] | None = None,
        extra_evaluation_frequency: int | None = None,
        jsonl_path: str | Path | None = None,
    ) -> None:
        if extra_evaluation_frequency is not None:
            if extra_evaluation_frequency <= 0:
                raise ValueError("`extra_evaluation_frequency` must be positive.")
            if extra_objective is None:
                raise ValueError(
                    "`extra_objective` is required when "
                    "`extra_evaluation_frequency` is set."
                )

        self.record_initial_value = bool(record_initial_value)
        self.record_gradients = bool(record_gradients)
        self.extra_objective = extra_objective
        self.extra_evaluation_frequency = extra_evaluation_frequency
        self.jsonl_path = None if jsonl_path is None else Path(jsonl_path)

        self.runs: list[TrainingRunRecord] = []
        self._active_run: TrainingRunRecord | None = None

    @property
    def active_run(self) -> TrainingRunRecord | None:
        """Return the run currently being recorded, if any."""
        return self._active_run

    @property
    def last_run(self) -> TrainingRunRecord | None:
        """Return the most recently started run, if any."""
        return None if not self.runs else self.runs[-1]

    def __len__(self) -> int:
        """Return the number of recorded training runs."""
        return len(self.runs)

    def start_run(
        self,
        *,
        param_names: Sequence[str],
        initial_point: Sequence[float] | np.ndarray,
        initial_value: float | None = None,
        outer_iteration: int | None = None,
        action: str | None = None,
        note: str | None = None,
    ) -> TrainingRunRecord:
        """Open and return a new inner-loop run."""
        if self._active_run is not None:
            raise RuntimeError("Cannot start a run while another run is active.")

        names = [str(name) for name in param_names]
        point = np.asarray(initial_point, dtype=float).copy()
        if point.shape != (len(names),):
            raise ValueError(
                f"Initial point has shape {point.shape}, expected ({len(names)},)."
            )

        record = TrainingRunRecord(
            run_index=len(self.runs),
            param_names=names,
            initial_point=point,
            initial_value=None if initial_value is None else float(initial_value),
            outer_iteration=outer_iteration,
            action=action,
            note=note,
        )
        self.runs.append(record)
        self._active_run = record
        self._write_jsonl("run_started", self._serialize_run(record))
        return record

    def __call__(
        self,
        *,
        iteration: int,
        nfev: int,
        params: Sequence[float] | np.ndarray,
        value: float,
        stepsize: float,
        accepted: bool,
        gradient: Sequence[float] | np.ndarray | None = None,
    ) -> IterationRecord:
        """Record one inner-loop optimizer step in the active run."""
        if self._active_run is None:
            raise RuntimeError("Cannot record an iteration without an active run.")

        point = np.asarray(params, dtype=float).copy()
        expected_shape = (len(self._active_run.param_names),)
        if point.shape != expected_shape:
            raise ValueError(
                f"Parameter vector has shape {point.shape}, expected {expected_shape}."
            )

        gradient_array = None
        if self.record_gradients and gradient is not None:
            gradient_array = np.asarray(gradient, dtype=float).copy()
            if gradient_array.shape != expected_shape:
                raise ValueError(
                    f"Gradient has shape {gradient_array.shape}, expected {expected_shape}."
                )

        extra_value = None
        extra_std = None
        step_number = len(self._active_run.iterations) + 1
        if (
            self.extra_evaluation_frequency is not None
            and step_number % self.extra_evaluation_frequency == 0
        ):
            extra_value, extra_std = _nominal_and_std(self.extra_objective(point.copy()))

        record = IterationRecord(
            iteration=int(iteration),
            nfev=int(nfev),
            params=point,
            value=float(value),
            stepsize=float(stepsize),
            accepted=bool(accepted),
            gradient=gradient_array,
            extra_value=extra_value,
            extra_std=extra_std,
        )
        self._active_run.iterations.append(record)
        self._write_jsonl(
            "iteration_recorded",
            {
                "run_index": self._active_run.run_index,
                "iteration": self._serialize_iteration(record),
            },
        )
        return record

    def finish_run(
        self,
        *,
        final_params: Sequence[float] | np.ndarray,
        final_value: float,
    ) -> TrainingRunRecord:
        """Close the active run and store its final optimizer result."""
        if self._active_run is None:
            raise RuntimeError("Cannot finish a run when no run is active.")

        point = np.asarray(final_params, dtype=float).copy()
        expected_shape = (len(self._active_run.param_names),)
        if point.shape != expected_shape:
            raise ValueError(
                f"Final parameter vector has shape {point.shape}, expected {expected_shape}."
            )

        record = self._active_run
        record.final_params = point
        record.final_value = float(final_value)
        self._active_run = None
        self._write_jsonl("run_finished", self._serialize_run(record))
        return record

    def abort_run(self, note: str | None = None) -> TrainingRunRecord | None:
        """Close an incomplete run while retaining all data recorded so far."""
        if self._active_run is None:
            return None

        record = self._active_run
        if note:
            record.note = note if record.note is None else f"{record.note} {note}"
        self._active_run = None
        self._write_jsonl("run_aborted", self._serialize_run(record))
        return record

    def set_outer_result(
        self,
        *,
        accepted: bool,
        note: str | None = None,
        run_index: int | None = None,
    ) -> TrainingRunRecord:
        """Attach the outer-loop decision to a completed training run."""
        if not self.runs:
            raise RuntimeError("No training run is available for outer-loop metadata.")

        index = len(self.runs) - 1 if run_index is None else int(run_index)
        record = self.runs[index]
        record.accepted_outer_step = bool(accepted)
        if note is not None:
            record.note = note
        self._write_jsonl(
            "outer_result_recorded",
            {
                "run_index": record.run_index,
                "accepted_outer_step": record.accepted_outer_step,
                "note": record.note,
            },
        )
        return record

    def clear(self) -> None:
        """Discard all recorded runs while preserving recorder configuration."""
        if self._active_run is not None:
            raise RuntimeError("Cannot clear the recorder while a run is active.")
        self.runs.clear()

    @property
    def values(self) -> list[float]:
        """Return all per-iteration objective values, flattened by run."""
        return [iteration.value for run in self.runs for iteration in run.iterations]

    @property
    def params(self) -> list[np.ndarray]:
        """Return copies of all per-iteration parameter vectors."""
        return [iteration.params.copy() for run in self.runs for iteration in run.iterations]

    @property
    def nfevs(self) -> list[int]:
        """Return all per-iteration objective evaluation counts."""
        return [iteration.nfev for run in self.runs for iteration in run.iterations]

    @property
    def stepsizes(self) -> list[float]:
        """Return all per-iteration optimizer step sizes."""
        return [iteration.stepsize for run in self.runs for iteration in run.iterations]

    @property
    def gradients(self) -> list[np.ndarray | None]:
        """Return copies of all recorded gradients, flattened by run."""
        return [
            None if iteration.gradient is None else iteration.gradient.copy()
            for run in self.runs
            for iteration in run.iterations
        ]

    @property
    def gradient_norms(self) -> list[float | None]:
        """Return the norm of every recorded gradient."""
        return [
            None if gradient is None else float(np.linalg.norm(gradient))
            for gradient in self.gradients
        ]

    @property
    def gradient_history(self) -> dict[int, list[np.ndarray]]:
        """Return gradients grouped by run index."""
        return {
            run.run_index: [
                iteration.gradient.copy()
                for iteration in run.iterations
                if iteration.gradient is not None
            ]
            for run in self.runs
        }

    def traces(
        self,
        *,
        include_initial: bool = True,
        missing_initial_value: float = np.nan,
    ):
        """Build plot-ready traces for all recorded runs."""
        from qadaptive.utils.plotting.traces import build_training_run_traces

        return build_training_run_traces(
            self.runs,
            include_initial=include_initial,
            missing_initial_value=missing_initial_value,
        )

    def plot_objective(self, *, include_initial: bool = True, **kwargs):
        """Plot the objective history across all inner-loop runs."""
        from qadaptive.utils.plotting.objective_plots import (
            plot_cost_with_outer_boundaries,
        )

        return plot_cost_with_outer_boundaries(
            self.traces(include_initial=include_initial),
            **kwargs,
        )

    def plot_parameters(
        self,
        *,
        parameters: list[str] | None = None,
        include_initial: bool = True,
        **kwargs,
    ):
        """Plot named parameter trajectories across all inner-loop runs."""
        from qadaptive.utils.plotting.parameter_plots import plot_parameter_lifelines

        return plot_parameter_lifelines(
            self.traces(include_initial=include_initial),
            parameters=parameters,
            **kwargs,
        )

    def plot_parameter_heatmap(
        self,
        *,
        parameters: list[str] | None = None,
        include_initial: bool = True,
        **kwargs,
    ):
        """Plot parameter activity as a heatmap across all inner-loop runs."""
        from qadaptive.utils.plotting.parameter_plots import plot_parameter_heatmap

        return plot_parameter_heatmap(
            self.traces(include_initial=include_initial),
            parameters=parameters,
            **kwargs,
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation of the recorder."""
        return {
            "schema_version": 1,
            "record_initial_value": self.record_initial_value,
            "record_gradients": self.record_gradients,
            "extra_evaluation_frequency": self.extra_evaluation_frequency,
            "runs": [self._serialize_run(run) for run in self.runs],
        }

    def save(self, path: str | Path) -> Path:
        """Save the complete recorder state as formatted JSON."""
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as file:
            json.dump(self.to_dict(), file, indent=2)
        return output_path

    @staticmethod
    def _serialize_iteration(record: IterationRecord) -> dict[str, Any]:
        return {
            "iteration": int(record.iteration),
            "nfev": int(record.nfev),
            "params": np.asarray(record.params, dtype=float).tolist(),
            "value": float(record.value),
            "stepsize": float(record.stepsize),
            "accepted": bool(record.accepted),
            "gradient": None
            if record.gradient is None
            else np.asarray(record.gradient, dtype=float).tolist(),
            "extra_value": record.extra_value,
            "extra_std": record.extra_std,
        }

    @classmethod
    def _serialize_run(cls, record: TrainingRunRecord) -> dict[str, Any]:
        return {
            "run_index": int(record.run_index),
            "param_names": list(record.param_names),
            "initial_point": np.asarray(record.initial_point, dtype=float).tolist(),
            "initial_value": record.initial_value,
            "iterations": [cls._serialize_iteration(item) for item in record.iterations],
            "final_params": None
            if record.final_params is None
            else np.asarray(record.final_params, dtype=float).tolist(),
            "final_value": record.final_value,
            "outer_iteration": None
            if record.outer_iteration is None
            else int(record.outer_iteration),
            "action": record.action,
            "accepted_outer_step": record.accepted_outer_step,
            "note": record.note,
        }

    def _write_jsonl(self, event: str, payload: dict[str, Any]) -> None:
        if self.jsonl_path is None:
            return

        self.jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        with self.jsonl_path.open("a", encoding="utf-8") as file:
            json.dump({"event": event, **payload}, file)
            file.write("\n")
