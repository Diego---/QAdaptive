from __future__ import annotations

from typing import Any, TYPE_CHECKING

from qadaptive.core.mutation import get_two_qubit_gate_indices

if TYPE_CHECKING:
    from qadaptive.outer.mutable_ansatz_experiment import MutableAnsatzExperiment
    from qadaptive.persistence.loading import LoadedExperimentHistory


def _field(record: Any, name: str) -> Any:
    """Read a field from a live record or its saved dictionary."""
    if isinstance(record, dict):
        return record.get(name)
    return getattr(record, name, None)


def build_experiment_summary(
    experiment: MutableAnsatzExperiment | LoadedExperimentHistory,
) -> dict[str, int | float | None]:
    """Build summary metrics from existing experiment records."""
    runs = experiment.training_run_history
    trials = experiment.trial_ansatz_history

    accepted = sum(bool(trial["accepted"]) for trial in trials)
    completed = sum(_field(run, "final_value") is not None for run in runs)

    optimizer_nfev = None
    if experiment.result_history is not None:
        counts = [_field(result, "nfev") for result in experiment.result_history]
        if all(count is not None for count in counts):
            optimizer_nfev = sum(int(count) for count in counts)

    # The trainer initially holds 0.0, even before any training.
    cost = experiment.last_cost if completed else None

    return {
        "num_outer_proposals": len(trials),
        "num_accepted_proposals": accepted,
        "num_rejected_proposals": len(trials) - accepted,
        "num_training_runs": len(runs),
        "num_completed_training_runs": completed,
        "num_recorded_inner_steps": sum(
            len(_field(run, "iterations")) for run in runs
        ),
        "last_cost": None if cost is None else float(cost),
        "num_parameters": int(experiment.ansatz.num_parameters),
        "num_two_qubit_instructions": len(
            get_two_qubit_gate_indices(experiment.ansatz)
        ),
        "optimizer_nfev": optimizer_nfev,
    }


def print_experiment_summary(
    experiment: MutableAnsatzExperiment | LoadedExperimentHistory,
) -> None:
    """Print the same compact summary for live and loaded experiments."""
    summary = build_experiment_summary(experiment)

    cost = summary["last_cost"]
    cost_text = "N/A" if cost is None else f"{cost:.10f}"

    nfev = summary["optimizer_nfev"]
    nfev_text = "N/A" if nfev is None else str(nfev)

    print("QAdaptive experiment summary")
    print(f"Current cost: {cost_text}")
    print(
        f"Structural proposals: {summary['num_outer_proposals']} "
        f"(accepted: {summary['num_accepted_proposals']}, "
        f"rejected: {summary['num_rejected_proposals']})"
    )
    print(
        f"Inner runs: {summary['num_training_runs']} "
        f"(completed: {summary['num_completed_training_runs']})"
    )
    print(f"Recorded inner steps: {summary['num_recorded_inner_steps']}")
    print(f"Current parameters: {summary['num_parameters']}")
    print(f"Two-qubit instructions: {summary['num_two_qubit_instructions']}")
    print(f"Optimizer nfev (completed results): {nfev_text}")
