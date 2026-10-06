from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np


@dataclass
class IterationRecord:
    """
    Record of one accepted inner-loop optimization step.
    """

    iteration: int
    nfev: int
    params: np.ndarray
    optimizer_estimate: float
    stepsize: float
    accepted: bool
    evaluation_value: float | None = None
    gradient: np.ndarray | None = None
    schedule_steps_used: dict[str, int] | None = None
    learning_rates: dict[str, float] | None = None
    perturbations: dict[str, float] | None = None
    extra_value: float | None = None
    extra_std: float | None = None


    @property
    def value(self) -> float:
        """Return the explicit evaluation when present, else the optimizer estimate."""
        if self.evaluation_value is not None:
            return float(self.evaluation_value)
        return float(self.optimizer_estimate)


@dataclass
class TrainingRunRecord:
    """
    Record of one inner-loop training run for a fixed ansatz structure.
    """

    run_index: int
    param_names: list[str]
    initial_point: np.ndarray
    initial_value: float | None
    iterations: list[IterationRecord] = field(default_factory=list)
    final_params: np.ndarray | None = None
    final_value: float | None = None
    outer_iteration: int | None = None
    parameter_birth_outer_iterations: dict[str, int] | None = None
    action: str | None = None
    accepted_outer_step: bool | None = None
    note: str | None = None
