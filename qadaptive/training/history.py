from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np


@dataclass
class IterationRecord:
    """
    Record of one accepted inner-loop optimization step.
    """

    iteration: int
    params: np.ndarray
    value: float
    stepsize: float
    accepted: bool
    gradient: np.ndarray | None = None


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
    final_value: float | None = None
    outer_iteration: int | None = None
    action: str | None = None
    accepted_outer_step: bool | None = None
    note: str | None = None
