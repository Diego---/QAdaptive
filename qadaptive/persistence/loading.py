from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import qiskit.qpy as qpy
from qiskit.circuit import QuantumCircuit


@dataclass
class LoadedExperimentHistory:
    directory: Path
    manifest: dict[str, Any]
    ansatz: QuantumCircuit
    current_state: dict[str, Any]
    outer_step_history: list[dict[str, Any]]
    parameter_memory_history: list[dict[str, Any]]
    trial_ansatz_history: list[dict[str, Any]]
    accepted_ansatz_history: list[dict[str, Any]]
    training_run_history: list[dict[str, Any]]
    gradient_history: dict[str, Any] | None
    result_history: list[dict[str, Any]] | None

    @property
    def last_params(self) -> np.ndarray | None:
        values = self.current_state["last_params"]
        return None if values is None else np.asarray(values, dtype=float)

    @property
    def last_cost(self) -> float | None:
        return self.current_state["last_cost"]


def load_experiment_history(
    directory: str | Path,
) -> LoadedExperimentHistory:
    """Load the saved current circuit, parameter values, and JSON histories."""
    directory = Path(directory)

    with (directory / "manifest.json").open("r", encoding="utf-8") as file:
        manifest = json.load(file)

    if not isinstance(manifest, dict):
        raise ValueError("Archive manifest must contain a JSON object.")

    schema_version = manifest.get("schema_version")
    if type(schema_version) is not int or schema_version != 1:
        raise ValueError(
            f"Unsupported archive schema version: {schema_version!r}"
        )

    json_names = (
        "current_state",
        "outer_step_history",
        "parameter_memory_history",
        "trial_ansatz_history",
        "accepted_ansatz_history",
        "training_run_history",
        "gradient_history",
        "result_history",
    )
    payloads = {}
    for name in json_names:
        path = directory / manifest["files"][name]
        with path.open("r", encoding="utf-8") as file:
            payloads[name] = json.load(file)

    circuit_path = directory / manifest["files"]["current_ansatz_qpy"]
    with circuit_path.open("rb") as file:
        circuits = qpy.load(file)

    if len(circuits) != 1:
        raise ValueError("Current ansatz QPY must contain exactly one circuit.")

    history = LoadedExperimentHistory(
        directory=directory,
        manifest=manifest,
        ansatz=circuits[0],
        **payloads,
    )

    params = history.last_params
    if params is not None and params.shape != (history.ansatz.num_parameters,):
        raise ValueError(
            "Saved parameter vector does not match the current circuit."
        )

    return history