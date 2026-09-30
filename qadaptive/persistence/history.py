from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import qiskit.qpy as qpy
from qiskit.circuit import QuantumCircuit

if TYPE_CHECKING:
    from qadaptive.outer.mutable_ansatz_experiment import MutableAnsatzExperiment


def save_experiment_history(
    experiment: MutableAnsatzExperiment,
    directory: str | Path,
) -> Path:
    """
    Save experiment history and current state to disk.

    Parameters
    ----------
    experiment : MutableAnsatzExperiment
        Experiment whose data will be saved.
    directory : str | Path
        Directory where all history files should be written.

    Returns
    -------
    Path
        Path to the directory where the history was saved.

    Notes
    -----
    This method stores:
    - current ansatz as QPY
    - accepted ansatz history as QPY files plus JSON metadata
    - outer-step history
    - parameter-memory history
    - training-run history
    - gradient history
    - optimizer-result summaries
    - current live experiment state

    Rejected outer-loop proposals are saved only as metadata if they were not
    recorded as explicit ansatz objects elsewhere.
    """
    save_dir = Path(directory)
    save_dir.mkdir(parents=True, exist_ok=True)

    circuits_dir = save_dir / "circuits"
    trial_dir = circuits_dir / "trial_ansatz_history"
    accepted_dir = circuits_dir / "accepted_ansatz_history"
    circuits_dir.mkdir(parents=True, exist_ok=True)
    trial_dir.mkdir(parents=True, exist_ok=True)
    accepted_dir.mkdir(parents=True, exist_ok=True)

    def _slugify(text: str) -> str:
        text = re.sub(r"[^A-Za-z0-9._-]+", "_", text.strip())
        return text.strip("_") or "record"

    def _jsonable(obj):
        """Recursively convert objects to JSON-serializable Python types."""
        if obj is None or isinstance(obj, (bool, int, float, str)):
            return obj

        if isinstance(obj, Path):
            return str(obj)

        if isinstance(obj, np.ndarray):
            return obj.tolist()

        if isinstance(obj, (np.integer, np.floating)):
            return obj.item()

        if isinstance(obj, dict):
            return {str(k): _jsonable(v) for k, v in obj.items()}

        if isinstance(obj, (list, tuple)):
            return [_jsonable(v) for v in obj]

        if isinstance(obj, set):
            return sorted(_jsonable(v) for v in obj)

        # Qiskit Parameters
        if hasattr(obj, "name") and obj.__class__.__name__ == "Parameter":
            return obj.name

        # Generic fallback for simple record-like objects
        if hasattr(obj, "__dict__"):
            return {k: _jsonable(v) for k, v in vars(obj).items()}

        # Last resort
        return str(obj)

    def _write_json(path: Path, payload: dict | list) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    def _save_circuit(path: Path, circuit: QuantumCircuit) -> None:
        with open(path, "wb") as f:
            qpy.dump(circuit, f)

    def _circuit_summary(circuit: QuantumCircuit) -> dict:
        try:
            count_ops = {str(k): int(v) for k, v in circuit.count_ops().items()}
        except Exception:
            count_ops = {}

        try:
            depth = int(circuit.depth())
        except Exception:
            depth = None

        return {
            "num_qubits": int(circuit.num_qubits),
            "num_clbits": int(circuit.num_clbits),
            "num_parameters": int(circuit.num_parameters),
            "size": int(circuit.size()),
            "depth": depth,
            "count_ops": count_ops,
        }

    def _serialize_parameter_memory_cache(cache) -> dict:
        return {
            "parameter_names": list(cache.parameter_names),
            "dictionary": {str(k): float(v) for k, v in cache.dictionary.items()},
        }

    def _serialize_parameter_memory_record(record) -> dict:
        return {
            "outer_iteration": int(record.outer_iteration),
            "action": record.action,
            "accepted": bool(record.accepted),
            "cost": None if record.cost is None else float(record.cost),
            "values": _serialize_parameter_memory_cache(record.values),
        }

    def _serialize_outer_step_result(result) -> dict:
        return {
            "iteration": int(result.iteration),
            "action": result.action,
            "accepted": bool(result.accepted),
            "cost_before": None if result.cost_before is None else float(result.cost_before),
            "cost_after": None if result.cost_after is None else float(result.cost_after),
            "delta_cost": None if result.delta_cost is None else float(result.delta_cost),
            "num_parameters_before": int(result.num_parameters_before),
            "num_parameters_after": int(result.num_parameters_after),
            "num_two_qubit_before": int(result.num_two_qubit_before),
            "num_two_qubit_after": int(result.num_two_qubit_after),
            "note": result.note,
        }

    def _serialize_training_run_record(record) -> dict:
        return {
            "run_index": int(record.run_index),
            "param_names": list(record.param_names),
            "initial_point": np.asarray(record.initial_point, dtype=float).tolist(),
            "initial_value": None if record.initial_value is None else float(record.initial_value),
            "final_params": None if record.final_params is None else np.asarray(record.final_params, dtype=float).tolist(),
            "final_value": None if record.final_value is None else float(record.final_value),
            "outer_iteration": record.outer_iteration,
            "action": record.action,
            "accepted_outer_step": record.accepted_outer_step,
            "note": record.note,
            "iterations": [
                {
                    "iteration": int(it.iteration),
                    "nfev": int(it.nfev),
                    "params": np.asarray(it.params, dtype=float).tolist(),
                    "value": float(it.value),
                    "stepsize": float(it.stepsize),
                    "accepted": bool(it.accepted),
                    "gradient": None if it.gradient is None else np.asarray(it.gradient, dtype=float).tolist(),
                    "extra_value": it.extra_value,
                    "extra_std": it.extra_std,
                }
                for it in record.iterations
            ],
        }

    def _serialize_optimizer_result(result) -> dict:
        payload = {}

        if hasattr(result, "x") and result.x is not None:
            payload["x"] = np.asarray(result.x, dtype=float).tolist()
        if hasattr(result, "fun") and result.fun is not None:
            payload["fun"] = float(result.fun)
        if hasattr(result, "nit") and result.nit is not None:
            payload["nit"] = int(result.nit)
        if hasattr(result, "nfev") and result.nfev is not None:
            payload["nfev"] = int(result.nfev)

        return payload

    timestamp = datetime.now().isoformat(timespec="seconds")

    # Save current ansatz
    current_ansatz_path = circuits_dir / "current_ansatz.qpy"
    _save_circuit(current_ansatz_path, experiment.ansatz)
    
    # Save trial ansatz history circuits + metadata
    trial_ansatz_manifest = []

    for idx, record in enumerate(experiment.trial_ansatz_history):
        action_slug = _slugify(record["action"])

        before_name = (
            f"{idx:04d}_outer_{int(record['outer_iteration']):04d}_"
            f"{action_slug}_before.qpy"
        )
        after_name = (
            f"{idx:04d}_outer_{int(record['outer_iteration']):04d}_"
            f"{action_slug}_{'accepted' if record['accepted'] else 'rejected'}.qpy"
        )

        before_path = trial_dir / before_name
        after_path = trial_dir / after_name

        _save_circuit(before_path, record["ansatz_before"])
        _save_circuit(after_path, record["ansatz_after"])

        trial_ansatz_manifest.append(
            {
                "history_index": idx,
                "outer_iteration": int(record["outer_iteration"]),
                "action": record["action"],
                "accepted": bool(record["accepted"]),
                "cost_before": record["cost_before"],
                "cost_after": record["cost_after"],
                "delta_cost": record["delta_cost"],
                "num_parameters_before": int(record["num_parameters_before"]),
                "num_parameters_after": int(record["num_parameters_after"]),
                "num_two_qubit_before": int(record["num_two_qubit_before"]),
                "num_two_qubit_after": int(record["num_two_qubit_after"]),
                "parameter_values": {
                    str(k): float(v) for k, v in record["parameter_values"].items()
                },
                "note": record["note"],
                "before_qpy_file": str(before_path.relative_to(save_dir)),
                "after_qpy_file": str(after_path.relative_to(save_dir)),
                "before_summary": _circuit_summary(record["ansatz_before"]),
                "after_summary": _circuit_summary(record["ansatz_after"]),
            }
        )

    # Save accepted ansatz history circuits + metadata
    accepted_ansatz_manifest = []
    for idx, record in enumerate(experiment.accepted_ansatz_history):
        slug = _slugify(record.action)
        qpy_name = f"{idx:04d}_outer_{int(record.outer_iteration):04d}_{slug}.qpy"
        qpy_path = accepted_dir / qpy_name
        _save_circuit(qpy_path, record.ansatz)

        accepted_ansatz_manifest.append(
            {
                "history_index": idx,
                "outer_iteration": int(record.outer_iteration),
                "action": record.action,
                "cost": None if record.cost is None else float(record.cost),
                "num_parameters": int(record.num_parameters),
                "num_two_qubit_gates": int(record.num_two_qubit_gates),
                "parameter_values": {
                    str(k): float(v) for k, v in record.parameter_values.items()
                },
                "note": record.note,
                "qpy_file": str(qpy_path.relative_to(save_dir)),
                "circuit_summary": _circuit_summary(record.ansatz),
            }
        )

    # Current live state
    current_state = {
        "saved_at": timestamp,
        "outer_iteration": int(experiment._outer_iteration),
        "last_cost": None if experiment.last_cost is None else float(experiment.last_cost),
        "last_params": None if experiment.last_params is None else np.asarray(experiment.last_params, dtype=float).tolist(),
        "current_parameter_dict": {
            str(k): float(v) for k, v in experiment.get_current_parameter_dict().items()
        },
        "parameter_memory": _serialize_parameter_memory_cache(experiment.parameter_memory),
        "locked_gates": _jsonable(experiment.locked_gates),
        "two_qubit_gate_map": {
            str(k): list(v) for k, v in experiment._2qbg_positions.items()
        },
        "current_ansatz_qpy": str(current_ansatz_path.relative_to(save_dir)),
        "current_ansatz_summary": _circuit_summary(experiment.ansatz),
    }

    # Histories
    parameter_memory_history_payload = [
        _serialize_parameter_memory_record(record)
        for record in experiment.parameter_memory_history
    ]

    outer_step_history_payload = [
        _serialize_outer_step_result(record)
        for record in experiment.outer_step_history
    ]

    training_run_history_payload = [
        _serialize_training_run_record(record)
        for record in experiment.training_run_history
    ]

    gradient_history_payload = (
        None
        if experiment.gradient_history is None
        else {
            str(k): [np.asarray(g, dtype=float).tolist() for g in v]
            for k, v in experiment.gradient_history.items()
        }
    )

    result_history_payload = (
        None
        if experiment.result_history is None
        else [_serialize_optimizer_result(result) for result in experiment.result_history]
    )

    manifest = {
        "schema_version": 1,
        "saved_at": timestamp,
        "num_outer_steps": len(experiment.outer_step_history),
        "num_parameter_memory_records": len(experiment.parameter_memory_history),
        "num_accepted_ansatz_records": len(experiment.accepted_ansatz_history),
        "num_training_runs": len(experiment.training_run_history),
        "has_gradient_history": experiment.gradient_history is not None,
        "has_result_history": experiment.result_history is not None,
        "files": {
            "current_state": "current_state.json",
            "outer_step_history": "outer_step_history.json",
            "parameter_memory_history": "parameter_memory_history.json",
            "trial_ansatz_history": "trial_ansatz_history.json",
            "accepted_ansatz_history": "accepted_ansatz_history.json",
            "training_run_history": "training_run_history.json",
            "gradient_history": "gradient_history.json",
            "result_history": "result_history.json",
            "current_ansatz_qpy": str(current_ansatz_path.relative_to(save_dir)),
        },
    }

    _write_json(save_dir / "manifest.json", manifest)
    _write_json(save_dir / "current_state.json", current_state)
    _write_json(save_dir / "outer_step_history.json", outer_step_history_payload)
    _write_json(save_dir / "parameter_memory_history.json", parameter_memory_history_payload)
    _write_json(save_dir / "trial_ansatz_history.json", trial_ansatz_manifest)
    _write_json(save_dir / "accepted_ansatz_history.json", accepted_ansatz_manifest)
    _write_json(save_dir / "training_run_history.json", training_run_history_payload)
    _write_json(save_dir / "gradient_history.json", gradient_history_payload)
    _write_json(save_dir / "result_history.json", result_history_payload)

    return save_dir
    