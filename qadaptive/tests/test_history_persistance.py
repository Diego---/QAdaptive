import json, pytest

import numpy as np
from qiskit import QuantumCircuit, qpy
from qiskit.circuit import Parameter
from qiskit_algorithms.utils import algorithm_globals

from qadaptive.core.adaptive_ansatz import AdaptiveAnsatz
from qadaptive.outer.action_definitions import INSERT_GATE
from qadaptive.outer.mutable_ansatz_experiment import MutableAnsatzExperiment
from qadaptive.outer.outer_loop import ActionSpec, OuterStepPlan
from qadaptive.persistence import history as persistence_history
from qadaptive.persistence.loading import load_experiment_history
from qadaptive.training.optimizers import SPSA
from qadaptive.training.recorder import InnerLoopRecorder
from qadaptive.training.trainer import InnerLoopTrainer


def _read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _load_circuit(path):
    with path.open("rb") as file:
        circuits = qpy.load(file)

    assert len(circuits) == 1
    return circuits[0]


def _quadratic_loss(params, ansatz, **kwargs):
    del ansatz, kwargs
    return float(np.sum(np.asarray(params, dtype=float) ** 2))


def _append_rx_plan(experiment):
    return OuterStepPlan(
        name="append_rx",
        actions=[
            ActionSpec(
                action=INSERT_GATE,
                kwargs={
                    "gate": "rx",
                    "qubits": [0],
                    "circ_ind": len(experiment.ansatz.data),
                },
            )
        ],
        acceptance_mode="force",
    )


def test_save_history_preserves_accepted_run_data(tmp_path, monkeypatch):
    """Save and read back a trained experiment with one accepted insertion."""
    monkeypatch.setattr(algorithm_globals, "random_seed", 1234)

    circuit = QuantumCircuit(2)
    circuit.rx(Parameter("θ_0"), 0)
    circuit.cx(0, 1)

    recorder = InnerLoopRecorder(
        record_initial_value=True,
        record_gradients=True,
        extra_objective=lambda params: float(np.sum(params)),
        extra_evaluation_frequency=1,
    )
    trainer = InnerLoopTrainer(
        optimizer=SPSA(learning_rate=0.1, perturbation=0.1),
        recorder=recorder,
    )
    experiment = MutableAnsatzExperiment(AdaptiveAnsatz(circuit), trainer)

    results = experiment.run_outer_loop(
        loss_function=_quadratic_loss,
        plan_schedule=[_append_rx_plan],
        outer_iterations=1,
        train_iterations=2,
        initial_point=[0.5],
        reuse_parameter_memory=True,
    )

    assert len(results) == 1
    assert results[0].accepted is True
    assert len(recorder.runs) == 2

    output = tmp_path / "run"
    assert experiment.save_history(output) == output

    # Every declared file exists, and declared JSON files are readable.
    manifest = _read_json(output / "manifest.json")

    assert manifest["schema_version"] == 1
    assert manifest["num_outer_steps"] == 2  # Includes initial training.
    assert manifest["num_training_runs"] == 2
    assert manifest["num_accepted_ansatz_records"] == 2
    
    metadata = manifest["inner_loop_recorder"]

    assert metadata["schema_version"] == 1
    assert metadata["record_initial_value"] is True
    assert metadata["record_gradients"] is True
    assert metadata["extra_evaluation_frequency"] == 1
    assert "runs" not in metadata

    payloads = {}
    for name, relative_path in manifest["files"].items():
        path = output / relative_path
        assert path.is_file(), f"Missing archive file: {relative_path}"

        if path.suffix == ".json":
            payloads[name] = _read_json(path)

    # The circuit and saved parameter values reconstruct the final state.
    state = payloads["current_state"]
    circuit_path = manifest["files"]["current_ansatz_qpy"]
    saved_circuit = _load_circuit(output / circuit_path)

    assert state["current_ansatz_qpy"] == circuit_path
    assert saved_circuit == experiment.ansatz
    assert [p.name for p in saved_circuit.parameters] == [
        p.name for p in experiment.ansatz.parameters
    ]
    assert state["last_cost"] == experiment.last_cost
    np.testing.assert_array_equal(
        state["last_params"],
        experiment.last_params,
    )
    assert (
        state["current_parameter_dict"]
        == experiment.get_current_parameter_dict()
    )

    bound_circuit = saved_circuit.assign_parameters(
        {
            p: state["current_parameter_dict"][p.name]
            for p in saved_circuit.parameters
        }
    )
    expected_bound_circuit = experiment.ansatz.assign_parameters(
        experiment.last_params
    )

    assert bound_circuit.num_parameters == 0
    assert bound_circuit == expected_bound_circuit

    # Both inner runs retain their numerical data and outer-loop metadata.
    saved_runs = payloads["training_run_history"]
    assert len(saved_runs) == len(recorder.runs)

    for saved_run, run in zip(saved_runs, recorder.runs):
        for field in (
            "run_index",
            "param_names",
            "initial_value",
            "final_value",
            "outer_iteration",
            "action",
            "accepted_outer_step",
            "note",
        ):
            assert saved_run[field] == getattr(run, field)

        np.testing.assert_array_equal(
            saved_run["initial_point"], run.initial_point
        )
        np.testing.assert_array_equal(
            saved_run["final_params"], run.final_params
        )

        assert len(run.iterations) == 2
        assert len(saved_run["iterations"]) == len(run.iterations)

        for saved_step, step in zip(saved_run["iterations"], run.iterations):
            for field in (
                "iteration",
                "nfev",
                "value",
                "stepsize",
                "accepted",
                "extra_value",
                "extra_std",
            ):
                assert saved_step[field] == getattr(step, field)

            assert step.gradient is not None
            assert step.extra_value is not None

            np.testing.assert_array_equal(saved_step["params"], step.params)
            np.testing.assert_array_equal(saved_step["gradient"], step.gradient)

    assert payloads["outer_step_history"] == [
        vars(record) for record in experiment.outer_step_history
    ]

    # Accepted snapshots preserve their circuits and parameter values.
    saved_accepted = payloads["accepted_ansatz_history"]
    assert len(saved_accepted) == len(experiment.accepted_ansatz_history) == 2

    for saved, record in zip(
        saved_accepted, experiment.accepted_ansatz_history
    ):
        assert _load_circuit(output / saved["qpy_file"]) == record.ansatz
        assert saved["parameter_values"] == record.parameter_values
        assert saved["cost"] == record.cost
        assert saved["num_parameters"] == record.num_parameters
        assert saved["num_two_qubit_gates"] == record.num_two_qubit_gates == 1

    # The attempted structural change preserves both circuit snapshots.
    saved_trials = payloads["trial_ansatz_history"]
    assert len(saved_trials) == len(experiment.trial_ansatz_history) == 1

    saved_trial = saved_trials[0]
    trial = experiment.trial_ansatz_history[0]

    assert saved_trial["accepted"] is True
    assert (
        _load_circuit(output / saved_trial["before_qpy_file"])
        == trial["ansatz_before"]
    )
    assert (
        _load_circuit(output / saved_trial["after_qpy_file"])
        == trial["ansatz_after"]
    )
    assert saved_trial["parameter_values"] == trial["parameter_values"]
    
    loaded = load_experiment_history(output)

    assert loaded.manifest == manifest
    assert loaded.ansatz == experiment.ansatz
    assert loaded.last_cost == experiment.last_cost
    np.testing.assert_array_equal(loaded.last_params, experiment.last_params)

    for name, payload in payloads.items():
        assert getattr(loaded, name) == payload

def test_save_history_preserves_rejected_trial_and_accepted_state(
    tmp_path, monkeypatch
):
    """Retain a rejected proposal without replacing the accepted state."""
    monkeypatch.setattr(algorithm_globals, "random_seed", 1234)

    circuit = QuantumCircuit(2)
    circuit.rx(Parameter("θ_0"), 0)
    circuit.cx(0, 1)

    recorder = InnerLoopRecorder(
        record_initial_value=True,
        record_gradients=True,
    )
    trainer = InnerLoopTrainer(
        optimizer=SPSA(learning_rate=0.1, perturbation=0.1),
        recorder=recorder,
    )
    experiment = MutableAnsatzExperiment(AdaptiveAnsatz(circuit), trainer)

    experiment.run_outer_loop(
        loss_function=_quadratic_loss,
        plan_schedule=[_append_rx_plan],
        outer_iterations=1,
        train_iterations=2,
        initial_point=[0.5],
        reuse_parameter_memory=True,
    )

    accepted_circuit = experiment.ansatz.copy()
    accepted_params = experiment.last_params.copy()
    accepted_cost = experiment.last_cost
    accepted_values = experiment.get_current_parameter_dict().copy()

    assert accepted_circuit.num_parameters == 2
    assert len(experiment.accepted_ansatz_history) == 2

    plan = OuterStepPlan(
        name="append_rx_rejected",
        actions=_append_rx_plan(experiment).actions,
        acceptance_mode="outer",
    )

    # The loss is nonnegative. Adding one parameter with this penalty
    # guarantees rejection, even if retraining reduces the raw loss to zero.
    penalty_scale = accepted_cost + 1.0
    result = experiment.run_outer_step(
        loss_function=_quadratic_loss,
        plan=plan,
        train_iterations=1,
        reuse_parameter_memory=True,
        complexity_penalty=lambda circuit: (
            penalty_scale * circuit.num_parameters
        ),
        metropolis_temperature=None,
    )

    assert result.accepted is False
    assert experiment.ansatz == accepted_circuit
    np.testing.assert_array_equal(experiment.last_params, accepted_params)
    assert experiment.last_cost == accepted_cost

    output = experiment.save_history(tmp_path / "run")
    manifest = _read_json(output / "manifest.json")
    payloads = {
        name: _read_json(output / manifest["files"][name])
        for name in (
            "current_state",
            "outer_step_history",
            "accepted_ansatz_history",
            "trial_ansatz_history",
            "training_run_history",
        )
    }

    # The final archive still represents the accepted circuit and parameters.
    state = payloads["current_state"]
    saved_circuit = _load_circuit(
        output / manifest["files"]["current_ansatz_qpy"]
    )

    assert saved_circuit == accepted_circuit
    np.testing.assert_array_equal(state["last_params"], accepted_params)
    assert state["last_cost"] == accepted_cost
    assert state["current_parameter_dict"] == accepted_values

    accepted_history = payloads["accepted_ansatz_history"]
    assert len(accepted_history) == 2
    assert (
        _load_circuit(output / accepted_history[-1]["qpy_file"])
        == accepted_circuit
    )
    assert accepted_history[-1]["parameter_values"] == accepted_values

    # The rejected proposal remains available as a separate circuit snapshot.
    trials = payloads["trial_ansatz_history"]
    assert len(trials) == 2

    trial = trials[-1]
    assert trial["accepted"] is False
    assert (
        _load_circuit(output / trial["before_qpy_file"])
        == accepted_circuit
    )

    rejected_circuit = _load_circuit(output / trial["after_qpy_file"])
    assert rejected_circuit.num_parameters == 3
    assert (
        rejected_circuit
        == experiment.trial_ansatz_history[-1]["ansatz_after"]
    )

    # Its inner-loop data survives, with the outer rejection attached.
    runs = payloads["training_run_history"]
    assert [run["accepted_outer_step"] for run in runs] == [
        True, True, False
    ]

    rejected_run = runs[-1]
    assert len(rejected_run["param_names"]) == 3
    assert len(rejected_run["iterations"]) == 1
    assert rejected_run["final_value"] == result.cost_after

    np.testing.assert_array_equal(
        rejected_run["final_params"],
        recorder.last_run.final_params,
    )
    assert trial["parameter_values"] == dict(
        zip(rejected_run["param_names"], rejected_run["final_params"])
    )

    saved_step = rejected_run["iterations"][0]
    live_step = recorder.last_run.iterations[0]

    assert live_step.gradient is not None
    assert saved_step["value"] == live_step.value
    assert saved_step["nfev"] == live_step.nfev
    np.testing.assert_array_equal(saved_step["params"], live_step.params)
    np.testing.assert_array_equal(saved_step["gradient"], live_step.gradient)

    assert [step["accepted"] for step in payloads["outer_step_history"]] == [
        True, True, False
    ]
    
    loaded = load_experiment_history(output)

    assert loaded.ansatz == accepted_circuit
    assert loaded.last_cost == accepted_cost
    np.testing.assert_array_equal(loaded.last_params, accepted_params)
    assert loaded.trial_ansatz_history[-1]["accepted"] is False
    
def test_failed_json_write_preserves_existing_file(tmp_path):
    path = tmp_path / "history.json"
    path.write_text('{"original": true}\n', encoding="utf-8")
    original_bytes = path.read_bytes()

    # Serialization fails after some replacement content has been written.
    with pytest.raises(TypeError):
        persistence_history._write_json(
            path,
            {"first": True, "unserializable": object()},
        )

    assert path.read_bytes() == original_bytes
    assert set(tmp_path.iterdir()) == {path}


def test_failed_qpy_write_preserves_existing_file(tmp_path, monkeypatch):
    path = tmp_path / "circuit.qpy"

    original_circuit = QuantumCircuit(1)
    original_circuit.x(0)
    with path.open("wb") as file:
        qpy.dump(original_circuit, file)

    original_bytes = path.read_bytes()

    def failing_dump(circuit, file):
        file.write(b"partial replacement")
        raise OSError("simulated QPY write failure")

    monkeypatch.setattr(persistence_history.qpy, "dump", failing_dump)

    replacement = QuantumCircuit(1)
    replacement.h(0)

    with pytest.raises(OSError, match="simulated QPY write failure"):
        persistence_history._save_circuit(path, replacement)

    assert path.read_bytes() == original_bytes
    assert _load_circuit(path) == original_circuit
    assert set(tmp_path.iterdir()) == {path}

def _make_untrained_experiment():
    circuit = QuantumCircuit(1)
    circuit.rx(Parameter("θ_0"), 0)

    trainer = InnerLoopTrainer(
        optimizer=SPSA(learning_rate=0.1, perturbation=0.1),
        recorder=InnerLoopRecorder(),
    )
    return MutableAnsatzExperiment(AdaptiveAnsatz(circuit), trainer)


def test_save_history_refuses_to_overwrite_completed_archive(tmp_path):
    experiment = _make_untrained_experiment()
    output = experiment.save_history(tmp_path / "run")

    def saved_files():
        return {
            path.relative_to(output): path.read_bytes()
            for path in output.rglob("*")
            if path.is_file()
        }

    original_files = saved_files()

    with pytest.raises(FileExistsError, match="already exists"):
        experiment.save_history(output)

    assert saved_files() == original_files


def test_failed_archive_save_does_not_publish_manifest(
    tmp_path, monkeypatch
):
    experiment = _make_untrained_experiment()
    output = tmp_path / "run"
    original_write = persistence_history._write_json

    def failing_write(path, payload):
        # Fail near the end, after earlier files have been saved.
        if path.name == "result_history.json":
            raise OSError("simulated archive write failure")
        original_write(path, payload)

    monkeypatch.setattr(
        persistence_history, "_write_json", failing_write
    )

    with pytest.raises(
        OSError, match="simulated archive write failure"
    ):
        experiment.save_history(output)

    assert (output / "circuits" / "current_ansatz.qpy").is_file()
    assert (output / "training_run_history.json").is_file()
    assert not (output / "manifest.json").exists()
    
    with pytest.raises(FileNotFoundError, match="manifest.json"):
        load_experiment_history(output)

def test_load_history_rejects_unknown_schema(tmp_path):
    output = _make_untrained_experiment().save_history(tmp_path / "run")
    path = output / "manifest.json"

    manifest = _read_json(path)
    manifest["schema_version"] = 999
    path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="Unsupported archive schema version"):
        load_experiment_history(output)
