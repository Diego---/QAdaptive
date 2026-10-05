import json

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

from qadaptive.training.recorder import InnerLoopRecorder


def test_recorder_owns_complete_multi_run_history():
    recorder = InnerLoopRecorder(record_gradients=True)

    first = recorder.start_run(
        param_names=["theta_0", "theta_1"],
        initial_point=[1.0, -1.0],
        initial_value=2.0,
        outer_iteration=0,
        action="initial_train",
    )
    recorder(
        iteration=1,
        nfev=3,
        params=[0.8, -0.7],
        value=1.5,
        stepsize=0.2,
        accepted=True,
        gradient=[1.0, -0.5],
    )
    recorder.finish_run(final_params=[0.8, -0.7], final_value=1.13)
    recorder.set_outer_result(accepted=True)

    second = recorder.start_run(
        param_names=["theta_0", "theta_1", "theta_2"],
        initial_point=[0.8, -0.7, 0.0],
        outer_iteration=1,
        action="insert_block",
    )
    recorder.finish_run(final_params=[0.8, -0.7, 0.0], final_value=1.13)
    recorder.set_outer_result(accepted=False, note="Rejected and rolled back.")

    assert recorder.runs == [first, second]
    assert first.accepted_outer_step is True
    assert second.accepted_outer_step is False
    assert second.note == "Rejected and rolled back."
    assert recorder.values == [1.5]
    assert recorder.nfevs == [3]
    assert recorder.gradient_norms == [pytest.approx(np.sqrt(1.25))]


def test_recorder_evaluates_extra_objective_at_configured_frequency():
    recorder = InnerLoopRecorder(
        extra_objective=lambda params: np.sum(params),
        extra_evaluation_frequency=2,
    )
    recorder.start_run(param_names=["theta"], initial_point=[1.0])

    for iteration, value in enumerate([0.9, 0.8, 0.7], start=1):
        recorder(
            iteration=iteration,
            nfev=iteration,
            params=[value],
            value=value**2,
            stepsize=0.1,
            accepted=True,
        )

    recorder.finish_run(final_params=[0.7], final_value=0.49)

    extras = [item.extra_value for item in recorder.runs[0].iterations]
    assert extras == [None, pytest.approx(0.8), None]
    figure, _ = recorder.plot_objective(include_initial=False)
    assert figure is not None


def test_recorder_plots_and_saves_without_external_trace_arrays(tmp_path):
    recorder = InnerLoopRecorder()
    recorder.start_run(
        param_names=["theta_0"],
        initial_point=[1.0],
        initial_value=1.0,
        outer_iteration=0,
        action="initial_train",
    )
    recorder(
        iteration=1,
        nfev=1,
        params=[0.5],
        value=0.25,
        stepsize=0.5,
        accepted=True,
        gradient=[1.0],
    )
    recorder.finish_run(final_params=[0.5], final_value=0.25)
    recorder.set_outer_result(accepted=True)

    objective_figure, _ = recorder.plot_objective()
    parameter_figure, _ = recorder.plot_parameters()
    heatmap_figure, _ = recorder.plot_parameter_heatmap()

    assert objective_figure is not None
    assert parameter_figure is not None
    assert heatmap_figure is not None

    output = recorder.save(tmp_path / "recorder.json")
    payload = json.loads(output.read_text(encoding="utf-8"))

    assert payload["schema_version"] == 1
    assert payload["runs"][0]["iterations"][0]["nfev"] == 1
    assert payload["runs"][0]["final_params"] == [0.5]


def test_recorder_retains_partial_run_after_abort():
    recorder = InnerLoopRecorder()
    recorder.start_run(param_names=["theta"], initial_point=[1.0])
    recorder.abort_run("hardware interruption")

    assert recorder.active_run is None
    assert len(recorder.runs) == 1
    assert recorder.last_run.note == "hardware interruption"



def test_recorder_records_serializes_and_plots_parameter_schedules():
    recorder = InnerLoopRecorder()
    recorder.start_run(
        param_names=["θ_0", "θ_1"],
        initial_point=[0.0, 0.0],
    )
    
    recorder.set_parameter_birth_outer_iterations(
        {
            "θ_0": 0,
            "θ_1": 3,
        }
    )

    recorder(
        iteration=1,
        nfev=2,
        params=[0.1, -0.1],
        value=0.5,
        stepsize=0.1,
        accepted=True,
        schedule_steps_used={"θ_0": 0, "θ_1": 3},
        learning_rates={"θ_0": 0.8, "θ_1": 0.4},
        perturbations={"θ_0": 0.2, "θ_1": 0.1},
    )
    recorder(
        iteration=2,
        nfev=4,
        params=[0.2, -0.2],
        value=0.4,
        stepsize=0.1,
        accepted=True,
        schedule_steps_used={"θ_0": 1, "θ_1": 4},
        learning_rates={"θ_0": 0.6, "θ_1": 0.3},
        perturbations={"θ_0": 0.18, "θ_1": 0.09},
    )
    recorder.finish_run(final_params=[0.2, -0.2], final_value=0.4)

    payload = recorder.to_dict()
    first = payload["runs"][0]["iterations"][0]
    assert first["schedule_steps_used"] == {"θ_0": 0, "θ_1": 3}
    assert first["learning_rates"] == {"θ_0": 0.8, "θ_1": 0.4}
    assert first["perturbations"] == {"θ_0": 0.2, "θ_1": 0.1}
    assert payload["runs"][0]["parameter_birth_outer_iterations"] == {
        "θ_0": 0,
        "θ_1": 3,
    }

    lr_figure, lr_axes = recorder.plot_learning_rates(parameters=["θ_1"])
    c_figure, c_axes = recorder.plot_perturbations(parameters=["θ_0"])

    lr_line = next(line for line in lr_axes.lines if line.get_label() == "θ_1")
    c_line = next(line for line in c_axes.lines if line.get_label() == "θ_0")

    np.testing.assert_allclose(
        lr_line.get_ydata()[np.isfinite(lr_line.get_ydata())],
        [0.4, 0.3],
    )
    np.testing.assert_allclose(
        c_line.get_ydata()[np.isfinite(c_line.get_ydata())],
        [0.2, 0.18],
    )

    assert lr_figure is not None
    assert c_figure is not None
