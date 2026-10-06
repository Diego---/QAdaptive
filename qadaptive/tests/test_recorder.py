import json

import matplotlib
import numpy as np
import pytest
from typing import get_args, get_type_hints

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
        optimizer_estimate=1.5,
        evaluation_value=1.4,
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
    assert recorder.values == [1.4]
    assert recorder.optimizer_estimates == [1.5]
    assert recorder.evaluation_values == [1.4]
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
            optimizer_estimate=value**2,
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
        optimizer_estimate=0.25,
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

    assert payload["schema_version"] == 2
    saved_iteration = payload["runs"][0]["iterations"][0]
    assert saved_iteration["nfev"] == 1
    assert saved_iteration["optimizer_estimate"] == pytest.approx(0.25)
    assert saved_iteration["evaluation_value"] is None
    assert saved_iteration["value"] == pytest.approx(0.25)
    assert payload["runs"][0]["final_params"] == [0.5]


def test_objective_plot_evaluation_source_uses_updated_points_and_final_value():
    recorder = InnerLoopRecorder()
    recorder.start_run(
        param_names=["theta"],
        initial_point=[1.0],
        initial_value=1.0,
    )
    recorder(
        iteration=1,
        nfev=2,
        params=[0.8],
        optimizer_estimate=1.0,
        evaluation_value=0.64,
        stepsize=0.2,
        accepted=True,
    )
    recorder(
        iteration=2,
        nfev=4,
        params=[0.6],
        optimizer_estimate=0.64,
        evaluation_value=0.36,
        stepsize=0.2,
        accepted=True,
    )
    recorder.finish_run(final_params=[0.6], final_value=0.35)

    figure, axes = recorder.plot_objective(source="evaluation")
    line = next(
        line for line in axes.lines
        if line.get_label() == "Explicit evaluation"
    )

    np.testing.assert_array_equal(line.get_xdata(), [0.0, 1.0, 2.0])
    np.testing.assert_allclose(line.get_ydata(), [1.0, 0.64, 0.35])
    assert figure is not None


def test_objective_plot_optimizer_estimate_is_aligned_to_pre_update_points():
    recorder = InnerLoopRecorder()
    recorder.start_run(
        param_names=["theta"],
        initial_point=[1.0],
        initial_value=1.0,
    )
    recorder(
        iteration=1,
        nfev=2,
        params=[0.8],
        optimizer_estimate=1.0,
        stepsize=0.2,
        accepted=True,
    )
    recorder(
        iteration=2,
        nfev=4,
        params=[0.6],
        optimizer_estimate=0.64,
        stepsize=0.2,
        accepted=True,
    )
    recorder.finish_run(final_params=[0.6], final_value=0.36)

    figure, axes = recorder.plot_objective(source="optimizer_estimate")
    line = next(
        line for line in axes.lines
        if line.get_label() == "Optimizer estimate"
    )

    x = np.asarray(line.get_xdata(), dtype=float)
    y = np.asarray(line.get_ydata(), dtype=float)
    finite = np.isfinite(y)

    np.testing.assert_array_equal(x[finite], [0.0, 1.0])
    np.testing.assert_allclose(y[finite], [1.0, 0.64])
    assert np.isnan(y[-1])
    assert figure is not None


def test_objective_plot_auto_selects_source_from_recorded_data():
    estimates_only = InnerLoopRecorder()
    estimates_only.start_run(param_names=["theta"], initial_point=[1.0])
    estimates_only(
        iteration=1,
        nfev=2,
        params=[0.8],
        optimizer_estimate=1.0,
        stepsize=0.2,
        accepted=True,
    )
    estimates_only.finish_run(final_params=[0.8], final_value=0.64)

    _, axes = estimates_only.plot_objective(source="auto")
    assert axes.lines[0].get_label() == "Optimizer estimate"

    evaluated = InnerLoopRecorder()
    evaluated.start_run(param_names=["theta"], initial_point=[1.0])
    evaluated(
        iteration=1,
        nfev=3,
        params=[0.8],
        optimizer_estimate=1.0,
        evaluation_value=0.64,
        stepsize=0.2,
        accepted=True,
    )
    evaluated.finish_run(final_params=[0.8], final_value=0.63)

    _, axes = evaluated.plot_objective(source="auto")
    assert axes.lines[0].get_label() == "Explicit evaluation"


def test_objective_plot_rejects_unknown_source():
    recorder = InnerLoopRecorder()
    recorder.start_run(param_names=["theta"], initial_point=[1.0])
    recorder.finish_run(final_params=[1.0], final_value=1.0)

    with pytest.raises(ValueError, match="objective_source"):
        recorder.plot_objective(source="mystery")

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
        optimizer_estimate=0.5,
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
        optimizer_estimate=0.4,
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


def test_plot_objective_source_annotation_lists_supported_modes():
    hints = get_type_hints(InnerLoopRecorder.plot_objective)
    assert set(get_args(hints["source"])) == {
        "auto",
        "evaluation",
        "optimizer_estimate",
    }
