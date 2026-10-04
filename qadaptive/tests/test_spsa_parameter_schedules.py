import pytest

import numpy as np

from qadaptive.training.optimizers import SPSA


def quadratic(x):
    return float(np.sum(np.asarray(x) ** 2))


def make_optimizer():
    return SPSA(
        learning_rate=0.1,
        perturbation=0.1,
        parameter_dependent_schedules=True,
    )
    
def test_parameter_schedule_steps_initialize_at_zero():
    optimizer = make_optimizer()

    optimizer.initialize(
        np.zeros(2),
        quadratic,
        parameter_names=["θ_0", "θ_1"],
    )

    assert optimizer.parameter_schedule_steps == {
        "θ_0": 0,
        "θ_1": 0,
    }
    
def test_parameter_schedule_steps_advance_after_step():
    optimizer = make_optimizer()

    x = np.zeros(2)

    optimizer.initialize(
        x,
        quadratic,
        parameter_names=["θ_0", "θ_1"],
    )

    optimizer.step(x, quadratic)

    assert optimizer.parameter_schedule_steps == {
        "θ_0": 1,
        "θ_1": 1,
    }
    
def test_new_parameters_start_at_zero_while_survivors_keep_their_steps():
    optimizer = make_optimizer()

    optimizer.initialize(
        np.zeros(2),
        quadratic,
        parameter_names=["θ_0", "θ_1"],
    )

    optimizer.step(np.zeros(2), quadratic)

    assert optimizer.parameter_schedule_steps == {
        "θ_0": 1,
        "θ_1": 1,
    }

    optimizer.initialize(
        np.zeros(3),
        quadratic,
        parameter_names=["θ_0", "θ_1", "θ_2"],
    )

    assert optimizer.parameter_schedule_steps == {
        "θ_0": 1,
        "θ_1": 1,
        "θ_2": 0,
    }
    
def test_removed_parameters_are_dropped_from_schedule_state():
    optimizer = make_optimizer()

    optimizer.initialize(
        np.zeros(3),
        quadratic,
        parameter_names=["θ_0", "θ_1", "θ_2"],
    )

    optimizer.step(np.zeros(3), quadratic)

    optimizer.initialize(
        np.zeros(2),
        quadratic,
        parameter_names=["θ_0", "θ_2"],
    )

    assert optimizer.parameter_schedule_steps == {
        "θ_0": 1,
        "θ_2": 1,
    }
    
def test_restart_parameter_schedules_resets_all_active_parameters():
    optimizer = make_optimizer()

    optimizer.initialize(
        np.zeros(2),
        quadratic,
        parameter_names=["θ_0", "θ_1"],
    )

    optimizer.step(np.zeros(2), quadratic)
    optimizer.step(np.zeros(2), quadratic)

    assert optimizer.parameter_schedule_steps == {
        "θ_0": 2,
        "θ_1": 2,
    }

    optimizer.restart_parameter_schedules()

    assert optimizer.parameter_schedule_steps == {
        "θ_0": 0,
        "θ_1": 0,
    }


def test_parameter_schedule_values_follow_individual_parameter_steps():
    optimizer = SPSA(
        parameter_dependent_schedules=True,
    )

    optimizer.set_power_series_hyperparameters(
        a=0.8,
        alpha=0.5,
        c=0.2,
        gamma=0.25,
        stability_constant=1.0,
    )

    optimizer.initialize(
        np.zeros(3),
        quadratic,
        parameter_names=["θ_0", "θ_1", "θ_2"],
    )

    optimizer._parameter_schedule_steps = {
        "θ_0": 0,
        "θ_1": 3,
        "θ_2": 8,
    }

    learning_rates, perturbations = optimizer._get_parameter_schedule_values()

    expected_learning_rates = np.array([
        0.8 / (1.0 + 1) ** 0.5,
        0.8 / (4.0 + 1) ** 0.5,
        0.8 / (9.0 + 1) ** 0.5,
    ])

    expected_perturbations = np.array([
        0.2 / 1.0 ** 0.25,
        0.2 / 4.0 ** 0.25,
        0.2 / 9.0 ** 0.25,
    ])

    np.testing.assert_allclose(
        learning_rates,
        expected_learning_rates,
    )
    np.testing.assert_allclose(
        perturbations,
        expected_perturbations,
    )

def test_parameter_dependent_schedules_reject_second_order_spsa():
    with pytest.raises(
        ValueError,
        match="first-order SPSA",
    ):
        SPSA(
            learning_rate=0.1,
            perturbation=0.1,
            parameter_dependent_schedules=True,
            second_order=True,
        )
