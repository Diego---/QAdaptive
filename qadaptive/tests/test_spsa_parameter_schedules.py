import pytest

import numpy as np

from qadaptive.training.optimizers import SPSA


def quadratic(x):
    return float(np.sum(np.asarray(x) ** 2))


def make_optimizer():
    return SPSA(
        learning_rate=0.1,
        perturbation=0.1,
        parameter_dependent_lr_schedule=True,
        parameter_dependent_perturbation_schedule=True,
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
        parameter_dependent_lr_schedule=True,
        parameter_dependent_perturbation_schedule=True,
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

    learning_rates = optimizer._get_parameter_learning_rates()
    perturbations = optimizer._get_parameter_perturbations()

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
            parameter_dependent_lr_schedule=True,
            parameter_dependent_perturbation_schedule=True,
            second_order=True,
        )

def test_point_sample_supports_parameter_dependent_perturbations():
    optimizer = SPSA(
        learning_rate=0.1,
        perturbation=0.1,
        parameter_dependent_lr_schedule=True,
        parameter_dependent_perturbation_schedule=True,
    )

    evaluated_points = []

    def linear_loss(x):
        x = np.asarray(x, dtype=float)
        evaluated_points.append(x.copy())
        return 2.0 * x[0] + 3.0 * x[1]

    x = np.array([1.0, 2.0])
    eps = np.array([0.2, 0.1])
    delta = np.array([1.0, -1.0])

    fx, gradient, hessian = optimizer._point_sample(
        linear_loss,
        x,
        eps,
        delta,
    )

    np.testing.assert_allclose(
        evaluated_points[0],
        np.array([1.2, 1.9]),
    )
    np.testing.assert_allclose(
        evaluated_points[1],
        np.array([0.8, 2.1]),
    )

    np.testing.assert_allclose(
        gradient,
        np.array([0.5, -1.0]),
    )

    assert fx == pytest.approx(8.0)
    assert hessian is None

def test_process_update_uses_parameter_dependent_learning_rates():
    optimizer = SPSA(
        parameter_dependent_lr_schedule=True,
        parameter_dependent_perturbation_schedule=True,
    )

    optimizer.set_power_series_hyperparameters(
        a=0.8,
        alpha=0.5,
        c=0.2,
        gamma=0.25,
        stability_constant=0.0,
    )

    x = np.array([1.0, 2.0])
    gradient = np.array([2.0, 3.0])

    optimizer.initialize(
        x,
        lambda x: float(np.sum(x**2)),
        parameter_names=["θ_0", "θ_1"],
    )

    optimizer._parameter_schedule_steps = {
        "θ_0": 0,
        "θ_1": 3,
    }

    skip, x_next, fx_next = optimizer.process_update(
        gradient_estimate=gradient,
        x=x,
        fx=0.0,
        fun=lambda x: float(np.sum(x**2)),
    )

    expected_learning_rates = np.array([
        0.8 / 1**0.5,
        0.8 / 4**0.5,
    ])

    expected_update = gradient * expected_learning_rates
    expected_x_next = x - expected_update

    assert not skip
    assert fx_next is None

    np.testing.assert_allclose(
        x_next,
        expected_x_next,
    )

def test_parameter_dependent_step_uses_current_schedule_then_advances(
    monkeypatch,
):
    optimizer = SPSA(
        parameter_dependent_lr_schedule=True,
        parameter_dependent_perturbation_schedule=True,
    )

    optimizer.set_power_series_hyperparameters(
        a=0.8,
        alpha=0.5,
        c=0.2,
        gamma=0.25,
        stability_constant=0.0,
    )

    x = np.array([1.0, 2.0])

    optimizer.initialize(
        x,
        lambda x: 2.0 * x[0] + 3.0 * x[1],
        parameter_names=["θ_0", "θ_1"],
    )

    optimizer._parameter_schedule_steps = {
        "θ_0": 0,
        "θ_1": 3,
    }

    monkeypatch.setattr(
        "qadaptive.training.optimizers.spsa.bernoulli_perturbation",
        lambda dim, perturbation_dims=None: np.array([1.0, -1.0]),
    )

    skip, x_next, _, gradient, _ = optimizer.step(
        x,
        lambda x: 2.0 * x[0] + 3.0 * x[1],
    )

    assert not skip

    assert optimizer.parameter_schedule_steps == {
        "θ_0": 1,
        "θ_1": 4,
    }

    assert optimizer.last_parameter_schedule_steps_used == {
        "θ_0": 0,
        "θ_1": 3,
    }
    assert optimizer.last_parameter_learning_rates == pytest.approx({
        "θ_0": 0.8,
        "θ_1": 0.4,
    })
    assert optimizer.last_parameter_perturbations == pytest.approx({
        "θ_0": 0.2,
        "θ_1": 0.2 / np.sqrt(2.0),
    })

def test_parameter_birth_outer_iterations_are_preserved_for_survivors():
    optimizer = make_optimizer()

    optimizer.initialize(
        np.zeros(2),
        quadratic,
        parameter_names=["θ_0", "θ_1"],
        outer_iteration=0,
    )

    optimizer.initialize(
        np.zeros(3),
        quadratic,
        parameter_names=["θ_0", "θ_1", "θ_2"],
        outer_iteration=4,
    )

    assert optimizer.parameter_birth_outer_iterations == {
        "θ_0": 0,
        "θ_1": 0,
        "θ_2": 4,
    }
    
def test_parameter_birth_modulation_scales_new_parameter_schedules():
    optimizer = SPSA(
        parameter_dependent_lr_schedule=True,
        parameter_dependent_perturbation_schedule=True,
    )

    optimizer.set_power_series_hyperparameters(
        a=0.8,
        alpha=0.5,
        c=0.2,
        gamma=0.25,
        stability_constant=0.0,
    )

    optimizer.set_parameter_birth_power_series(
        learning_rate_exponent=0.5,
        perturbation_exponent=0.25,
    )

    optimizer.initialize(
        np.zeros(1),
        quadratic,
        parameter_names=["θ_0"],
        outer_iteration=0,
    )

    optimizer.initialize(
        np.zeros(2),
        quadratic,
        parameter_names=["θ_0", "θ_1"],
        outer_iteration=3,
    )

    learning_rates = optimizer._get_parameter_learning_rates()
    perturbations = optimizer._get_parameter_perturbations()

    np.testing.assert_allclose(
        learning_rates,
        [
            0.8,
            0.8 / 4**0.5,
        ],
    )

    np.testing.assert_allclose(
        perturbations,
        [
            0.2,
            0.2 / 4**0.25,
        ],
    )
    
def test_restart_parameter_schedules_preserves_birth_iterations():
    optimizer = make_optimizer()

    optimizer.initialize(
        np.zeros(1),
        quadratic,
        parameter_names=["θ_0"],
        outer_iteration=2,
    )

    optimizer.step(np.zeros(1), quadratic)

    births_before = optimizer.parameter_birth_outer_iterations

    optimizer.restart_parameter_schedules()

    assert optimizer.parameter_schedule_steps == {
        "θ_0": 0,
    }

    assert optimizer.parameter_birth_outer_iterations == births_before


def test_spsa_minimize_termination_checker_receives_value_at_reported_point():
    observations = []

    def checker(nfev, params, value, stepsize, accepted):
        del nfev, stepsize, accepted
        observations.append((np.asarray(params, dtype=float).copy(), float(value)))
        return True

    optimizer = SPSA(
        maxiter=5,
        learning_rate=0.1,
        perturbation=0.1,
        termination_checker=checker,
    )

    optimizer.minimize(quadratic, np.array([0.8, -0.4]))

    assert len(observations) == 1
    params, value = observations[0]
    assert value == pytest.approx(quadratic(params))
