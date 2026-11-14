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

def test_parameter_dependent_perturbation_rejects_second_order_spsa():
    with pytest.raises(ValueError, match="first-order SPSA"):
        SPSA(
            parameter_dependent_perturbation_schedule=True,
            second_order=True,
        )


def test_parameter_dependent_lr_allows_second_order_spsa():
    optimizer = SPSA(
        parameter_dependent_lr_schedule=True,
        parameter_dependent_perturbation_schedule=False,
        second_order=True,
    )

    assert optimizer.second_order

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
    

@pytest.mark.parametrize("reference", ["global", "previous"])
@pytest.mark.parametrize("age", [0, 1, 10, 50])
def test_birth_offsets_follow_fixed_shifted_power_laws(reference, age):
    a, alpha, A = 0.8, 0.5, 1.0
    c, gamma = 0.2, 0.25
    optimizer = SPSA(
        parameter_dependent_lr_schedule=True,
        parameter_dependent_perturbation_schedule=True,
    )
    optimizer.set_power_series_hyperparameters(
        a=a, alpha=alpha, c=c, gamma=gamma, stability_constant=A
    )
    optimizer.set_parameter_birth_schedules(
        learning_rate_reference=reference,
        perturbation_reference=reference,
        learning_rate_ratio=1.2,
        perturbation_ratio=1.0,
        birth_position_rule=0.4,
    )
    optimizer.initialize(np.zeros(1), quadratic, parameter_names=["θ_0"])
    # Simulate 10 elapsed SPSA updates before growth.
    optimizer._total_spsa_updates = 10
    optimizer._parameter_schedule_steps["θ_0"] = 10
    optimizer._last_birth_generation["schedule_age"] = 10
    optimizer.initialize(
        np.zeros(2), quadratic, parameter_names=["θ_0", "θ_1"],
        outer_iteration=1,
    )
    event = optimizer.last_birth_schedule_event
    assert event["generation"] == 1
    assert event["weight"] == pytest.approx(0.4)
    offsets = optimizer.birth_schedule_state
    offset_a = offsets["learning_rate_offsets"]["θ_1"]
    offset_c = offsets["perturbation_offsets"]["θ_1"]
    optimizer._parameter_schedule_steps = {"θ_0": age + 10, "θ_1": age}
    np.testing.assert_allclose(
        optimizer._get_parameter_learning_rates(),
        [a / (A + age + 11)**alpha, a / (age + offset_a + 1)**alpha],
    )
    np.testing.assert_allclose(
        optimizer._get_parameter_perturbations(),
        [c / (age + 11)**gamma, c / (age + offset_c + 1)**gamma],
    )
    assert event["learning_rate"]["start"] == pytest.approx(
        a / (offset_a + 1)**alpha
    )
    assert event["perturbation"]["start"] == pytest.approx(
        c / (offset_c + 1)**gamma
    )


def test_birth_weight_list_repeats_last_value_and_tracks_insertion_batches():
    optimizer = SPSA(
        parameter_dependent_lr_schedule=True,
        parameter_dependent_perturbation_schedule=True,
    )
    optimizer.set_power_series_hyperparameters(
        a=0.4, alpha=0.602, c=0.1, gamma=0.101
    )
    optimizer.set_parameter_birth_schedules(
        birth_position_rule=[0.8, 0.3],
        learning_rate_ratio=1.0,
        perturbation_ratio=1.0,
    )
    optimizer.initialize(np.zeros(1), quadratic, parameter_names=["θ_0"])
    assert optimizer.last_birth_schedule_event["kind"] == "initial"
    assert optimizer.birth_schedule_state["generation_count"] == 0
    optimizer._total_spsa_updates = 20
    optimizer._last_birth_generation["schedule_age"] = 20
    optimizer.initialize(
        np.zeros(3), quadratic, parameter_names=["θ_0", "θ_1", "θ_2"]
    )
    assert optimizer.last_birth_schedule_event["weight"] == pytest.approx(0.8)
    assert optimizer.birth_schedule_state["generation_count"] == 1
    optimizer.initialize(
        np.zeros(3), quadratic, parameter_names=["θ_0", "θ_1", "θ_2"]
    )
    assert optimizer.last_birth_schedule_event is None
    assert optimizer.birth_schedule_state["generation_count"] == 1
    optimizer._total_spsa_updates = 40
    optimizer._last_birth_generation["schedule_age"] = 20
    optimizer.initialize(
        np.zeros(4), quadratic, parameter_names=["θ_0", "θ_1", "θ_2", "θ_3"]
    )
    assert optimizer.last_birth_schedule_event["weight"] == pytest.approx(0.3)
    optimizer._total_spsa_updates = 60
    optimizer._last_birth_generation["schedule_age"] = 20
    optimizer.initialize(
        np.zeros(5), quadratic, parameter_names=["θ_0", "θ_1", "θ_2", "θ_3", "θ_4"]
    )
    assert optimizer.last_birth_schedule_event["weight"] == pytest.approx(0.3)


def test_previous_generation_caps_infeasible_birth_interval(caplog):
    optimizer = SPSA(
        parameter_dependent_lr_schedule=True,
        parameter_dependent_perturbation_schedule=True,
    )
    optimizer.set_power_series_hyperparameters(
        a=0.4, alpha=0.602, c=0.1, gamma=0.101
    )
    optimizer.set_parameter_birth_schedules(
        learning_rate_reference="previous",
        perturbation_reference="previous",
        learning_rate_ratio=2.0,
        perturbation_ratio=2.0,
        birth_position_rule=0.3,
    )
    optimizer.initialize(np.zeros(1), quadratic, parameter_names=["θ_0"])
    # One update is too short for a twofold increase under either schedule.
    optimizer._total_spsa_updates = 1
    optimizer._last_birth_generation["schedule_age"] = 1
    optimizer.initialize(
        np.zeros(2), quadratic, parameter_names=["θ_0", "θ_1"]
    )
    event = optimizer.last_birth_schedule_event
    for key in ("learning_rate", "perturbation"):
        metric = event[key]
        assert metric["capped"] is True
        assert metric["start"] == pytest.approx(metric["upper"])
        assert metric["requested_lower"] > metric["upper"]
    assert "exceeds maximum" in caplog.text


def test_previous_generation_reference_survives_pruning_and_rollback():
    optimizer = SPSA(
        parameter_dependent_lr_schedule=True,
        parameter_dependent_perturbation_schedule=True,
    )
    optimizer.set_power_series_hyperparameters(
        a=0.4, alpha=0.602, c=0.1, gamma=0.101
    )
    optimizer.set_parameter_birth_schedules(
        learning_rate_reference="previous",
        perturbation_reference="previous",
        birth_position_rule=0.5,
    )
    optimizer.initialize(np.zeros(1), quadratic, parameter_names=["θ_0"])
    optimizer._total_spsa_updates = 50
    optimizer._last_birth_generation["schedule_age"] = 50
    optimizer.initialize(
        np.zeros(2), quadratic, parameter_names=["θ_0", "θ_1"]
    )
    prior = optimizer.birth_schedule_state
    optimizer._total_spsa_updates = 80
    optimizer._last_birth_generation["schedule_age"] = 30
    optimizer.initialize(np.zeros(1), quadratic, parameter_names=["θ_0"])
    assert optimizer.birth_schedule_state["previous_generation"] == prior["previous_generation"]
    optimizer.initialize(
        np.zeros(2), quadratic, parameter_names=["θ_0", "θ_temp"]
    )
    assert optimizer.birth_schedule_state["generation_count"] == 2
    optimizer.restore_parameter_schedules(
        {"θ_0": 0}, birth_outer_iterations={"θ_0": 0},
        birth_global_steps={"θ_0": 0},
        birth_policy_state={
            **prior,
            "learning_rate_offsets": {"θ_0": prior["learning_rate_offsets"]["θ_0"]},
            "perturbation_offsets": {"θ_0": prior["perturbation_offsets"]["θ_0"]},
        },
    )
    assert optimizer.birth_schedule_state["generation_count"] == 1
    assert optimizer.birth_schedule_state["previous_generation"] == prior["previous_generation"]
    assert optimizer._total_spsa_updates == 80


@pytest.mark.parametrize("bad", [[], [0.3, 1.1], -0.2, "unknown"])
def test_invalid_birth_position_rule_rejected(bad):
    optimizer = SPSA(parameter_dependent_lr_schedule=True)
    with pytest.raises((TypeError, ValueError)):
        optimizer.set_parameter_birth_schedules(birth_position_rule=bad)




def test_disabled_birth_schedule_requires_parameter_dependent_mode():
    optimizer = SPSA()
    with pytest.raises(ValueError, match="parameter-dependent"):
        optimizer.set_parameter_birth_schedules()


def test_birth_policy_requires_power_series_and_blocks_custom_override():
    optimizer = SPSA(parameter_dependent_lr_schedule=True)
    optimizer.set_power_series_hyperparameters(
        a=0.5, alpha=0.602, c=0.1, gamma=0.101
    )
    optimizer.set_parameter_birth_schedules()
    with pytest.raises(RuntimeError, match="Custom learning-rate"):
        optimizer.set_learning_rate(0.1)
    with pytest.raises(RuntimeError, match="Custom perturbation"):
        optimizer.set_perturbation(0.1)
    optimizer.initialize(np.zeros(1), quadratic, parameter_names=["θ_0"])
    with pytest.raises(RuntimeError, match="before initializing"):
        optimizer.set_parameter_birth_schedules(birth_position_rule=0.4)


def test_birth_schedule_harmonic_default_and_global_ordering():
    optimizer = SPSA(
        parameter_dependent_lr_schedule=True,
        parameter_dependent_perturbation_schedule=True,
    )
    optimizer.set_power_series_hyperparameters(
        a=0.5, alpha=0.602, c=0.1, gamma=0.101
    )
    optimizer.set_parameter_birth_schedules(
        birth_position_rule="harmonic",
        learning_rate_ratio=1.2,
        perturbation_ratio=1.2,
    )
    optimizer.initialize(np.zeros(1), quadratic, parameter_names=["θ_0"])
    optimizer._total_spsa_updates = 100
    optimizer._parameter_schedule_steps["θ_0"] = 100
    optimizer._last_birth_generation["schedule_age"] = 100
    optimizer.initialize(np.zeros(2), quadratic, parameter_names=["θ_0", "θ_1"])
    assert optimizer.last_birth_schedule_event["weight"] == pytest.approx(0.5)
    for age in (0, 10, 100):
        optimizer._parameter_schedule_steps["θ_0"] = 100 + age
        optimizer._parameter_schedule_steps["θ_1"] = age
        lr = optimizer._get_parameter_learning_rates()
        pert = optimizer._get_parameter_perturbations()
        assert lr[1] > lr[0]
        assert pert[1] > pert[0]



def test_previous_generation_reference_uses_restored_age_not_global_clock():
    a, alpha = 0.5, 0.602
    optimizer = SPSA(
        parameter_dependent_lr_schedule=True,
        parameter_dependent_perturbation_schedule=True,
    )
    optimizer.set_power_series_hyperparameters(
        a=a, alpha=alpha, c=0.1, gamma=0.101
    )
    optimizer.set_parameter_birth_schedules(
        learning_rate_reference="previous",
        perturbation_reference="previous",
        birth_position_rule=0.5,
    )
    x = np.zeros(1)
    optimizer.initialize(x, quadratic, parameter_names=["θ_0"])
    optimizer.step(x, quadratic)
    steps = optimizer.parameter_schedule_steps
    births = optimizer.parameter_birth_outer_iterations
    global_births = optimizer.parameter_birth_global_steps
    snapshot = optimizer.birth_schedule_state

    # A trial step consumes a global SPSA update but gets rolled back.
    optimizer.step(x, quadratic)
    assert optimizer._total_spsa_updates == 2
    optimizer.restore_parameter_schedules(
        steps, birth_outer_iterations=births,
        birth_global_steps=global_births, birth_policy_state=snapshot
    )
    assert optimizer.birth_schedule_state == snapshot
    optimizer.initialize(
        np.zeros(2), quadratic, parameter_names=["θ_0", "θ_1"],
        outer_iteration=1,
    )
    event = optimizer.last_birth_schedule_event
    assert event["birth_global_step"] == 2
    assert event["learning_rate"]["reference_current"] == pytest.approx(
        a / 2**alpha
    )
    assert event["perturbation"]["reference_current"] == pytest.approx(
        0.1 / 2**0.101
    )



def test_initial_generation_follows_global_schedule_from_nonzero_start():
    optimizer = SPSA(
        parameter_dependent_lr_schedule=True,
        parameter_dependent_perturbation_schedule=True,
    )
    a, alpha, A, c, gamma = 0.6, 0.602, 10.0, 0.2, 0.101
    optimizer.set_power_series_hyperparameters(
        a=a, alpha=alpha, c=c, gamma=gamma, stability_constant=A
    )
    optimizer.set_parameter_birth_schedules()
    optimizer.initialize(
        np.zeros(1), quadratic, iteration_start=30,
        parameter_names=["θ_0"],
    )
    assert optimizer._get_parameter_learning_rates()[0] == pytest.approx(
        a / (30 + A + 1)**alpha
    )
    assert optimizer._get_parameter_perturbations()[0] == pytest.approx(
        c / 31**gamma
    )
    event = optimizer.last_birth_schedule_event
    assert event["birth_global_step"] == 30
    assert event["generation"] == 0

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

def test_parameter_dependent_lr_with_global_perturbation(monkeypatch):
    optimizer = SPSA(
        parameter_dependent_lr_schedule=True,
        parameter_dependent_perturbation_schedule=False,
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

    # Give the parameters different optimization ages.
    optimizer._parameter_schedule_steps = {
        "θ_0": 3,
        "θ_1": 0,
    }

    monkeypatch.setattr(
        "qadaptive.training.optimizers.spsa.bernoulli_perturbation",
        lambda dim, perturbation_dims=None: np.array([1.0, -1.0]),
    )

    optimizer.step(
        x,
        lambda x: 2.0 * x[0] + 3.0 * x[1],
    )

    # Learning rate follows each parameter's own age.
    assert optimizer.last_parameter_learning_rates == pytest.approx({
        "θ_0": 0.8 / 4**0.5,
        "θ_1": 0.8,
    })

    # Perturbation follows the single global SPSA clock.
    assert optimizer.last_parameter_perturbations == pytest.approx({
        "θ_0": 0.2,
        "θ_1": 0.2,
    })
    
def test_global_lr_with_parameter_dependent_perturbation(monkeypatch):
    optimizer = SPSA(
        parameter_dependent_lr_schedule=False,
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
        "θ_0": 3,
        "θ_1": 0,
    }

    monkeypatch.setattr(
        "qadaptive.training.optimizers.spsa.bernoulli_perturbation",
        lambda dim, perturbation_dims=None: np.array([1.0, -1.0]),
    )

    optimizer.step(
        x,
        lambda x: 2.0 * x[0] + 3.0 * x[1],
    )

    # Learning rate follows one global SPSA clock.
    assert optimizer.last_parameter_learning_rates == pytest.approx({
        "θ_0": 0.8,
        "θ_1": 0.8,
    })

    # Perturbation follows each parameter's own age.
    assert optimizer.last_parameter_perturbations == pytest.approx({
        "θ_0": 0.2 / 4**0.25,
        "θ_1": 0.2,
    })

def test_global_perturbation_continues_when_new_parameter_is_added(
    monkeypatch,
):
    optimizer = SPSA(
        parameter_dependent_lr_schedule=True,
        parameter_dependent_perturbation_schedule=False,
    )

    optimizer.set_power_series_hyperparameters(
        a=0.8,
        alpha=0.5,
        c=0.2,
        gamma=0.25,
        stability_constant=0.0,
    )

    monkeypatch.setattr(
        "qadaptive.training.optimizers.spsa.bernoulli_perturbation",
        lambda dim, perturbation_dims=None: np.ones(dim),
    )

    # First architecture.
    x = np.zeros(2)

    optimizer.initialize(
        x,
        quadratic,
        parameter_names=["θ_0", "θ_1"],
        outer_iteration=0,
    )

    optimizer.step(x, quadratic)

    # Structural growth: θ_2 is newly born.
    x_grown = np.zeros(3)

    optimizer.initialize(
        x_grown,
        quadratic,
        iteration_start=None,
        parameter_names=["θ_0", "θ_1", "θ_2"],
        outer_iteration=1,
    )

    assert optimizer.parameter_schedule_steps == {
        "θ_0": 1,
        "θ_1": 1,
        "θ_2": 0,
    }

    optimizer.step(x_grown, quadratic)

    expected_global_perturbation = 0.2 / 2**0.25

    assert optimizer.last_parameter_perturbations == pytest.approx({
        "θ_0": expected_global_perturbation,
        "θ_1": expected_global_perturbation,
        "θ_2": expected_global_perturbation,
    })

    # But the new parameter still gets the beginning of its LR schedule.
    assert optimizer.last_parameter_learning_rates == pytest.approx({
        "θ_0": 0.8 / 2**0.5,
        "θ_1": 0.8 / 2**0.5,
        "θ_2": 0.8,
    })

def test_restoring_schedules_discards_rejected_birth_steps_without_rewinding_clock():
    optimizer = SPSA(
        learning_rate=0.1,
        perturbation=0.1,
        parameter_dependent_lr_schedule=True,
        parameter_dependent_perturbation_schedule=True,
    )
    x = np.zeros(1)
    optimizer.initialize(x, quadratic, parameter_names=["θ_0"], outer_iteration=0)
    optimizer.step(x, quadratic)

    steps_before = optimizer.parameter_schedule_steps
    births_before = optimizer.parameter_birth_outer_iterations
    global_births_before = optimizer.parameter_birth_global_steps

    optimizer.initialize(
        np.zeros(2),
        quadratic,
        parameter_names=["θ_0", "θ_temp"],
        outer_iteration=1,
    )
    assert optimizer.parameter_birth_global_steps["θ_temp"] == 1
    optimizer.step(np.zeros(2), quadratic)
    assert optimizer._total_spsa_updates == 2

    optimizer.restore_parameter_schedules(
        steps_before,
        birth_outer_iterations=births_before,
        birth_global_steps=global_births_before,
    )

    assert optimizer.parameter_schedule_steps == steps_before
    assert optimizer.parameter_birth_outer_iterations == births_before
    assert optimizer.parameter_birth_global_steps == global_births_before
    assert optimizer._total_spsa_updates == 2

    with pytest.raises(ValueError, match="same parameter names"):
        optimizer.restore_parameter_schedules(
            steps_before,
            birth_outer_iterations=births_before,
            birth_global_steps={"θ_temp": 1},
        )

