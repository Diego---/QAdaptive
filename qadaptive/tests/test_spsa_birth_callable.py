
import json
import math

import numpy as np
import pytest

from dataclasses import dataclass
from qadaptive.training.optimizers import SPSA


@dataclass(frozen=True)
class DelayedLinearRamp:
    """One-parameter delayed linear birth-weight schedule.

    w(j) = min(0.25, 0.005 + slope * max(0, j - 3))

    j: insertion number, starting at 1
    k: cumulative global SPSA iteration at insertion
    """

    slope: float

    def __post_init__(self):
        if (
            isinstance(self.slope, bool)
            or not isinstance(self.slope, (int, float))
        ):
            raise TypeError("slope must be a real number")

        if not math.isfinite(self.slope) or self.slope < 0:
            raise ValueError("slope must be finite and >= 0")

    def __call__(self, j: int, k: int) -> float:
        if isinstance(j, bool) or not isinstance(j, int) or j < 1:
            raise ValueError("j must be a positive integer")

        if isinstance(k, bool) or not isinstance(k, int) or k < 0:
            raise ValueError("k must be a non-negative integer")

        return float(
            min(0.25, 0.005 + self.slope * max(0, j - 3))
        )

    def birth_policy_spec(self) -> dict:
        """JSON-compatible description of this policy."""
        return {
            "type": "delayed_linear_ramp",
            "slope": float(self.slope),
            "early_weight": 0.005,
            "onset": 3,
            "maximum": 0.25,
        }



def objective(x):
    return float(np.sum(np.asarray(x, dtype=float)**2))


def make_optimizer():
    optimizer = SPSA(
        parameter_dependent_lr_schedule=True,
        parameter_dependent_perturbation_schedule=True,
    )
    optimizer.set_power_series_hyperparameters(
        a=0.4,
        alpha=0.602,
        c=0.1,
        gamma=0.101,
        stability_constant=0.0,
    )
    return optimizer


def test_delayed_linear_ramp():
    rule = DelayedLinearRamp(slope=0.20)

    weights = [
        rule(j, 50 * j)
        for j in range(1, 6)
    ]

    assert weights == pytest.approx([
        0.005, 0.005, 0.005, 0.205, 0.25
    ])

    assert rule(10, 500) == pytest.approx(0.25)

    with pytest.raises(ValueError):
        DelayedLinearRamp(slope=-0.1)


def test_callable_invoked_once_per_insertion():
    calls = []

    def rule(j, k):
        calls.append((j, k))
        return 0.1

    optimizer = make_optimizer()

    optimizer.set_parameter_birth_schedules(
        birth_position_rule=rule,
    )

    optimizer.initialize(
        np.zeros(1),
        objective,
        parameter_names=["theta_0"],
    )

    # The initial parameter batch must not consume a weight.
    assert calls == []

    optimizer._total_spsa_updates = 50
    optimizer._last_birth_generation["schedule_age"] = 50

    optimizer.initialize(
        np.zeros(3),
        objective,
        parameter_names=["theta_0", "theta_1", "theta_2"],
    )

    # Two new parameters, but only one insertion.
    assert calls == [(1, 50)]

    # Reinitialization without growth must not invoke the rule.
    optimizer.initialize(
        np.zeros(3),
        objective,
        parameter_names=["theta_0", "theta_1", "theta_2"],
    )

    assert calls == [(1, 50)]

    optimizer._total_spsa_updates = 100
    optimizer._last_birth_generation["schedule_age"] = 50

    optimizer.initialize(
        np.zeros(4),
        objective,
        parameter_names=[
            "theta_0", "theta_1", "theta_2", "theta_3"
        ],
    )

    assert calls == [(1, 50), (2, 100)]


def test_callable_configuration_is_serializable():
    optimizer = make_optimizer()

    optimizer.set_parameter_birth_schedules(
        birth_position_rule=DelayedLinearRamp(0.20),
    )

    config = optimizer.settings["parameter_birth_schedules"]

    assert config["birth_position_rule"]["type"] == (
        "delayed_linear_ramp"
    )
    assert config["birth_position_rule"]["slope"] == 0.20

    json.dumps(config)

    optimizer.initialize(
        np.zeros(1),
        objective,
        parameter_names=["theta_0"],
    )

    optimizer._total_spsa_updates = 50
    optimizer._last_birth_generation["schedule_age"] = 50

    optimizer.initialize(
        np.zeros(2),
        objective,
        parameter_names=["theta_0", "theta_1"],
    )

    event = optimizer.last_birth_schedule_event

    assert event["weight"] == pytest.approx(0.005)
    json.dumps(event)


@pytest.mark.parametrize(
    "bad", [-0.2, 1.1, float("nan"), float("inf"), True, "invalid"]
)
def test_invalid_callable_weights(bad):
    optimizer = make_optimizer()

    optimizer.set_parameter_birth_schedules(
        birth_position_rule=lambda j, k: bad
    )

    optimizer.initialize(
        np.zeros(1), objective, parameter_names=["theta_0"]
    )

    optimizer._total_spsa_updates = 20
    optimizer._last_birth_generation["schedule_age"] = 20

    with pytest.raises((TypeError, ValueError)):
        optimizer.initialize(
            np.zeros(2),
            objective,
            parameter_names=["theta_0", "theta_1"],
        )
