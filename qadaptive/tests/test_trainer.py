import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit_algorithms.optimizers.optimizer import (
    OptimizerResult,
    OptimizerSupportLevel,
)

from qadaptive.training.trainer import InnerLoopTrainer
from qadaptive.training.stepwise_optimizer import StepwiseOptimizer


class DummyOptimizer(StepwiseOptimizer):
    """
    Minimal stepwise optimizer for trainer tests.

    The optimizer uses a constant dummy gradient of all ones and applies a
    fixed-size gradient-descent step.
    """

    def __init__(self, step_size: float = 0.1):
        super().__init__()
        self._step_size = step_size

    @property
    def settings(self) -> dict:
        """Return optimizer settings."""
        return {"step_size": self._step_size}

    def get_support_level(self) -> dict[str, OptimizerSupportLevel]:
        """Return support levels for optimizer features."""
        return {
            "gradient": OptimizerSupportLevel.ignored,
            "bounds": OptimizerSupportLevel.ignored,
            "initial_point": OptimizerSupportLevel.supported,
        }

    def initialize(
        self,
        x0: np.ndarray,
        loss_function,
        iteration_start: int | None = None,
        **kwargs,
    ) -> None:
        del x0, loss_function, kwargs

        if iteration_start is None:
            iteration_start = 0

        self.reset_runtime_state(iteration_start=iteration_start)
        self._initialized = True

    def step(
        self,
        x: np.ndarray,
        loss_function,
        loss_next=None,
        **kwargs,
    ) -> tuple[bool, np.ndarray, float | None, np.ndarray | None, float | None]:
        del loss_next

        x = np.asarray(x, dtype=float)
        fx_estimate = float(loss_function(x, **kwargs))
        gradient_estimate = np.ones_like(x, dtype=float)
        x_next = x - self._step_size * gradient_estimate

        self._nfev += 1
        self._iteration += 1
        self._last_gradient = gradient_estimate.copy()
        self._last_fx = fx_estimate
        self._last_stepsize = float(np.linalg.norm(x_next - x))

        return False, x_next, None, gradient_estimate, fx_estimate

    def minimize(
        self,
        fun,
        x0,
        jac=None,
        bounds=None,
        **kwargs,
    ) -> OptimizerResult:
        """
        Minimal compatibility implementation of the Qiskit optimizer interface.
        """
        del jac, bounds

        x = np.asarray(x0, dtype=float)
        self.initialize(x, fun, **kwargs)
        skip, x_next, fx_next, gradient_estimate, fx_estimate = self.step(
            x,
            fun,
            **kwargs,
        )

        result = OptimizerResult()
        result.x = x if skip else x_next
        result.fun = float(fx_estimate if fx_next is None else fx_next)
        result.nfev = self.nfev
        result.nit = self.iteration
        return result

def build_parameterized_ansatz() -> QuantumCircuit:
    th = ParameterVector("t", 2)
    qc = QuantumCircuit(1)
    qc.rx(th[0], 0)
    qc.rz(th[1], 0)
    return qc


def quadratic_loss(x, ansatz, **kwargs):
    del ansatz, kwargs
    return float(np.sum(np.asarray(x, dtype=float) ** 2))


def test_inner_loop_trainer_requires_optimizer_or_options():
    with pytest.raises(ValueError):
        InnerLoopTrainer(optimizer=None, optimizer_options=None)


def test_update_last_evaluation_updates_cost_and_optionally_params():
    trainer = InnerLoopTrainer(optimizer=DummyOptimizer())
    trainer.update_last_evaluation(cost=1.5, params=[0.2, -0.1])

    assert trainer.last_cost == 1.5
    assert np.allclose(trainer.last_params, np.array([0.2, -0.1]))


def test_set_optimizer_replaces_optimizer_instance():
    trainer = InnerLoopTrainer(optimizer=DummyOptimizer(step_size=0.1))
    new_optimizer = DummyOptimizer(step_size=0.3)

    trainer.set_optimizer(optimizer=new_optimizer)

    assert trainer.optimizer is new_optimizer


def test_step_advances_optimizer_iteration_when_update_is_applied():
    trainer = InnerLoopTrainer(optimizer=DummyOptimizer(step_size=0.2))
    ansatz = build_parameterized_ansatz()
    x = np.array([1.0, -1.0])

    skip, x_next, fx_next, gradient_estimate, fx_estimate = trainer.step(
        ansatz=ansatz,
        loss_function=quadratic_loss,
        x=x,
    )

    assert not skip
    assert fx_next is None
    assert np.allclose(gradient_estimate, np.ones_like(x))
    assert fx_estimate == pytest.approx(2.0)
    assert np.allclose(x_next, np.array([0.8, -1.2]))
    assert trainer.optimizer.iteration == 1


def test_train_one_time_returns_result_and_tracks_gradients():
    optimizer = DummyOptimizer(step_size=0.1)
    trainer = InnerLoopTrainer(optimizer=optimizer, track_gradients=True)
    ansatz = build_parameterized_ansatz()

    result = trainer.train_one_time(
        ansatz=ansatz,
        loss_function=quadratic_loss,
        initial_point=np.array([1.0, 1.0]),
        iterations=2,
    )

    assert np.allclose(result.x, np.array([0.8, 0.8]))
    assert result.fun == pytest.approx(1.28)
    assert result.nit == 2
    assert trainer.last_cost == pytest.approx(1.28)
    assert np.allclose(trainer.last_params, np.array([0.8, 0.8]))
    assert len(trainer.gradient_history[0]) == 2
    assert trainer.gradient_history[1] == []


def test_train_one_time_records_training_run_history_when_requested():
    optimizer = DummyOptimizer(step_size=0.1)
    trainer = InnerLoopTrainer(optimizer=optimizer, track_gradients=False)
    ansatz = build_parameterized_ansatz()

    result = trainer.train_one_time(
        ansatz=ansatz,
        loss_function=quadratic_loss,
        initial_point=np.array([1.0, 1.0]),
        iterations=2,
        record_run_history=True,
        initial_value=2.0,
    )

    record = trainer.last_training_run_record

    assert record is not None
    assert len(trainer.training_run_history) == 1
    assert trainer.training_run_history[0] is record

    assert record.run_index == 0
    assert record.param_names == [p.name for p in ansatz.parameters]
    assert np.allclose(record.initial_point, np.array([1.0, 1.0]))
    assert record.initial_value == pytest.approx(2.0)
    assert record.final_value == pytest.approx(result.fun)

    assert len(record.iterations) == 2

    assert record.iterations[0].iteration == 1
    assert np.allclose(record.iterations[0].params, np.array([0.9, 0.9]))
    assert record.iterations[0].value == pytest.approx(2.0)

    assert record.iterations[1].iteration == 2
    assert np.allclose(record.iterations[1].params, np.array([0.8, 0.8]))
    assert record.iterations[1].value == pytest.approx(1.62)


def test_train_one_time_without_history_leaves_last_training_run_record_none():
    optimizer = DummyOptimizer(step_size=0.1)
    trainer = InnerLoopTrainer(optimizer=optimizer, track_gradients=False)
    ansatz = build_parameterized_ansatz()

    trainer.train_one_time(
        ansatz=ansatz,
        loss_function=quadratic_loss,
        initial_point=np.array([1.0, 1.0]),
        iterations=1,
        record_run_history=False,
    )

    assert trainer.last_training_run_record is None
    assert trainer.training_run_history == []
