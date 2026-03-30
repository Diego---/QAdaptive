import itertools
import sys
import types

import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector

from qadaptive.adaptive_ansatz import AdaptiveAnsatz
from qae.optimization.my_spsa import SPSA
from qadaptive.trainer import InnerLoopTrainer

class DummyOptimizer:
    def __init__(self, step_size: float = 0.1):
        self.blocking = False
        self.p_iterator = None
        self.lr_iterator = None
        self._step_size = step_size
        self._lr_iterator_copy = itertools.repeat(step_size)
        self.callback = None
        self.termination_checker = None
        self._nfev = 0
        self._nextfev = 0
        self.last_iteration = 0
        self._size_full_batch = None

    def _create_iterators(self, loss_function, initial_point):
        self.p_iterator = object()
        self.lr_iterator = object()
        self._lr_iterator_copy = itertools.repeat(self._step_size)

    def compute_loss_and_gradient_estimate(self, loss_function, x, **kwargs):
        self._nfev += 1
        return float(loss_function(x, **kwargs)), np.ones_like(x)

    def process_update(
        self,
        gradient_estimate,
        x,
        fx_estimate,
        loss_next,
        iteration_start,
        inner_iteration,
    ):
        x_next = x - self._step_size * gradient_estimate
        return False, x_next, None



def build_parameterized_ansatz() -> QuantumCircuit:
    th = ParameterVector("t", 2)
    qc = QuantumCircuit(1)
    qc.rx(th[0], 0)
    qc.rz(th[1], 0)
    return qc



def quadratic_loss(x, ansatz, **kwargs):
    return float(np.sum(np.asarray(x, dtype=float) ** 2))



def test_inner_loop_trainer_requires_optimizer_or_options():
    with pytest.raises(AssertionError):
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
    assert trainer.optimizer.last_iteration == 1



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
