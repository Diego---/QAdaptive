import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter

from qadaptive.core.adaptive_ansatz import AdaptiveAnsatz
from qadaptive.outer.action_definitions import INSERT_GATE
from qadaptive.outer.mutable_ansatz_experiment import MutableAnsatzExperiment
from qadaptive.outer.outer_loop import ActionSpec, OuterStepPlan
from qadaptive.training.optimizers import SPSA
from qadaptive.training.recorder import InnerLoopRecorder
from qadaptive.training.trainer import InnerLoopTrainer


def quadratic_loss(params, ansatz, **kwargs):
    del ansatz, kwargs
    return float(np.sum(np.asarray(params, dtype=float) ** 2))


def append_rx_plan(experiment):
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
    
def make_experiment():
    circuit = QuantumCircuit(1)
    circuit.rx(Parameter("θ_0"), 0)

    recorder = InnerLoopRecorder(record_initial_value=True)
    optimizer = SPSA(learning_rate=0.1, perturbation=0.1)
    trainer = InnerLoopTrainer(
        optimizer=optimizer,
        recorder=recorder,
    )

    experiment = MutableAnsatzExperiment(
        adaptive_ansatz=AdaptiveAnsatz(circuit),
        trainer=trainer,
    )

    return experiment


def test_outer_loop_annotates_persistent_recorder_runs():
    circuit = QuantumCircuit(1)
    circuit.rx(Parameter("θ_0"), 0)

    recorder = InnerLoopRecorder(record_initial_value=True)
    optimizer = SPSA(learning_rate=0.1, perturbation=0.1)
    trainer = InnerLoopTrainer(optimizer=optimizer, recorder=recorder)
    experiment = MutableAnsatzExperiment(
        adaptive_ansatz=AdaptiveAnsatz(circuit),
        trainer=trainer,
    )

    results = experiment.run_outer_loop(
        loss_function=quadratic_loss,
        plan_schedule=[append_rx_plan],
        outer_iterations=1,
        train_iterations=1,
        initial_point=[0.5],
        reuse_parameter_memory=True,
    )

    assert len(results) == 1
    assert results[0].accepted is True
    assert experiment.recorder is recorder
    assert len(recorder.runs) == 2

    initial_run, proposal_run = recorder.runs
    assert initial_run.outer_iteration == 0
    assert initial_run.action == "Initial training before first plan"
    assert initial_run.accepted_outer_step is True
    assert initial_run.initial_value is not None

    assert proposal_run.outer_iteration == 1
    assert proposal_run.action == "append_rx"
    assert proposal_run.accepted_outer_step is True
    assert len(proposal_run.param_names) == 2

def test_outer_termination_checker_false_continues_before_builder():
    experiment = make_experiment()
    events = []

    def checker(exp):
        events.append(("check", exp.outer_iteration))
        return False

    def builder(exp):
        events.append(("build", exp.outer_iteration))
        return append_rx_plan(exp)

    results = experiment.run_outer_loop(
        loss_function=quadratic_loss,
        plan_schedule=[builder],
        outer_iterations=1,
        outer_termination_checker=checker,
        train_iterations=1,
        initial_point=[0.5],
        reuse_parameter_memory=True,
    )

    assert len(results) == 1
    assert events == [
        ("check", 1),
        ("build", 1),
    ]

def test_outer_termination_checker_stops_before_plan_builder():
    experiment = make_experiment()
    events = []

    def checker(exp):
        events.append("check")
        return True

    def builder(exp):
        events.append("build")
        return append_rx_plan(exp)

    results = experiment.run_outer_loop(
        loss_function=quadratic_loss,
        plan_schedule=[builder],
        outer_iterations=3,
        outer_termination_checker=checker,
        train_before_first_plan=False,
    )

    assert results == []
    assert events == ["check"]
    assert experiment.outer_iteration == 0
    assert experiment.outer_step_history == []

def test_plan_builder_none_terminates_outer_loop():
    experiment = make_experiment()
    calls = []

    def builder(exp):
        calls.append(exp.outer_iteration)
        return None

    results = experiment.run_outer_loop(
        loss_function=quadratic_loss,
        plan_schedule=[builder],
        outer_iterations=3,
        train_before_first_plan=False,
    )

    assert results == []
    assert calls == [0]
    assert experiment.outer_iteration == 0
    assert experiment.outer_step_history == []

def test_plan_builder_none_after_step_does_not_create_history_entry():
    experiment = make_experiment()
    calls = 0

    def builder(exp):
        nonlocal calls
        calls += 1

        if calls == 1:
            return append_rx_plan(exp)

        return None

    results = experiment.run_outer_loop(
        loss_function=quadratic_loss,
        plan_schedule=[builder],
        outer_iterations=5,
        train_iterations=1,
        initial_point=[0.5],
        reuse_parameter_memory=True,
    )

    assert len(results) == 1
    assert calls == 2

    # Initial training + one actual structural step.
    assert len(experiment.outer_step_history) == 2

    # Initial training advances to 1; the accepted structural step advances to 2.
    # Returning None must not advance it again.
    assert experiment.outer_iteration == 2

    # Only one structural proposal was actually attempted.
    assert len(experiment.trial_ansatz_history) == 1

def test_outer_termination_checker_can_stop_after_completed_step():
    experiment = make_experiment()

    def checker(exp):
        return exp.outer_iteration >= 2

    results = experiment.run_outer_loop(
        loss_function=quadratic_loss,
        plan_schedule=[append_rx_plan],
        outer_iterations=5,
        outer_termination_checker=checker,
        train_iterations=1,
        initial_point=[0.5],
        reuse_parameter_memory=True,
    )

    assert len(results) == 1
    assert experiment.outer_iteration == 2
