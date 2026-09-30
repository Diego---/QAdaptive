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
