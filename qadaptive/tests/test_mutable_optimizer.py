import pytest

import numpy as np

from qiskit.circuit import QuantumCircuit, ParameterVector

from qadaptive.training import SPSA
from qadaptive.core.adaptive_ansatz import AdaptiveAnsatz
from qadaptive.training.trainer import InnerLoopTrainer
from qadaptive.outer.action_definitions import PRUNE_TWO_QUBIT
from qadaptive.outer.mutable_ansatz_experiment import MutableAnsatzExperiment
from qadaptive.outer.outer_loop import ActionSpec, OuterStepPlan, OuterStepResult

th = ParameterVector("t", 10)
tiny_ansatz = QuantumCircuit(3)
tiny_ansatz.rx(th[0], 0)
tiny_ansatz.rx(th[1], 1)
tiny_ansatz.rx(th[2], 2)
tiny_ansatz.cz(0, 1)
tiny_ansatz.cz(1, 2)
tiny_ansatz.rx(th[3], 0)
tiny_ansatz.rx(th[4], 1)
tiny_ansatz.rx(th[5], 2)
simple_ansatz = AdaptiveAnsatz.from_generic_circuit(tiny_ansatz)

simple_trainer = InnerLoopTrainer(SPSA())

def quadratic(x, ansatz=None):
    x = np.asarray(x, dtype=float)
    return float(np.sum(x**2))

def make_parameter_schedule_experiment():
    params = ParameterVector("t", 2)

    circuit = QuantumCircuit(2)
    circuit.rx(params[0], 0)
    circuit.rx(params[1], 1)
    circuit.cz(0, 1)

    adaptive_ansatz = AdaptiveAnsatz.from_generic_circuit(circuit)

    optimizer = SPSA(
        learning_rate=0.1,
        perturbation=0.1,
        parameter_dependent_lr_schedule=True,
        parameter_dependent_perturbation_schedule=True,
    )

    trainer = InnerLoopTrainer(optimizer)

    experiment = MutableAnsatzExperiment(
        adaptive_ansatz,
        trainer,
    )

    return experiment, optimizer

def operation_names(circuit: QuantumCircuit) -> list[str]:
    return [inst.operation.name for inst in circuit.data]

def test_muable_optimizer_initialization():
    """Test that MutableOptimizer correctly initializes."""
    mo = MutableAnsatzExperiment(simple_ansatz, simple_trainer)
    assert isinstance(mo, MutableAnsatzExperiment)
    assert isinstance(mo.trainer, InnerLoopTrainer)
    assert isinstance(mo.ansatz, QuantumCircuit)

def test_insert_at():
    mo = MutableAnsatzExperiment(simple_ansatz, simple_trainer)
    mo.insert_at('cz', [0, 1], 0)
    mo.insert_at('cz', [1, 2], 0)
    mo.insert_at('rx', [0], 3)
    mo.insert_at('ry', [0], 3)
    mo.insert_at('rx', [0], 3)
    mo.insert_at('ry', [0], 3)
    mo.insert_at('rz', [0], 2)
    
    with pytest.raises(AssertionError):
        mo.insert_at('h', [0], 0)

    assert len(mo.ansatz.data) > len(tiny_ansatz.data)
    
def test_mutable_ansatz_experiment_copies_input_ansatz_before_mutation():
    adaptive = simple_ansatz.copy()
    original_ops = operation_names(adaptive.current_ansatz)

    experiment = MutableAnsatzExperiment(adaptive, simple_trainer)
    experiment.insert_at("rx", [0], 0)

    assert operation_names(adaptive.current_ansatz) == original_ops
    assert len(experiment.ansatz.data) == len(adaptive.current_ansatz.data) + 1
    
def test_insert_block_at_rejects_wrong_qubit_count():
    ansatz = simple_ansatz.copy()
    experiment = MutableAnsatzExperiment(ansatz, simple_trainer)

    with pytest.raises(ValueError):
        experiment.insert_block_at("cx_identity", [0], 0)
        
def test_lock_gates_marks_requested_two_qubit_gates():
    ansatz = simple_ansatz.copy()
    experiment = MutableAnsatzExperiment(ansatz, simple_trainer)
    two_q_indices = sorted(experiment._2qbg_positions)

    experiment.lock_gates([two_q_indices[0]])

    assert experiment._is_locked_circuit_index(two_q_indices[0])
    assert experiment._get_locked_circuit_indices() == [two_q_indices[0]]

def test_insert_random():
    pass

def test_simplify_methods():
    pass

def test_insert_block_at():
    mo = MutableAnsatzExperiment(simple_ansatz, simple_trainer)
    old_len = len(mo.ansatz.data)
    old_num_params = len(mo.ansatz.parameters)

    mo.insert_block_at("rz_rx_rz", [0], 0)

    assert len(mo.ansatz.data) == old_len + 3
    assert len(mo.ansatz.parameters) == old_num_params + 3

def test_insert_two_qubit_block_at():
    mo = MutableAnsatzExperiment(simple_ansatz, simple_trainer)
    old_len = len(mo.ansatz.data)
    old_num_params = len(mo.ansatz.parameters)

    mo.insert_block_at("cx_identity", [0, 1], 0)

    assert len(mo.ansatz.data) == old_len + 6
    assert len(mo.ansatz.parameters) == old_num_params + 4

def test_get_pair_occurrence_from_circuit_index():
    mo = MutableAnsatzExperiment(simple_ansatz, simple_trainer)
    mo._2qbg_positions = {
        2: (0, 1),
        5: (0, 1),
        9: (2, 3),
        12: (2, 1),
    }

    assert mo._get_pair_occurrence_from_circuit_index(2) == (0, (0, 1))
    assert mo._get_pair_occurrence_from_circuit_index(5) == (1, (0, 1))
    assert mo._get_pair_occurrence_from_circuit_index(9) == (0, (2, 3))
    assert mo._get_pair_occurrence_from_circuit_index(12) == (0, (2, 1))
    
def test_get_pair_occurrence_from_circuit_index_invalid():
    mo = MutableAnsatzExperiment(simple_ansatz, simple_trainer)
    mo._2qbg_positions = {
        2: (0, 1),
    }

    with pytest.raises(KeyError):
        mo._get_pair_occurrence_from_circuit_index(3)
        
def test_get_circuit_index_from_pair_occurrence():
    mo = MutableAnsatzExperiment(simple_ansatz, simple_trainer)
    mo._2qbg_positions = {
        2: (0, 1),
        5: (0, 1),
        9: (2, 3),
        12: (2, 1),
    }

    assert mo._get_circuit_index_from_pair_occurrence(0, (0, 1)) == 2
    assert mo._get_circuit_index_from_pair_occurrence(1, (0, 1)) == 5
    assert mo._get_circuit_index_from_pair_occurrence(0, (2, 3)) == 9
    assert mo._get_circuit_index_from_pair_occurrence(0, (2, 1)) == 12
    
def test_get_circuit_index_from_pair_occurrence_missing():
    mo = MutableAnsatzExperiment(simple_ansatz, simple_trainer)
    mo._2qbg_positions = {
        2: (0, 1),
        5: (0, 1),
    }

    assert mo._get_circuit_index_from_pair_occurrence(2, (0, 1)) is None
    assert mo._get_circuit_index_from_pair_occurrence(0, (2, 3)) is None
    assert mo._get_circuit_index_from_pair_occurrence(-1, (0, 1)) is None
    
def test_pair_occurrence_and_circuit_index_are_inverse():
    mo = MutableAnsatzExperiment(simple_ansatz, simple_trainer)
    mo._2qbg_positions = {
        2: (0, 1),
        5: (0, 1),
        9: (2, 3),
        12: (2, 1),
    }

    for circ_index in mo._2qbg_positions:
        occ, pair = mo._get_pair_occurrence_from_circuit_index(circ_index)
        recovered = mo._get_circuit_index_from_pair_occurrence(occ, pair)
        assert recovered == circ_index

def test_lock_and_check_circuit_index():
    mo = MutableAnsatzExperiment(simple_ansatz, simple_trainer)
    mo._2qbg_positions = {
        2: (0, 1),
        5: (0, 1),
        9: (2, 3),
    }
    mo.locked_gates = set()

    mo._lock_circuit_index(5)

    assert (1, (0, 1)) in mo.locked_gates
    assert mo._is_locked_circuit_index(5)
    assert not mo._is_locked_circuit_index(2)
    assert not mo._is_locked_circuit_index(9)
    
def test_get_locked_circuit_indices():
    mo = MutableAnsatzExperiment(simple_ansatz, simple_trainer)
    mo._2qbg_positions = {
        2: (0, 1),
        5: (0, 1),
        9: (2, 3),
        12: (2, 1),
    }
    mo.locked_gates = {
        (1, (0, 1)),
        (0, (2, 1)),
    }

    assert mo._get_locked_circuit_indices() == [5, 12]
    
def test_pair_occurrence_and_circuit_index_are_inverse():
    mo = MutableAnsatzExperiment(simple_ansatz, simple_trainer)
    mo._2qbg_positions = {
        2: (0, 1),
        5: (0, 1),
        9: (2, 3),
        12: (2, 1),
    }

    for circ_index in mo._2qbg_positions:
        occ, pair = mo._get_pair_occurrence_from_circuit_index(circ_index)
        recovered = mo._get_circuit_index_from_pair_occurrence(occ, pair)
        assert recovered == circ_index

def test_remove_at_locked_two_qubit_gate_does_nothing():
    mo = MutableAnsatzExperiment(simple_ansatz, simple_trainer)
    mo.lock_gates([2])

    original_len = len(mo.ansatz.data)
    original_locked = mo.locked_gates.copy()

    mo.remove_at(2)

    assert len(mo.ansatz.data) == original_len
    assert mo.locked_gates == original_locked

def test_restore_parameter_schedules_restores_steps_and_active_parameters():
    optimizer = SPSA(
        learning_rate=0.1,
        perturbation=0.1,
        parameter_dependent_lr_schedule=True,
        parameter_dependent_perturbation_schedule=True,
    )

    optimizer.initialize(
        np.zeros(2),
        quadratic,
        parameter_names=["θ_0", "θ_1"],
    )

    optimizer._parameter_schedule_steps = {
        "θ_0": 12,
        "θ_1": 7,
    }

    snapshot = optimizer.parameter_schedule_steps

    optimizer.initialize(
        np.zeros(3),
        quadratic,
        parameter_names=["θ_0", "θ_1", "θ_2"],
    )

    optimizer._advance_parameter_schedule_steps()

    optimizer.restore_parameter_schedules(snapshot)

    assert optimizer.parameter_schedule_steps == {
        "θ_0": 12,
        "θ_1": 7,
    }

    assert optimizer._active_parameter_names == (
        "θ_0",
        "θ_1",
    )

def test_successful_pruning_restarts_parameter_schedules_when_requested():
    experiment, optimizer = make_parameter_schedule_experiment()

    parameter_names = [
        parameter.name
        for parameter in experiment.adaptive_ansatz.params
    ]

    optimizer.initialize(
        np.zeros(2),
        quadratic,
        parameter_names=parameter_names,
    )

    optimizer.restore_parameter_schedules({
        parameter_names[0]: 8,
        parameter_names[1]: 8,
    })

    prune_index = next(iter(experiment._2qbg_positions))

    plan = OuterStepPlan(
        name="prune",
        actions=[
            ActionSpec(
                action=PRUNE_TWO_QUBIT,
                kwargs={"gate_to_remove": prune_index},
            )
        ],
        acceptance_mode="force",
    )

    result = experiment.run_outer_step(
        loss_function=quadratic,
        plan=plan,
        train_iterations=1,
        initial_point_generator=lambda _: np.zeros(2),
        restart_parameter_schedules_after_pruning=True,
    )

    assert result.accepted

    assert optimizer.parameter_schedule_steps == {
        parameter_names[0]: 1,
        parameter_names[1]: 1,
    }

    assert len(experiment._2qbg_positions) == 0

def test_skipped_pruning_does_not_restart_parameter_schedules():
    experiment, optimizer = make_parameter_schedule_experiment()

    parameter_names = [
        parameter.name
        for parameter in experiment.adaptive_ansatz.params
    ]

    optimizer.initialize(
        np.zeros(2),
        quadratic,
        parameter_names=parameter_names,
    )

    optimizer.restore_parameter_schedules({
        parameter_names[0]: 8,
        parameter_names[1]: 8,
    })

    plan = OuterStepPlan(
        name="skipped_prune",
        actions=[
            ActionSpec(
                action=PRUNE_TWO_QUBIT,
                kwargs={
                    "target_pair": (0, 1),
                    "target_occurrence": 99,
                },
            )
        ],
        acceptance_mode="force",
    )

    result = experiment.run_outer_step(
        loss_function=quadratic,
        plan=plan,
        train_iterations=1,
        initial_point_generator=lambda _: np.zeros(2),
        restart_parameter_schedules_after_pruning=True,
    )

    assert result.accepted

    assert optimizer.parameter_schedule_steps == {
        parameter_names[0]: 9,
        parameter_names[1]: 9,
    }

    assert len(experiment._2qbg_positions) == 1

def test_rejected_pruning_restores_parameter_schedule_steps(monkeypatch):
    experiment, optimizer = make_parameter_schedule_experiment()

    parameter_names = [
        parameter.name
        for parameter in experiment.adaptive_ansatz.params
    ]

    # Establish valid trainer state for the outer-loop snapshot.
    experiment.train_one_time(
        loss_function=quadratic,
        initial_point=np.zeros(2),
        iterations=1,
    )

    # Define the schedule state that should survive the rejected trial.
    optimizer.restore_parameter_schedules({
        parameter_names[0]: 8,
        parameter_names[1]: 8,
    })

    # run_outer_step() automatically accepts the first outer proposal if no
    # previous outer-step baseline exists. Supply a minimal previous result so
    # that the normal acceptance path is exercised.
    experiment.outer_step_history.append(
        OuterStepResult(
            iteration=-1,
            action="baseline",
            accepted=True,
            cost_before=None,
            cost_after=float(experiment.last_cost),
            delta_cost=None,
            num_parameters_before=2,
            num_parameters_after=2,
            num_two_qubit_before=1,
            num_two_qubit_after=1,
        )
    )

    # We are testing rollback semantics here, not the acceptance criterion.
    monkeypatch.setattr(
        experiment,
        "_accept_outer_step",
        lambda **kwargs: False,
    )

    prune_index = next(iter(experiment._2qbg_positions))

    plan = OuterStepPlan(
        name="rejected_prune",
        actions=[
            ActionSpec(
                action=PRUNE_TWO_QUBIT,
                kwargs={"gate_to_remove": prune_index},
            )
        ],
        acceptance_mode="outer",
    )

    result = experiment.run_outer_step(
        loss_function=quadratic,
        plan=plan,
        train_iterations=1,
        initial_point_generator=lambda _: np.zeros(2),
        restart_parameter_schedules_after_pruning=True,
    )

    assert not result.accepted

    assert optimizer.parameter_schedule_steps == {
        parameter_names[0]: 8,
        parameter_names[1]: 8,
    }

    # The structural pruning must also have been rolled back.
    assert len(experiment._2qbg_positions) == 1
