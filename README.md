# QAdaptive

Adaptive variational quantum circuits in Qiskit.

QAdaptive is a research-oriented Python package for workflows in which the **circuit structure itself changes during optimization**. Instead of fixing an ansatz once and only training its parameters, QAdaptive lets you alternate between:

- an **inner loop** that optimizes the current parameters, and
- an **outer loop** that modifies the circuit by inserting gates or blocks, simplifying the circuit, and pruning structure.

At the package root, QAdaptive currently exports three main entry points:

- `AdaptiveAnsatz`
- `MutableAnsatzExperiment`
- `InnerLoopTrainer`

## Why QAdaptive?

Many variational workflows start from a fixed ansatz and focus only on parameter optimization. QAdaptive is built for a different regime:

- start from a very small or deliberately simple circuit,
- grow the circuit when the current structure is insufficient,
- simplify or prune when the circuit becomes unnecessarily large,
- warm-start retraining after structural changes,
- keep enough history to analyze how the ansatz evolved.

This makes the package useful for experiments in adaptive VQE, variational quantum control, quantum autoencoders, and related structure-learning problems.

## Core ideas

QAdaptive revolves around three abstractions:

### `AdaptiveAnsatz`
Wraps a parameterized `QuantumCircuit` so that it can be modified structurally while keeping parameter bookkeeping consistent.

### `InnerLoopTrainer`
Runs parameter optimization for a fixed ansatz using a stepwise optimizer. The trainer can keep gradient history and, if requested, record a full per-iteration training trace.

### `MutableAnsatzExperiment`
Orchestrates the full adaptive workflow. It combines an `AdaptiveAnsatz` and an `InnerLoopTrainer`, applies outer-loop actions such as gate insertion, block insertion, simplification, and pruning, and keeps histories of accepted structures and optimization results.

## Features

- Convert a generic parameterized `QuantumCircuit` into an adaptive ansatz.
- Alternate parameter training with structural updates.
- Insert single-qubit gates, two-qubit gates, or predefined blocks.
- Simplify circuits through transpiler-based passes.
- Prune two-qubit structure while respecting locked gates.
- Warm-start new training runs from the last accepted parameters.
- Record optimization results, parameter memory, accepted ansatz history, and training traces.
- Use plotting utilities to visualize objective trajectories and parameter evolution.

## Installation

QAdaptive targets Python 3.10+ and currently declares compatibility with `qiskit >= 1.1, < 2`.

Install in editable mode during development:

```bash
pip install -e .
```

From the current package metadata, the main runtime dependencies are:

- `numpy`
- `matplotlib`
- `IPython`
- `qiskit >= 1.1, < 2`
- `qiskit_experiments >= 0.6.1`
- `qiskit_algorithms >= 0.3.0`

## Quickstart: adaptive VQE from a separable ansatz

The example below illustrates the intended usage pattern on a small toy VQE problem. The starting circuit is deliberately simple: one `rx` rotation per qubit, with no entanglement. QAdaptive then grows and prunes the ansatz around that seed.

```python
from functools import partial

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp, Statevector

from qadaptive import AdaptiveAnsatz, MutableAnsatzExperiment, InnerLoopTrainer
from qadaptive.outer import (
    build_single_qubit_block_plan,
    build_star_growth_plan,
    build_targeted_prune_plan,
    combine_plan_builders,
    default_append_index_policy,
    between_2qg_indices_policy,
    make_select_random_gates,
)
from qadaptive.training import SPSA, TerminationChecker

# --- Problem Hamiltonian ----------------------------------------------------
# A small toy Hamiltonian for demonstration.
H = SparsePauliOp.from_list([
    ("ZII", -0.7),
    ("IZI", -0.7),
    ("IIZ", -0.7),
    ("ZZI",  0.4),
    ("IZZ",  0.4),
    ("XXI",  0.2),
])


def vqe_cost(params, ansatz):
    """Return the exact statevector energy of the current ansatz."""
    bound = ansatz.assign_parameters(params, inplace=False)
    psi = Statevector.from_instruction(bound)
    value = psi.expectation_value(H)
    return float(np.real(value))


# --- Initial separable ansatz ----------------------------------------------
num_qubits = 3
theta = ParameterVector("theta", num_qubits)

initial_circuit = QuantumCircuit(num_qubits)
for q in range(num_qubits):
    initial_circuit.rx(theta[q], q)

# Convert the generic circuit into an adaptive ansatz.
adaptive_ansatz = AdaptiveAnsatz.from_generic_circuit(initial_circuit)


# --- Inner-loop optimizer ---------------------------------------------------
termination_checker = TerminationChecker(
    mode="min",
    target_value=None,
    target_tol=1e-4,
    plateau_window=15,
    plateau_slope_tol=1e-4,
    plateau_improvement_tol=1e-4,
)

optimizer = SPSA(
    maxiter=100,
    termination_checker=termination_checker,
    resamplings=1,
)

trainer = InnerLoopTrainer(
    optimizer=optimizer,
    track_gradients=True,
)


# --- Adaptive experiment ----------------------------------------------------
experiment = MutableAnsatzExperiment(
    adaptive_ansatz=adaptive_ansatz,
    trainer=trainer,
)


# --- Outer-loop schedule ----------------------------------------------------
growth_builder = partial(
    build_star_growth_plan,
    block_name="cz_identity_rotation_on_one_qubit",
    center_qubit=0,
    max_insertions=1,
    repetitions=1,
    insert_index_policy=default_append_index_policy,
    force_accept=False,
    add_simplify=False,
)

single_qubit_builder = partial(
    build_single_qubit_block_plan,
    block_name="rz_rx_rz",
    qubits=[0],
    insert_index_policy=between_2qg_indices_policy,
    num_insertions=1,
    force_accept=False,
    add_simplify=False,
)

prune_builder = partial(
    build_targeted_prune_plan,
    targeting_function=make_select_random_gates(num_gates=1),
    max_num_2q_gates=1,
    add_simplify=False,
)

schedule = [
    combine_plan_builders(growth_builder, single_qubit_builder),
    prune_builder,
    combine_plan_builders(growth_builder, single_qubit_builder),
]


# --- Run the adaptive loop --------------------------------------------------
results = experiment.run_outer_loop(
    loss_function=vqe_cost,
    plan_schedule=schedule,
    outer_iterations=len(schedule),
    train_iterations=50,
    train_before_first_plan=True,
    initial_point=None,
    train_after_plan=True,
    trainer_iteration_reset=0,
    update_parameter_memory=True,
    reuse_parameter_memory=True,
    default_value_for_new_params=0.0,
    record_parameter_memory=True,
    record_run_history=True,
    store_initial_value_in_history=True,
    accept_tol=0.0,
    stop_on_error=True,
)


# --- Inspect what happened --------------------------------------------------
print("Final energy:", experiment.last_cost)
print("Number of accepted ansatz states:", len(experiment.accepted_ansatz_history))
print("Number of optimizer results:", len(experiment.result_history))
print("Number of training runs recorded:", len(experiment.training_run_history))

final_circuit = experiment.ansatz
final_params = experiment.get_current_parameter_dict()
```

## What this workflow is doing

The example above follows the same pattern used in the package notebook examples:

1. **Define a problem-specific objective** as a Python callable.
2. **Start from a generic parameterized circuit**.
3. **Convert it to an adaptive ansatz** with `AdaptiveAnsatz.from_generic_circuit(...)`.
4. **Configure an inner-loop optimizer** through `InnerLoopTrainer`.
5. **Define an outer-loop schedule** with plan builders.
6. **Run the adaptive loop** with `MutableAnsatzExperiment.run_outer_loop(...)`.
7. **Inspect the resulting histories**.

This separation is deliberate: QAdaptive handles the adaptive circuit workflow, while the user remains in control of the problem definition.

## Inspecting results

The experiment object is designed to expose the state of the run after optimization. In particular, the notebook workflow uses:

- `experiment.result_history` for the optimizer-level outcomes of each training phase,
- `experiment.accepted_ansatz_history` for the accepted structural milestones,
- `experiment.training_run_history` for detailed inner-loop trajectories when `record_run_history=True`,
- `experiment.last_cost` and `experiment.last_params` for the current accepted state.

Typical inspection patterns look like this:

```python
print(experiment.last_cost)
print(experiment.get_current_parameter_dict())
print(len(experiment.accepted_ansatz_history))

best_record = min(
    (record for record in experiment.accepted_ansatz_history if record.cost is not None),
    key=lambda record: record.cost,
)

best_circuit = best_record.ansatz.copy()
best_params = dict(best_record.parameter_values)
```

If you record full training traces, the plotting utilities can be used to reconstruct the global optimization history:

```python
from qadaptive.utils.plotting import (
    build_training_run_traces,
    plot_cost_with_outer_boundaries,
    plot_parameter_lifelines,
    plot_parameter_heatmap,
)

traces = build_training_run_traces(
    records=experiment.training_run_history,
    outer_step_history=experiment.outer_step_history,
    include_initial=True,
)

plot_cost_with_outer_boundaries(traces)
plot_parameter_lifelines(traces)
plot_parameter_heatmap(traces)
```

## Package layout

A high-level overview of the current repository layout:

```text
qadaptive/
├── core/
│   ├── adaptive_ansatz.py
│   ├── mutation.py
│   ├── operator_pool.py
│   ├── pruning.py
│   └── simplification.py
├── outer/
│   ├── action_definitions.py
│   ├── mutable_ansatz_experiment.py
│   ├── outer_loop.py
│   ├── plan_builders.py
│   └── plan_helpers.py
├── training/
│   ├── history.py
│   ├── trainer.py
│   ├── termination_and_callback.py
│   └── optimizers/
|   
└── utils/
    ├── plotting/
    └── simplification_utils.py
```

## Design philosophy

QAdaptive is intended for research code where **structure search is part of the experiment**. The package favors:

- explicit outer-loop plans,
- inspectable histories,
- warm starts after structural edits,
- integration with Qiskit circuits and transpiler passes,
- small building blocks that can be combined into larger adaptive strategies.

## Current scope

QAdaptive is currently best understood as an **experimental research framework** for adaptive variational circuits in Qiskit. The public API is already useful, but it is still evolving. Users should expect some interfaces and helper names to change as the package matures.

## Testing

The repository includes a `tests/` directory with unit tests covering core ansatz manipulation, mutation, pruning, simplification, trainer behavior, and utilities.

To run the test suite locally:

```bash
pytest
```

## Inspiration

QAdaptive is inspired in part by the VAns framework introduced in:

M. Bilkis, M. Cerezo, G. Verdon, P. J. Coles, and L. Cincio,
[*A semi-agnostic ansatz with variable structure for variational quantum algorithms*](https://doi.org/10.1007/s42484-023-00132-1),
Quantum Machine Intelligence 5, 43 (2023).

This package is an independent implementation and extension of related adaptive-ansatz ideas in a Qiskit-based workflow.

## Version

The package metadata currently identifies QAdaptive as version `0.1` / `0.1.0`.

## Citation

Coming soon

## License

This project is licensed under the Apache License 2.0. See `LICENSE` for details.
