# QAdaptive

A research toolkit for adaptive variational quantum circuits in Qiskit.

QAdaptive is a research-oriented Python package for workflows in which the **circuit structure itself changes during optimization**. One can configure operator pools, gate and block insertion rules, insertion positions, pruning strategies, and acceptance criteria, then combine them into an adaptive experiment.

An inner loop trains the current circuit parameters, while an outer loop proposes structural changes and decides which proposals to retain. Parameter memory supports continued training across those changes.

The package records training trajectories, structural decisions, and accepted and trial circuit snapshots. Summaries, plots, and structured archives help you inspect experiments and preserve their results for later analysis.

QAdaptive is an experimental research framework.

## What can you configure?

| Part of the experiment | Examples |
| --- | --- |
| Objective | A Python callable defining the task, such as a VQE energy. |
| Initial circuit | A parameterised Qiskit `QuantumCircuit`. |
| Operator and block pools | Supported gate names and built-in or custom `PoolBlock` definitions. |
| Growth | Star, uniform, nearest-neighbour, or custom structural proposals. |
| Insertion positions | Append, random positions, positions between two-qubit instructions, or a custom policy. |
| Pruning and simplification | Gate selectors, pruning sweeps, targeted pruning, and simplification passes. |
| Inner optimization | Compatible stepwise optimizers, including QAdaptive's SPSA and ADAM. |
| Acceptance | Objective tolerances, complexity penalties, Metropolis settings, and forced acceptance. |
| Scheduling | The sequence of plans, training budgets, parameter reuse, and optimizer iteration resets. |

The objective supplies the application-specific problem. QAdaptive coordinates circuit changes, parameter training, and experimental records.


### Core objects

| Object | Responsibility |
| --- | --- |
| `AdaptiveAnsatz` | Wraps a parameterised circuit and manages structural edits and parameter bookkeeping. |
| `InnerLoopTrainer` | Trains the parameters of the current circuit using a compatible stepwise optimizer. |
| `InnerLoopRecorder` | Records inner training runs, accepted optimizer updates, optional gradients, and additional diagnostics; provides inner-history plots. |
| `MutableAnsatzExperiment` | Coordinates outer proposals, training, acceptance, rollback, parameter memory, and histories. |

All four objects are available from the package root:

```python
from qadaptive import (
    AdaptiveAnsatz,
    InnerLoopRecorder,
    InnerLoopTrainer,
    MutableAnsatzExperiment,
)
```

## Installation

QAdaptive targets Python 3.10+ and is currently compatible with `qiskit >= 1.1, < 2`.

For a local installation,

```bash
git clone https://github.com/Diego---/QAdaptive.git
cd QAdaptive
python -m pip install .
```

## Quickstart: adaptive VQE from a separable ansatz

This example uses a fixed two-qubit Hamiltonian and exact statevector energies. It starts from an off-zero point, proposes growth and pruning/simplification, and retrains after structural proposals. The short training budgets are intended for demonstrating the workflow.

```python
import random
from functools import partial

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp, Statevector
from qiskit_algorithms.utils import algorithm_globals

from qadaptive import (
    AdaptiveAnsatz,
    InnerLoopRecorder,
    InnerLoopTrainer,
    MutableAnsatzExperiment,
)
from qadaptive.outer import (
    build_prune_sweep_plan,
    build_star_growth_plan,
    default_append_index_policy,
)
from qadaptive.training import SPSA

SEED = 1234
TRAIN_ITERATIONS = 40
random.seed(SEED)
np.random.seed(SEED)
algorithm_globals.random_seed = SEED

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

def energy(params, ansatz):
    bound = ansatz.assign_parameters(params, inplace=False)
    state = Statevector.from_instruction(bound)
    return float(state.expectation_value(H).real)


# --- Initial separable ansatz ----------------------------------------------
num_qubits = int(H.num_qubits)
theta = ParameterVector("theta", num_qubits)

initial_circuit = QuantumCircuit(num_qubits)
for q in range(num_qubits):
    initial_circuit.rx(theta[q], q)

# Convert the generic circuit into an adaptive ansatz.
adaptive_ansatz = AdaptiveAnsatz.from_generic_circuit(
    initial_circuit,
    operator_pool=["rx", "ry", "rz", "cx", "cz"],
)


# --- Inner-loop optimizer ---------------------------------------------------
spsa_settings = {
    "a": 0.2,
    "alpha": 0.602,
    "c": 0.1,
    "gamma": 0.101,
    "stability_constant": 0.0,
}
optimizer = SPSA(resamplings=1)
optimizer.set_power_series_hyperparameters(**spsa_settings)

recorder = InnerLoopRecorder(
    record_initial_value=True,
    record_gradients=True,
)

trainer = InnerLoopTrainer(
    optimizer=optimizer,
    recorder=recorder,
)


# --- Adaptive experiment ----------------------------------------------------
experiment = MutableAnsatzExperiment(
    adaptive_ansatz=adaptive_ansatz,
    trainer=trainer,
)


# --- Outer-loop schedule ----------------------------------------------------
growth_builder = partial(
    build_star_growth_plan,
    block_name="cx_identity",
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
    evaluation_loss=energy,
    plan_schedule=schedule,
    outer_iterations=len(schedule),
    train_iterations=TRAIN_ITERATIONS,
    train_before_first_plan=True,
    trainer_iteration_reset=None,
    reuse_parameter_memory=True,
    default_value_for_new_params=0.0,
    record_parameter_memory=True,
    accept_tol=0.0,
    stop_on_error=True,
)


# --- Inspect what happened --------------------------------------------------
experiment.print_summary()

final_circuit = experiment.ansatz
final_params = experiment.get_current_parameter_dict()
```

`evaluation_loss=energy` explicitly evaluates every accepted updated parameter point with `energy`. Leaving `evaluation_loss=None` performs no additional per-step evaluation; the recorder keeps the optimizer estimate instead. The final parameter point is always evaluated explicitly, using `evaluation_loss` when provided and otherwise `loss_function`. `trainer_iteration_reset=None` continues the SPSA iteration schedule across training phases. Parameter memory reuses values for parameters that remain active, newly introduced parameters start at zero.

The `cx_identity` block becomes the identity at zero rotation angles. Zero initialisation preserves the current circuit's action for such blocks, this property depends on the selected block.

### Parameter-dependent SPSA schedules

Standard SPSA uses a single learning-rate and perturbation schedule for the
complete parameter vector. In an adaptive circuit, however, parameters can be
introduced at very different stages of the optimization. A newly inserted
parameter would otherwise inherit the already-decayed global schedule and may
therefore receive only very small updates.

QAdaptive can instead track the SPSA schedule age independently for every
parameter:

```python
optimizer = SPSA(
    resamplings=1,
    parameter_dependent_schedules=True,
)
optimizer.set_power_series_hyperparameters(**spsa_settings)
```

Existing parameters retain their accumulated schedule age across structural
growth, while newly introduced parameters start at schedule index zero. For
the usual SPSA power-series schedules

$$
a_i\left(n_i\right)=\frac{a}{\left(n_i+1+A\right)^\alpha}, \quad c_i\left(n_i\right)=\frac{c}{\left(n_i+1\right)^\gamma},
$$

each parameter therefore evolves according to its own optimization age
$n_i$.

### Modulating parameter birth strength

A fresh schedule does not necessarily need to restart at full strength late in
an adaptive optimization. QAdaptive can additionally modulate the initial
learning rate and perturbation strength according to the outer-loop iteration
$K_i$ at which a parameter was introduced:

$$
a_i(n_i, K_i)=\frac{a}{(K_i + 1)^{\beta_a}}\frac{1}{(n_i + 1 + A)^\alpha},
$$

$$
c_i(n_i, K_i)=\frac{c}{(K_i + 1)^{\beta_c}}\frac{1}{(n_i + 1)^\gamma}.
$$

Configure the birth modulation independently for the learning-rate and
perturbation schedules:

```python
optimizer.set_parameter_birth_power_series(
    learning_rate_exponent=0.5,
    perturbation_exponent=0.25,
)
```

The birth exponents control how strongly later parameters are initialized:
- exponent = 0.0 disables birth-strength modulation and gives every new
  parameter the same fresh schedule;
- positive exponents make later-born parameters start more conservatively;
- negative exponents make later-born parameters start more aggressively.

The parameter's birth iteration is retained throughout its lifetime. Restarting
a schedule resets its optimization age but does not change when the parameter
was introduced.

#### Restarting schedules after pruning

Pruning can substantially change the represented unitary even when the
surviving parameter values are unchanged. The outer loop can therefore
optionally restart the schedules of all surviving parameters after a
successfully applied pruning action:

```python
results = experiment.run_outer_loop(
    loss_function=vqe_cost,
    plan_schedule=schedule,
    train_iterations=TRAIN_ITERATIONS,
    reuse_parameter_memory=True,
    restart_parameter_schedules_after_pruning=True,
)
```

If the pruning proposal is later rejected by the outer acceptance rule, both
the previous per-parameter schedule ages and parameter-birth metadata are
restored together with the previous ansatz.

The recorder stores the learning rate and perturbation strength actually used
for every parameter at every recorded optimizer step:

```python
recorder.plot_learning_rates(parameters=["θ_0", "θ_3"])
recorder.plot_perturbations(parameters=["θ_0", "θ_3"])
```

The optimizer also exposes the current parameter birth iterations:

```python
optimizer.parameter_birth_outer_iterations
```

Parameter-dependent schedules currently support first-order SPSA only.

## Inspecting results

The following examples continue from the quickstart.

```python
import matplotlib.pyplot as plt

# Plot the objective reached during each inner-step optimization
experiment.plot_outer_history(ylabel="Energy")
# Print first and last ansätze used
experiment.plot_architecture_evolution(indices=[0, -1])
experiment.plot_complexity_evolution()

# Select the objective provenance explicitly, or use "auto".
experiment.recorder.plot_objective(source="auto")
# experiment.recorder.plot_objective(source="evaluation")
# experiment.recorder.plot_objective(source="optimizer_estimate")
experiment.recorder.plot_parameters()
experiment.recorder.plot_parameter_heatmap(normalize=True)

plt.show()
```

Plotting methods return the Matplotlib figure and axes for further customization or export:

```python
fig, ax = experiment.plot_outer_history(ylabel="Energy")
fig.savefig("outer_history.pdf", bbox_inches="tight")
```

`plot_outer_history()` shows retained costs and separate markers for rejected trial costs. `plot_complexity_evolution()` applies the same distinction to parameter and two-qubit-instruction counts.

`recorder.plot_objective(source=...)` distinguishes objective provenance. `"evaluation"` plots values explicitly evaluated at the recorded parameter points and includes the mandatory final evaluation. `"optimizer_estimate"` plots optimizer estimates at the pre-update points they describe. `"auto"` uses evaluations only when every recorded step has one; otherwise it uses optimizer estimates.

`plot_architecture_evolution()` draws accepted circuit snapshots. Its `indices` select positions in `accepted_ansatz_history`, not outer iteration numbers; `[0, -1]` selects the first and last recorded accepted states.

If the recorder was configured with `extra_objective` and `extra_evaluation_frequency` before training, label that diagnostic curve with:

```python
experiment.recorder.plot_objective(extra_str="Alternative Objective")
```

`extra_str` labels existing extra-objective observations in the legend. The recorder can also be used directly, for example `recorder.plot_parameters()`; it is the same object exposed as `experiment.recorder`.

### Histories and final state

| Attribute | Contents |
| --- | --- |
| `outer_step_history` | Outer results, including acceptance decisions, costs, and complexity before and after proposals. |
| `trial_ansatz_history` | Proposed circuits before and after structural changes, including rejected proposals. |
| `accepted_ansatz_history` | Accepted trained circuit snapshots and their parameter values. |
| `parameter_memory_history` | Recorded parameter-memory states across the workflow. |
| `training_run_history` | The recorder's inner training runs. |
| `result_history` | Completed optimizer-result records, when result tracking is enabled. |
| `last_cost`, `last_params`, `ansatz` | The current accepted cost, parameter vector, and circuit. |

The final accepted state can differ from the lowest-cost accepted state when the chosen acceptance settings allow cost increases. To select the lowest-cost accepted snapshot:

```python
best_record = min(
    (
        record
        for record in experiment.accepted_ansatz_history
        if record.cost is not None
    ),
    key=lambda record: record.cost,
)

best_circuit = best_record.ansatz.copy()
best_params = dict(best_record.parameter_values)
best_cost = best_record.cost
```

## Saving an experiment

`save_history()` creates a structured archive of the recorded experiment and its current accepted state. Use a distinct directory for each save:

```python
import json
from datetime import datetime
from importlib.metadata import version
from pathlib import Path

output_directory = (
    Path("results")
    / datetime.now().strftime("%Y-%m-%d_%H-%M-%S_%f")
)
archive_directory = experiment.save_history(output_directory)

# Keep problem and configuration metadata alongside the experiment archive.
run_config = {
    "seed": SEED,
    "train_iterations": TRAIN_ITERATIONS,
    "spsa": spsa_settings,
    "operator_pool": list(experiment.adaptive_ansatz.operator_pool),
    "pauli_labels": H.paulis.to_labels(),
    "coefficients_real": H.coeffs.real.tolist(),
    "coefficients_imag": H.coeffs.imag.tolist(),
    "package_versions": {
        name: version(name)
        for name in ("qadaptive", "qiskit", "qiskit-algorithms", "numpy")
    },
}
with (archive_directory / "run_config.json").open(
    "w", encoding="utf-8"
) as file:
    json.dump(run_config, file, indent=2)

print(f"Saved experiment to: {archive_directory.resolve()}")
```

Completed archives are protected against overwrite: saving to a directory containing an existing `manifest.json` raises `FileExistsError`.

The archive includes:

| Files | Contents |
| --- | --- |
| `manifest.json` | Archive schema version, save timestamp, recorder settings, record counts, and file references. |
| `current_state.json` | Current accepted parameters and cost, parameter memory, and structural bookkeeping. |
| `outer_step_history.json` | Recorded outer-loop results. |
| `accepted_ansatz_history.json`, `trial_ansatz_history.json` | Snapshot metadata, parameter values, and circuit-file references. |
| `parameter_memory_history.json` | Parameter-memory records. |
| `training_run_history.json` | Recorded inner training runs and observations. |
| `gradient_history.json`, `result_history.json` | Gradient data and optimizer-result summaries, subject to tracking settings. |
| `circuits/` | QPY files for the current circuit and accepted and trial snapshots. |

The archive preserves what the experiment records. Store the problem definition, full strategy settings, seeds, and dependency versions alongside it. The example adds a `run_config.json` sidecar with some of that context; extend it with the settings relevant to your experiment.

Standalone optimization runs performed outside the experiment need their own export.

## Loading saved data for analysis

```python
from qadaptive.persistence.loading import load_experiment_history

# In a later session, replace archive_directory with the saved directory path.
loaded = load_experiment_history(archive_directory)

loaded.print_summary()
loaded.plot_outer_history(ylabel="Energy")
loaded.plot_architecture_evolution(indices=[0, -1])
loaded.plot_complexity_evolution()

final_circuit = loaded.ansatz
final_params = loaded.last_params
final_cost = loaded.last_cost
bound_final_circuit = final_circuit.assign_parameters(final_params)

# Retrieve archived accepted and trial circuits by history position.
accepted_circuit = loaded.load_accepted_ansatz(index=-1)
trial_before = loaded.load_trial_ansatz(index=-1, stage="before")
trial_after = loaded.load_trial_ansatz(index=-1, stage="after")
```

The loader returns a `LoadedExperimentHistory` analysis object. It supports summaries, outer-history plots, complexity plots, and accepted-architecture plots. Saved inner runs are available through `loaded.training_run_history`; the loader currently exposes those records as dictionaries and does not reconstruct an `InnerLoopRecorder` or a live optimizer.

Live history entries can be dataclasses; loaded history entries are dictionaries. For example, select the lowest-cost archived accepted snapshot with:

```python
best_index, best_record = min(
    (
        (index, record)
        for index, record in enumerate(loaded.accepted_ansatz_history)
        if record["cost"] is not None
    ),
    key=lambda item: item[1]["cost"],
)

best_circuit = loaded.load_accepted_ansatz(best_index)
best_params = dict(best_record["parameter_values"])
best_cost = best_record["cost"]
```

The loader validates the archive schema and the final parameter-vector shape. Notebook-added files such as `run_config.json` can be read separately:

```python
with (loaded.directory / "run_config.json").open("r", encoding="utf-8") as file:
    saved_config = json.load(file)
```

### Optional inner-loop recording during execution

To append inner-run events during training, configure the recorder before creating the trainer and experiment:

```python
recorder = InnerLoopRecorder(
    record_initial_value=True,
    record_gradients=True,
    jsonl_path="results/inner_runs.jsonl",
)
```

This JSON Lines stream contains run and recorded-step events. Use `experiment.save_history(...)` for the structured experiment archive.

## Testing

From the repository root, after installing QAdaptive:

```bash
python -m pip install pytest
python -m pytest qadaptive/tests
```

The tests cover circuit mutation, operator pools, pruning, simplification, training and recording, plotting, and archive save/load behaviour.

## Project status and feedback

QAdaptive is intended for research and experimentation. APIs and default strategies may evolve as the package develops.

Bug reports, questions, and suggestions are welcome through [GitHub Issues](https://github.com/Diego---/QAdaptive/issues). Include a minimal example and the package versions used when reporting a problem.


## Inspiration and attribution

QAdaptive is inspired in part by the VAns framework introduced in:

M. Bilkis, M. Cerezo, G. Verdon, P. J. Coles, and L. Cincio,
[*A semi-agnostic ansatz with variable structure for variational quantum algorithms*](https://doi.org/10.1007/s42484-023-00132-1),
Quantum Machine Intelligence 5, 43 (2023).

QAdaptive provides an independent Qiskit-based toolkit for implementing and studying related adaptive circuit strategies.

If you use QAdaptive in research, reference [this repository](https://github.com/Diego---/QAdaptive) and the version or commit used. Cite the VAns paper when discussing the corresponding algorithmic ideas.

## Citation

Coming soon.

## License

This project is licensed under the Apache License 2.0. See `LICENSE` for details.
