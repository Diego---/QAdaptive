# QAdaptive

QAdaptive is an experimental Python package for building and training **adaptive variational quantum circuits** on top of **Qiskit**.

The project is motivated by **variable-structure ansatz** methods such as VAns / QVANS: instead of fixing a circuit architecture in advance, the ansatz can be **modified during optimization** by inserting, removing, simplifying, and pruning gates or blocks.

The current focus of the package is the **mutable ansatz** and the infrastructure needed to support adaptive training loops.

---

## Current Status (March 2026)

The project is in an early, actively developed state. At the moment, the package already supports the core ingredients required for adaptive ansatz experiments:


Implemented:
- `AdaptiveAnsatz`
  - wraps a parameterized `QuantumCircuit`,
  - tracks trainable parameters,
  - supports insertion and removal of gates and blocks,
  - supports random insertion from an operator pool,
  - can optionally track ansatz history,
  - supports rollback to previous circuit states.

- Mutation / bookkeeping utilities
  - updating parameter bookkeeping after structural changes,
  - tracking two-qubit gate positions,
  - handling locked gates,
  - infrastructure for preserving structural consistency across insertions and removals.

- Simplification / transpilation tools
  - custom simplification passes,
  - custom pass manager construction,
  - passes for removing trivial leading/trailing structure and merging reducible rotations.

- Training infrastructure
  - `InnerLoopTrainer` for running parameter optimization on a fixed mutable ansatz state,
  - `MutableAnsatzExperiment` for coupling the ansatz object to an optimizer/trainer workflow,
  - support for gradient-aware training workflows.

- Optimizer integration
  - works with a modified SPSA implementation,
  - supports power-series hyperparameter setup.

### Not Yet Stable

The public API is still evolving. In particular, the following areas should still be considered unstable:

- locked-gate bookkeeping,
- two-qubit gate remapping after structural edits,
- simplification/transpilation interactions with internal bookkeeping,
- higher-level application wrappers.

---

## Design Philosophy

QAdaptive separates the problem into three layers:

1. **Mutable circuit object**  
   The ansatz itself, including circuit structure, parameters, history, and mutation methods.

2. **Inner-loop training**  
   Parameter optimization for a fixed ansatz instance.

3. **Application / cost-function layer**  
   Problem-specific composition and evaluation, such as:
   - expectation values for VQE,
   - classification losses for QNNs,
   - compression / reconstruction losses for QAEs.

This means that the mutable ansatz is intended to remain **problem-agnostic**.  
Task-specific composition of circuits should be handled by the **loss function** or a future **application wrapper**, rather than by the ansatz object itself.

## Compatibility Notes

QAdaptive currently targets the **Qiskit 1.x API**.  
Compatibility with Qiskit 2.x has not yet been tested or implemented.

`MutableAnsatzExperiment` currently relies on a custom optimizer interface for
its inner-loop training. In particular, it does not depend only on the standard
Qiskit `Optimizer.minimize(...)` entry point. Instead, the experiment performs
optimization by calling optimizer methods directly during each inner
iteration.

### Optimizer requirements for `MutableAnsatzExperiment`

To support the current inner-loop training implementation, a compatible optimizer
must provide the following methods:

- `compute_loss_and_gradient_estimate(loss, x, **kwargs) -> tuple[float, np.ndarray]`  
  Estimate the current loss value and gradient at parameter vector `x`.

- `process_update(gradient_estimate, x, fx, fun, fun_next, iteration_start=0, iteration=0)
  -> tuple[bool, np.ndarray, float | None]`  
  Convert a gradient estimate into a parameter update, returning whether the step
  should be skipped, the next parameter vector, and the next function value if
  it was evaluated.

The current implementation relies on an optimizer that maintains
iteration / schedule state through the following attributes or helpers:

- `p_iterator`
- `lr_iterator`
- `_lr_iterator_copy`
- `last_iteration`
- `_create_iterators(fun=None, x0=None)`

Depending on the training mode and callback configuration, the following
attributes are also accessed:

- `blocking`
- `_size_full_batch`
- `callback`
- `termination_checker`
- `_nfev`
- `_nextfev`

This means that, in its current state, `MutableAnsatzExperiment` is only compatible
with the custom SPSA implementation avilable at [QAE](https://github.com/Diego---/QAE313),
rather than with arbitrary Qiskit optimizers out of the box.

If you plan to use `MutableAnsatzExperiment` in its current state, ensure the
external dependency providing `qae.optimization.my_spsa.SPSA` is available in
your environment.

## Installation

```bash
pip install -e .
```

## Quick Start

### 1) Build and mutate an adaptive ansatz

```python
from qiskit import QuantumCircuit
from qadaptive import AdaptiveAnsatz

qc = QuantumCircuit(2)
adaptive = AdaptiveAnsatz(qc, track_history=True)

# Insert single gates
adaptive.add_gate_at_index("rx", 0, [0])
adaptive.add_gate_at_index("cz", 1, [0, 1])

# Insert a parameterized block from the default pool
adaptive.add_block_at_index("cx_identity", 2, [0, 1])

current = adaptive.get_current_ansatz()
print(current.num_parameters)

# Rollback the last mutation
adaptive.rollback(1)
```

### 2) Use the mutable experiment wrapper

```python
import numpy as np

from qiskit import QuantumCircuit

from qadaptive import AdaptiveAnsatz
from qadaptive.trainer import InnerLoopTrainer
from qadaptive.mutable_ansatz_experiment import MutableAnsatzExperiment
from qadaptive.utils import custom_pass_manager
from qadaptive.action_definitions import (
    INSERT_RANDOM_GATE,
    INSERT_GATE,
    INSERT_BLOCK,
    REMOVE_GATE,
    SIMPLIFY,
    PRUNE_TWO_QUBIT,
    ACTION_DEFINITIONS, ActionDefinition
)
from qadaptive.outer_loop import ActionSpec, OuterStepPlan
from qae.my_spsa import SPSA

# Example: create a starting ansatz
adaptive_ansatz = AdaptiveAnsatz(QuantumCircuit(2))

# Optimizer
spsa_mutable = SPSA(
    maxiter=100,
    callback=store_results,
    resamplings=resample,
)

spsa_mutable.set_power_series_hyperparameters(**hyperparams)

# Trainer
trainer = InnerLoopTrainer(
    optimizer=spsa_mutable,
    track_gradients=True,
)

# Optional pass manager
pass_manager = custom_pass_manager(
    remove_initial_rz=True,
    remove_input_controlled_gates=True,
)

# Mutable experiment
mo = MutableAnsatzExperiment(
    adaptive_ansatz=adaptive_ansatz,
    trainer=trainer,
)

# Initial point
random_initial_point = np.random.choice([-1, 1], size=mo.ansatz.num_parameters)

# Train once against a user-defined loss function
mo.train_one_time(
    loss_function=energy_function,
    initial_point=random_initial_point,
    iterations=100,
)

# Define actions and an action plan
actions = [
    ActionSpec(
        action=INSERT_BLOCK,
        kwargs={
            'block_name': 'cx_identity',
            'qubits': [0, 1],
            'circ_ind': 0
        }
    ),
    ActionSpec(
        action=INSERT_BLOCK,
        kwargs={
            'block_name': 'cz_identity',
            'qubits': [0, 1],
            'circ_ind': 3
        }
    ),
    ActionSpec(
        action=SIMPLIFY,
        kwargs={
            'pass_manager': pass_manager
        }
    )
]

outer_plan = OuterStepPlan(
    name="Initial Growth",
    actions=actions,
    acceptance_mode='outer',
)
```

Note: full training paths currently rely on a compatible optimizer API (notably the external SPSA implementation referenced above).

## Project Layout

- `qadaptive/__init__.py`: package exports and top-level import surface.
- `qadaptive/adaptive_ansatz.py`: core mutable ansatz container and structural edit operations.
- `qadaptive/applications.py`: task-level application helpers and abstractions for problem-specific workflows.
- `qadaptive/mutable_ansatz_experiment.py`: training loop + structure adaptation orchestration.
- `qadaptive/mutation.py`: low-level mutation utilities and bookkeeping updates for insertions/removals.
- `qadaptive/operator_pool.py`: parameterized block definitions and default operator pool.
- `qadaptive/pruning.py`: utilities for pruning or removing low-impact circuit structure.
- `qadaptive/simplification.py`: circuit simplification logic and rewrite rules.
- `qadaptive/trainer.py`: inner-loop optimization/training infrastructure for fixed ansatz states.
- `qadaptive/utils.py`: shared utilities, including transpiler passes and custom pass manager construction.
- `qadaptive/tests/`: unit tests.

### Notes on module roles

- **Core ansatz state**
  - `adaptive_ansatz.py` holds the mutable circuit object itself: circuit data, parameters, history, rollback, and user-facing edit methods.

- **Structure updates**
  - `mutation.py` contains the lower-level logic that keeps internal bookkeeping consistent when the circuit changes, especially around parameter tracking, two-qubit gate positions, and locked gates.

- **Growth primitives**
  - `operator_pool.py` defines the building blocks that can be inserted into the ansatz, such as parameterized single-qubit or two-qubit blocks.

- **Reduction / cleanup**
  - `simplification.py` handles deterministic cleanup of redundant structure.
  - `pruning.py` is the natural place for cost-aware removal of gates or blocks judged to be unimportant.

- **Training**
  - `trainer.py` handles parameter optimization for a fixed circuit structure.
  - `mutable_ansatz_experiment.py` sits one level above that and coordinates optimization together with ansatz mutation.

- **Outer Loop**
  - `action_definitions.py` gives the available actions to take on the ansatz of a MutableAnsatzExperiment object.
  - `outer_loop.py` defines atomic actions to be taken by an outer loop plan.

- **Applications**
  - `applications.py` is intended for problem-specific composition, such as VQE, QNN, or QAE workflows, where the mutable ansatz may be prepended or appended to other circuits.

- **Utilities**
  - `utils.py` contains shared helpers that do not belong cleanly to a single module, including transpilation-related helpers and pass-manager setup.

