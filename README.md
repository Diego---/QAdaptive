# QAdaptive

QAdaptive is an experimental Python package for building and training **adaptive variational quantum circuits** on top of Qiskit.

## Current Status (March 2026)

The project is in an early, actively developed state.

Implemented:
- `AdaptiveAnsatz` supports:
  - inserting/removing gates at explicit indices,
  - inserting parameterized blocks,
  - random gate/block insertion,
  - parameter tracking,
  - optional history tracking + rollback.
- `PoolBlock` and a default block pool (`rz_rx_rz`, `cx_identity`).
- `MutableAnsatzExperiment` is a class for iterative training of a mutable ansatz.
- Custom transpiler/simplification pass manager in `qadaptive/utils.py`.
- Unit tests for core `AdaptiveAnsatz` behavior and part of `MutableAnsatzExperiment` insertion behavior.

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
with the custom SPSA implementation avilable at [text](https://github.com/Diego---/QAE313).
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
from qadaptive import AdaptiveAnsatz, MutableAnsatzExperiment
from qiskit import QuantumCircuit

qc = QuantumCircuit(2)
adaptive = AdaptiveAnsatz(qc)
exp = MutableAnsatzExperiment(adaptive_ansatz=adaptive)

# Structural edits from the experiment API
exp.insert_at("rx", [0], 0)
exp.insert_block_at("rz_rx_rz", [0], 1)
```

Note: full training paths currently rely on a compatible optimizer API (notably the external SPSA implementation referenced above).

## Project Layout

- `qadaptive/adaptive_ansatz.py`: mutable ansatz container and edit operations.
- `qadaptive/mutable_ansatz_experiment.py`: training loop + structure adaptation orchestration.
- `qadaptive/operator_pool.py`: parameterized block definitions and default pool.
- `qadaptive/utils.py`: transpiler passes and custom pass manager.
- `qadaptive/tests/`: unit tests.

## Contributing / Next Priorities

High-impact next steps:
- implement and test `remove_random_gate`,
- formalize optimizer interface and remove hard dependency on external `qae` path,
- complete utility-pass tests,
- add CI with a pinned modern Python/Qiskit matrix.
