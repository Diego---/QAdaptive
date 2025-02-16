import pytest

from qiskit.circuit import QuantumCircuit, ParameterVector

from qae.optimization.my_spsa import SPSA
from qadaptive.adaptive_ansatz import AdaptiveAnsatz
from qadaptive.mutable_optimizer import MutableOptimizer

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

tiny_ansatz.draw('mpl')


def test_muable_optimizer_initialization():
    """Test that MutableOptimizer correctly initializes."""
    mo = MutableOptimizer(AdaptiveAnsatz(tiny_ansatz), SPSA())
    assert isinstance(mo.ansatz, QuantumCircuit)


