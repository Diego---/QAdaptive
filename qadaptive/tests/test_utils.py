from qiskit import QuantumCircuit
from qiskit.transpiler import PassManager

from qadaptive.utils.utils import (
    RemoveFinalRZ,
    RemoveInitialRZ,
    RemoveInputControlledGates,
    custom_pass_manager,
)



def operation_names(circuit: QuantumCircuit) -> list[str]:
    return [inst.operation.name for inst in circuit.data]



def test_remove_input_controlled_gates_removes_leading_two_qubit_gate():
    qc = QuantumCircuit(2)
    qc.cz(0, 1)
    qc.rx(0.1, 0)

    out = PassManager([RemoveInputControlledGates()]).run(qc)

    assert operation_names(out) == ["rx"]



def test_remove_input_controlled_gates_keeps_noninitial_two_qubit_gate():
    qc = QuantumCircuit(2)
    qc.rx(0.1, 0)
    qc.cz(0, 1)

    out = PassManager([RemoveInputControlledGates()]).run(qc)

    assert operation_names(out) == ["rx", "cz"]



def test_remove_initial_rz_removes_only_leading_rz():
    qc = QuantumCircuit(1)
    qc.rz(0.1, 0)
    qc.rx(0.2, 0)
    qc.rz(0.3, 0)

    out = PassManager([RemoveInitialRZ()]).run(qc)

    assert operation_names(out) == ["rx", "rz"]



def test_remove_final_rz_removes_only_trailing_rz():
    qc = QuantumCircuit(1)
    qc.rz(0.1, 0)
    qc.rx(0.2, 0)
    qc.rz(0.3, 0)

    out = PassManager([RemoveFinalRZ()]).run(qc)

    assert operation_names(out) == ["rz", "rx"]



def test_custom_pass_manager_can_enable_optional_initial_rz_removal():
    qc = QuantumCircuit(1)
    qc.rz(0.1, 0)

    out = custom_pass_manager(remove_initial_rz=True).run(qc)

    assert len(out.data) == 0
