import pytest
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import Operator

from qadaptive.core.operator_pool import DEFAULT_BLOCK_POOL, PoolBlock



def operation_names(circuit: QuantumCircuit) -> list[str]:
    return [inst.operation.name for inst in circuit.data]



def test_pool_block_build_validates_parameter_count():
    def builder(params):
        qc = QuantumCircuit(1)
        qc.rx(params[0], 0)
        return qc

    block = PoolBlock(
        name="single_rx",
        num_qubits=1,
        num_parameters=1,
        builder=builder,
    )

    with pytest.raises(ValueError):
        block.build([])



def test_default_block_pool_contains_expected_blocks():
    assert  all(block in set(DEFAULT_BLOCK_POOL) for block in {"rz_rx_rz", "cx_identity", "cz_identity", "single_cx_block", "single_cz_block"})



def test_cx_identity_block_metadata_and_structure_are_consistent():
    block = DEFAULT_BLOCK_POOL["cx_identity"]
    params = ParameterVector("t", block.num_parameters)
    qc = block.build(list(params))

    assert block.num_qubits == 2
    assert block.num_parameters == 4
    assert qc.num_qubits == 2
    assert qc.num_parameters == 4
    assert operation_names(qc) == ['ry', 'rz', 'cx', 'ry', 'rz', 'cx']



def test_default_blocks_are_identity_at_zero():
    for block in DEFAULT_BLOCK_POOL:
        if block != "single_cx_block" and block != "single_cz_block":
            block_obj = DEFAULT_BLOCK_POOL[block]
            num_qubits = block_obj.num_qubits
            params = ParameterVector("t", block_obj.num_parameters)
            qc = block_obj.build(list(params))
            bound = qc.assign_parameters({p: 0.0 for p in qc.parameters})

            assert Operator(bound).equiv(Operator(QuantumCircuit(num_qubits)))



def test_cz_identity_block_structure_matches_documented_pattern():
    block = DEFAULT_BLOCK_POOL["cz_identity"]
    params = ParameterVector("t", block.num_parameters)
    qc = block.build(list(params))

    assert operation_names(qc) == ['rx', 'ry', 'cz', 'rx', 'ry', 'cz']
