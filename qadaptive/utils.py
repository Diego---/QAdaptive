from qiskit.transpiler.basepasses import TransformationPass
from qiskit.dagcircuit import DAGCircuit, DAGInNode, DAGOpNode
from qiskit.circuit.library import RZGate, RXGate, RYGate
from qiskit.circuit import QuantumCircuit, ParameterVector
from qiskit.transpiler import PassManager
from qiskit.converters import circuit_to_dag, dag_to_circuit

class RemoveUnnecessaryControlledGates(TransformationPass):
    def run(self, dag: DAGCircuit) -> DAGCircuit:
        # dag = circuit_to_dag(circ)
        for node in dag.op_nodes():
            if len(node.qargs) > 1:
                predecessors = dag.predecessors(node)
                if all([isinstance(predecessor, DAGInNode) for predecessor in predecessors]):
                    dag.remove_op_node(node)
        return dag
    
class RemoveInitialRZ(TransformationPass):
    def run(self, dag: DAGCircuit) -> DAGCircuit:
        for node in dag.op_nodes():
            if isinstance(node.op, RZGate):
                predecessors = dag.predecessors(node)
                if isinstance(list(predecessors)[0], DAGInNode):
                    dag.remove_op_node(node)
        return dag
    
class MergeConsecutiveRotations(TransformationPass):
    def run(self, dag: DAGCircuit) -> DAGCircuit:
        for qubit in dag.qubits:
            prev_node = None
            for node in list(dag.nodes_on_wire(qubit)):
                if isinstance(node, DAGOpNode) and isinstance(node.op, (RXGate, RYGate, RZGate)):
                    prev_nodes = list(dag.predecessors(node))
                    if not prev_nodes:
                        continue
                    prev_node = prev_nodes[-1]
                    if isinstance(prev_node, DAGOpNode) and type(node.op) == type(prev_node.op):
                        dag.remove_op_node(prev_node)
        return dag
    
custom_pass_manager = PassManager([
    RemoveUnnecessaryControlledGates(),
    RemoveInitialRZ(),
    MergeConsecutiveRotations()
])    
    
def change_circuit_parameters(circ: QuantumCircuit) -> QuantumCircuit:
    num_params = circ.num_parameters
    new_vector = ParameterVector("θ", num_params)
    param_dict = {param: new_vector[i] for i, param in enumerate(circ.parameters)}
    
    return circ.assign_parameters(param_dict, inplace=False), new_vector
