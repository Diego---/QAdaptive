from qiskit.transpiler.basepasses import TransformationPass
from qiskit.dagcircuit import DAGCircuit, DAGInNode, DAGOpNode
from qiskit.circuit.library import RZGate, RXGate, RYGate
from qiskit.circuit import QuantumCircuit, ParameterVector
from qiskit.transpiler import PassManager
from qiskit.converters import circuit_to_dag, dag_to_circuit

from qiskit.transpiler.basepasses import TransformationPass
from qiskit.dagcircuit import DAGCircuit, DAGOpNode, DAGInNode
from qiskit.circuit.library import RXGate, RYGate, RZGate

class RemoveUnnecessaryControlledGates(TransformationPass):
    """
    A transpiler pass to remove unnecessary controlled gates from a DAGCircuit.

    This pass identifies controlled gates (multi-qubit gates) that have no dependencies
    (i.e., their predecessors are all input nodes) and removes them from the circuit.
    """

    def run(self, dag: DAGCircuit) -> DAGCircuit:
        """
        Run the pass on the DAGCircuit.

        Parameters
        ----------
        dag : DAGCircuit
            The DAGCircuit to be transformed.

        Returns
        -------
        DAGCircuit
            The transformed DAGCircuit with unnecessary controlled gates removed.
        """
        for node in dag.op_nodes():
            if len(node.qargs) > 1:
                predecessors = dag.predecessors(node)
                if all([isinstance(predecessor, DAGInNode) for predecessor in predecessors]):
                    dag.remove_op_node(node)
        return dag


class RemoveInitialRZ(TransformationPass):
    """
    A transpiler pass to remove initial RZ gates from a DAGCircuit.

    This pass identifies RZ gates that are applied at the beginning of the circuit
    (i.e., their predecessors are input nodes) and removes them.
    """

    def run(self, dag: DAGCircuit) -> DAGCircuit:
        """
        Run the pass on the DAGCircuit.

        Parameters
        ----------
        dag : DAGCircuit
            The DAGCircuit to be transformed.

        Returns
        -------
        DAGCircuit
            The transformed DAGCircuit with initial RZ gates removed.
        """
        for node in dag.op_nodes():
            if isinstance(node.op, RZGate):
                predecessors = dag.predecessors(node)
                if isinstance(list(predecessors)[0], DAGInNode):
                    dag.remove_op_node(node)
        return dag


class MergeConsecutiveRotations(TransformationPass):
    """
    A transpiler pass to merge consecutive rotations of the same type on the same qubit.

    This pass identifies consecutive RX, RY, or RZ gates on the same qubit and merges them
    into a single rotation gate by combining their parameters.
    """

    def run(self, dag: DAGCircuit) -> DAGCircuit:
        """
        Run the pass on the DAGCircuit.

        Parameters
        ----------
        dag : DAGCircuit
            The DAGCircuit to be transformed.

        Returns
        -------
        DAGCircuit
            The transformed DAGCircuit with consecutive rotations merged.
        """
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

class ReplaceConsecutiveRotationsWithRxRzRx(TransformationPass):
    """
    A transpiler pass to replace 3 or more consecutive rotations with an RxRzRx sequence.

    This pass identifies sequences of 3 or more consecutive RX, RY, or RZ gates on the same qubit
    and replaces them with a generic RxRzRx sequence. The parameters of the RxRzRx gates are
    arbitrary and can be adjusted in a subsequent pass.
    """

    def run(self, dag: DAGCircuit) -> DAGCircuit:
        """
        Run the pass on the DAGCircuit.

        Parameters
        ----------
        dag : DAGCircuit
            The DAGCircuit to be transformed.

        Returns
        -------
        DAGCircuit
            The transformed DAGCircuit with consecutive rotations replaced by RxRzRx sequences.
        """
        for qubit in dag.qubits:
            rotation_sequence = []
            nodes_to_remove = []
            parameters_present = []
            for node in list(dag.nodes_on_wire(qubit)):
                if isinstance(node, DAGOpNode): 
                    if isinstance(node.op, (RXGate, RYGate, RZGate)):
                        rotation_sequence.append(node)
                        parameters_present.append(node.op.params[0])
                        nodes_to_remove.append(node)
                else:
                    if len(rotation_sequence) >= 3:
                        for node in nodes_to_remove[1:]:
                            dag.remove_op_node(node)
                        replacement_circ = QuantumCircuit(1)
                        replacement_circ.rx(parameters_present[0], 0)
                        replacement_circ.rz(parameters_present[1], 0)
                        replacement_circ.rx(parameters_present[2], 0)
                        remaining_node = nodes_to_remove[0]
                        replacement_dag = circuit_to_dag(replacement_circ)
                        dag.substitute_node_with_dag(remaining_node, replacement_dag)
                        
        return dag
    
custom_pass_manager = PassManager([
    RemoveUnnecessaryControlledGates(),
    RemoveInitialRZ(),
    MergeConsecutiveRotations(),
    ReplaceConsecutiveRotationsWithRxRzRx()
])    
    
def change_circuit_parameters(circ: QuantumCircuit) -> QuantumCircuit:
    num_params = circ.num_parameters
    new_vector = ParameterVector("θ", num_params)
    param_dict = {param: new_vector[i] for i, param in enumerate(circ.parameters)}
    
    return circ.assign_parameters(param_dict, inplace=False), new_vector
