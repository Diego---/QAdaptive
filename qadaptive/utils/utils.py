from typing import Callable

from qiskit.circuit.library import RZGate, RXGate, RYGate
from qiskit.circuit import QuantumCircuit, Parameter
from qiskit.dagcircuit import DAGCircuit, DAGInNode, DAGOpNode, DAGOutNode
from qiskit.transpiler import PassManager
from qiskit.transpiler.basepasses import TransformationPass
from qiskit.transpiler.passes import CommutativeCancellation, CommutativeInverseCancellation
from qiskit.converters import circuit_to_dag

class RemoveInputControlledGates(TransformationPass):
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
    
class RemoveFinalRZ(TransformationPass):
    """
    A transpiler pass to remove final RZ gates from a DAGCircuit.

    This pass identifies RZ gates that are applied at the end of the circuit
    (i.e., their successor are output nodes) and removes them.
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
                successors = dag.successors(node)
                if isinstance(list(successors)[0], DAGOutNode):
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

class ReplaceConsecutiveRotationsWithRxRyRx(TransformationPass):
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
                    # Only aggregate parameterized gates, which assumes that other gates are there for a purpose.
                    if isinstance(node.op, (RXGate, RYGate, RZGate)) and isinstance(node.op.params[0], Parameter):
                        rotation_sequence.append(node)
                        parameters_present.append(node.op.params[0])
                        nodes_to_remove.append(node)
                    else:
                        if len(rotation_sequence) >= 3:
                            for node in nodes_to_remove[1:]:
                                dag.remove_op_node(node)
                            replacement_circ = QuantumCircuit(1)
                            # Which parameters are used is irrelevant, as they will be changed
                            replacement_circ.rx(parameters_present[0], 0)
                            replacement_circ.ry(parameters_present[1], 0)
                            replacement_circ.rx(parameters_present[2], 0)
                            remaining_node = nodes_to_remove[0]
                            replacement_dag = circuit_to_dag(replacement_circ)
                            dag.substitute_node_with_dag(remaining_node, replacement_dag)
                        # Reset counters
                        rotation_sequence = []
                        nodes_to_remove = []
                        parameters_present = []
                else:
                    # End of the circuit, check for the final rotations
                    if len(rotation_sequence) >= 3:
                        for node in nodes_to_remove[1:]:
                            dag.remove_op_node(node)
                        replacement_circ = QuantumCircuit(1)
                        replacement_circ.rx(parameters_present[0], 0)
                        replacement_circ.ry(parameters_present[1], 0)
                        replacement_circ.rx(parameters_present[2], 0)
                        remaining_node = nodes_to_remove[0]
                        replacement_dag = circuit_to_dag(replacement_circ)
                        dag.substitute_node_with_dag(remaining_node, replacement_dag)
                        
        return dag

def custom_pass_manager(
    remove_initial_rz: bool = False,
    remove_final_rz: bool = False,
    remove_input_controlled_gates: bool = False,
) -> PassManager:
    """
    Create a custom PassManager with a configurable sequence of transpilation passes.

    Parameters
    ----------
    remove_initial_rz : bool, optional
        Whether to include the pass that removes initial RZ gates.
    remove_final_rz : bool, optional
        Whether to include the pass that removes final RZ gates.
    remove_input_controlled_gates : bool, optional
        Whether to include the pass that removes unnecessary controlled gates.

    Returns
    -------
    PassManager
        The custom PassManager with the selected sequence of passes.
    """
    passes = [
        CommutativeCancellation(),
    ]

    if remove_initial_rz:
        passes.append(RemoveInitialRZ())

    if remove_final_rz:
        passes.append(RemoveFinalRZ())

    if remove_input_controlled_gates:
        passes.append(RemoveInputControlledGates())

    passes.extend([
        MergeConsecutiveRotations(),
        ReplaceConsecutiveRotationsWithRxRyRx(),
        CommutativeCancellation(),
        CommutativeInverseCancellation(),
    ])

    return PassManager(passes)

def create_callback_args(
    json_file_path: str | None = None,
    store_data: bool = False,
    extra_eval_freq: int | None = None,
    cost_extra: Callable | None = None,
    plot: bool = True,
    use_epoch: bool = False
) -> dict:
    """
    Generate the arguments for create_live_plot_callback.
    
    Parameters
    ----------
    json_file_path : str, optional
        Path to JSON file for storing data.
    store_data : bool, optional
        Whether to store data. Defaults to False.
    extra_eval_freq : int, optional
        Frequency of extra cost evaluations.
    cost_extra : Callable, optional
        Extra cost function to evaluate.
    plot : bool, optional
        Whether to plot results. Defaults to True.
    use_epoch : bool, optional
        Whether to use epoch tracking. Defaults to False.
    
    Returns
    -------
    dict
        A dictionary with the arguments for the callback builder.
    """
    counts = []
    values = []
    params = []
    stepsize = []
    values_extra = [] if extra_eval_freq is not None else None
    
    args = {
        "counts": counts,
        "values": values,
        "params": params,
        "stepsize": stepsize,
        "json_file_path": json_file_path,
        "store_data": store_data,
        "extra_eval_freq": extra_eval_freq,
        "cost_extra": cost_extra,
        "values_extra": values_extra,
        "plot": plot,
        "use_epoch": use_epoch
    }
    
    return args
