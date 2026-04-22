import re
from typing import Callable

from qiskit.circuit.library import RZGate, RXGate, RYGate
from qiskit.circuit import QuantumCircuit, Parameter, ParameterExpression
from qiskit.dagcircuit import DAGCircuit, DAGInNode, DAGOpNode, DAGOutNode
from qiskit.transpiler import PassManager
from qiskit.transpiler.basepasses import TransformationPass
from qiskit.transpiler.passes import CommutativeCancellation, CommutativeInverseCancellation
from qiskit.converters import circuit_to_dag

_THETA_PATTERN = re.compile(r"^θ_(\d+)$")

def _theta_index(param) -> int:
    if not isinstance(param, Parameter):
        raise ValueError(f"Expected plain Parameter, got {type(param).__name__}")
    match = _THETA_PATTERN.match(param.name)
    if match is None:
        raise ValueError(f"Parameter name '{param.name}' does not match θ_i")
    return int(match.group(1))

def _node_theta_index(node) -> int:
    if not node.op.params:
        raise ValueError("Rotation node has no parameters.")
    return _theta_index(node.op.params[0])

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
                predecessors = list(dag.predecessors(node))
                if predecessors and isinstance(predecessors[0], DAGInNode):
                    dag.remove_op_node(node)
        return dag

class RemoveFinalRZ(TransformationPass):
    """
    A transpiler pass to remove final RZ gates from a DAGCircuit.

    This pass identifies RZ gates that are applied at the end of the circuit
    (i.e., their successors are output nodes) and removes them.
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
            The transformed DAGCircuit with final RZ gates removed.
        """
        for node in dag.op_nodes():
            if isinstance(node.op, RZGate):
                successors = list(dag.successors(node))
                if successors and isinstance(successors[0], DAGOutNode):
                    dag.remove_op_node(node)
        return dag

class MergeConsecutiveRotations(TransformationPass):
    """
    A transpiler pass to merge consecutive rotations of the same type on the same qubit.

    This pass identifies consecutive RX, RY, or RZ gates on the same qubit and merges them
    into a single rotation gate by keeping the oldest parameter.

    Example: RX(θ_0) followed by RX(θ_1) on the same qubit will be merged into a single RX(θ_0).

    Notes
    -----
    This pass is intentionally heuristic and does not in general preserve the exact
    implemented unitary.
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
            run = []
            run_type = None

            for node in list(dag.nodes_on_wire(qubit)):
                if isinstance(node, DAGOpNode) and isinstance(node.op, (RXGate, RYGate, RZGate)):
                    node_type = type(node.op)

                    if run_type is None or node_type == run_type:
                        run.append(node)
                        run_type = node_type
                    else:
                        self._reduce_run_keep_oldest(dag, run)
                        run = [node]
                        run_type = node_type
                else:
                    self._reduce_run_keep_oldest(dag, run)
                    run = []
                    run_type = None

            self._reduce_run_keep_oldest(dag, run)

        return dag

    @staticmethod
    def _reduce_run_keep_oldest(dag: DAGCircuit, run: list) -> None:
        if len(run) <= 3:
            return

        survivor = min(run, key=_node_theta_index)
        for node in run:
            if node is not survivor:
                dag.remove_op_node(node)

class ReplaceConsecutiveRotationsWithRxRyRx(TransformationPass):
    """
    A transpiler pass to replace 3 or more consecutive rotations with an RxRyRx sequence.

    This pass identifies sequences of 3 or more consecutive RX, RY, or RZ gates on the same qubit
    and replaces them with a generic RxRyRx sequence. The parameters of the replacement gates are
    chosen as the oldest parameters.

    Notes
    -----
    This pass is intentionally heuristic and does not in general preserve the exact
    implemented unitary.
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
            The transformed DAGCircuit with consecutive rotations replaced by RxRyRx sequences.
        """
        for qubit in dag.qubits:
            rotation_sequence = []

            for node in list(dag.nodes_on_wire(qubit)):
                if (
                    isinstance(node, DAGOpNode)
                    and isinstance(node.op, (RXGate, RYGate, RZGate))
                    and isinstance(node.op.params[0], Parameter)
                ):
                    rotation_sequence.append(node)
                else:
                    self._replace_run_keep_three_oldest(dag, rotation_sequence)
                    rotation_sequence = []

            self._replace_run_keep_three_oldest(dag, rotation_sequence)

        return dag

    @staticmethod
    def _replace_run_keep_three_oldest(dag: DAGCircuit, run: list) -> None:
        if len(run) < 3:
            return

        oldest_three = sorted(run, key=_node_theta_index)[:3]
        oldest_three_params = [node.op.params[0] for node in oldest_three]

        anchor = run[0]
        for node in run[1:]:
            dag.remove_op_node(node)

        replacement_circ = QuantumCircuit(1)
        replacement_circ.rx(oldest_three_params[0], 0)
        replacement_circ.ry(oldest_three_params[1], 0)
        replacement_circ.rx(oldest_three_params[2], 0)

        replacement_dag = circuit_to_dag(replacement_circ)
        dag.substitute_node_with_dag(anchor, replacement_dag)


class ExactMergeConsecutiveRotations(TransformationPass):
    """
    Exactly merge consecutive same-axis one-qubit rotations.

    This pass preserves the implemented unitary by using the exact identities

        RX(a) RX(b) = RX(a + b)
        RY(a) RY(b) = RY(a + b)
        RZ(a) RZ(b) = RZ(a + b)

    on consecutive rotations of the same axis on the same qubit.
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
            The transformed DAGCircuit with exact merges applied.
        """
        for qubit in dag.qubits:
            run: list = []
            run_type = None

            for node in list(dag.nodes_on_wire(qubit)):
                if isinstance(node, DAGOpNode) and isinstance(node.op, (RXGate, RYGate, RZGate)):
                    node_type = type(node.op)

                    if run_type is None or node_type == run_type:
                        run.append(node)
                        run_type = node_type
                    else:
                        self._merge_run_exact(dag, run, run_type)
                        run = [node]
                        run_type = node_type
                else:
                    self._merge_run_exact(dag, run, run_type)
                    run = []
                    run_type = None

            self._merge_run_exact(dag, run, run_type)

        return dag

    @staticmethod
    def _merge_run_exact(
        dag: DAGCircuit,
        run: list,
        run_type,
    ) -> None:
        if len(run) <= 1 or run_type is None:
            return

        merged_param: Parameter | ParameterExpression = run[0].op.params[0]
        for node in run[1:]:
            merged_param = merged_param + node.op.params[0]

        replacement_circ = QuantumCircuit(1)
        if run_type is RXGate:
            replacement_circ.rx(merged_param, 0)
        elif run_type is RYGate:
            replacement_circ.ry(merged_param, 0)
        elif run_type is RZGate:
            replacement_circ.rz(merged_param, 0)
        else:
            return

        anchor = run[0]
        for node in run[1:]:
            dag.remove_op_node(node)

        replacement_dag = circuit_to_dag(replacement_circ)
        dag.substitute_node_with_dag(anchor, replacement_dag)

def custom_pass_manager(
    remove_initial_rz: bool = False,
    remove_final_rz: bool = False,
    remove_input_controlled_gates: bool = False,
) -> PassManager:
    """
    Create an aggressive heuristic PassManager.

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
        Aggressive heuristic pass manager.

    Notes
    -----
    This pass manager is intended to keep the ansatz compact, but it does not
    in general preserve the exact implemented unitary.
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


def unitary_preserving_pass_manager() -> PassManager:
    """
    Create a conservative PassManager that preserves mathematical equivalence
    of the implemented unitary.

    Returns
    -------
    PassManager
        Exact simplification pass manager.

    Notes
    -----
    This pass manager avoids heuristic parameter-dropping passes and only applies
    exact same-axis rotation merges together with exact cancellations.
    """
    return PassManager([
        ExactMergeConsecutiveRotations(),
        CommutativeCancellation(),
        CommutativeInverseCancellation(),
        ExactMergeConsecutiveRotations(),
        CommutativeCancellation(),
    ])
