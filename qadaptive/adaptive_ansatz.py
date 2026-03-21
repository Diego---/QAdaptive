import random, logging

from qiskit import QuantumCircuit
from qiskit.circuit import Parameter, CircuitInstruction, Qubit
from qiskit.transpiler.passes import RemoveBarriers
from qiskit.circuit.library.standard_gates import get_standard_gate_name_mapping

from qadaptive.operator_pool import DEFAULT_BLOCK_POOL, PoolBlock

logger = logging.getLogger(__name__)

INSTRUCTION_MAP = get_standard_gate_name_mapping()

class AdaptiveAnsatz:
    """
    A class representing a dynamically evolving quantum ansatz.

    This class tracks modifications to a parameterized quantum circuit, allowing
    gates to be added or removed iteratively while maintaining a history of previous
    ansatz states.
    """

    def __init__(
        self, 
        initial_ansatz: QuantumCircuit, 
        track_history: bool = True,
        operator_pool: list[str] | None = None,
        block_pool: dict[str, PoolBlock] | None = None,
        ) -> None:
        """
        Initialize the AdaptiveAnsatz.

        Parameters
        ----------
        initial_ansatz : QuantumCircuit
            The starting quantum circuit to be adapted over time.
        track_history : bool, optional
            Whether to keep track of previous ansatz versions (default: True).
        operator_pool : list[str], optional
            List of quantum instruction names from which to sample to mutate the ansatz.
            Defaults to None and the default pool ['rx', 'ry', 'rz', 'rzz'] is used.
        """
        
        if operator_pool is None:
            operator_pool = ['rx', 'ry', 'rz', 'cz']
        else:
            incorrect_gate_names = [gate_name for gate_name in operator_pool 
                                    if gate_name not in INSTRUCTION_MAP]
            assert not incorrect_gate_names,(
                f"The following gates are not part of the starndard gates: {incorrect_gate_names}"
            )
            
        self.operator_pool = operator_pool
        self.block_pool = DEFAULT_BLOCK_POOL if block_pool is None else block_pool
        self.track_history = track_history
        
        ansatz_no_barriers = RemoveBarriers()(initial_ansatz)
        # Re-arrange parameters
        ansatz_re_arranged_params = AdaptiveAnsatz.re_arrange_gate_params(ansatz_no_barriers)
        self.current_ansatz = ansatz_re_arranged_params

        # Create a list of parameters
        self.params: list[Parameter] = [Parameter(f"θ_{i}") for i in range(ansatz_re_arranged_params.num_parameters)]
        
        # Initialize a list to track the history
        self.history: list[QuantumCircuit] = [self.current_ansatz.copy()] if track_history else []

    @staticmethod
    def re_arrange_gate_params(circuit: QuantumCircuit) -> QuantumCircuit:
        """
        Reassigns the parameters of a quantum circuit to a new, ordered list of parameters.

        This is useful when you want to standardize or reorder the parameters in a circuit,
        for example to allow consistent mapping and optimization, especially after modifying
        the circuit structure (e.g., adding/removing gates).

        The method constructs a new list of `Parameter` of the same length as the number of
        parameters in the input circuit, and maps existing parameters to new parameters
        indexed in order of appearance.

        Parameters
        ----------
        circuit : QuantumCircuit
            The quantum circuit whose parameters should be reassigned.

        Returns
        -------
        QuantumCircuit
            A new quantum circuit with the same structure but with parameters replaced
            by a new list of `Parameter`s.
        """
        existing_params = circuit.parameters
        num_params = circuit.num_parameters
        
        params = [Parameter(f"θ_{i}") for i in  range(num_params)]
        param_map = {p: params[i] for i, p in enumerate(existing_params)}
        
        return circuit.assign_parameters(param_map)
    
    def _new_parameters(self, n: int) -> list[Parameter]:
        """
        Create and register `n` fresh parameters.

        Parameters
        ----------
        n : int
            Number of new parameters.

        Returns
        -------
        list[Parameter]
            The newly created parameters.
        """
        if n < 0:
            raise ValueError("Number of new parameters must be non-negative.")

        if not self.params:
            start = 0
        else:
            start = max(int(p.name.split("_")[1]) for p in self.params) + 1

        new_params = [Parameter(f"θ_{i}") for i in range(start, start + n)]
        self.params.extend(new_params)
        return new_params
        
    def add_gate_at_index(self, gate_name: str, index: int, qubits: list[int] | list[Qubit]) -> None:
        """
        Add a gate from the instruction map at a specific index in the circuit data and adjust the
        `Parameter` list accordingly.

        Parameters
        ----------
        gate_name : str
            The name of the gate to be added.
        index : int
            The index in the circuit data where the gate should be inserted.
        qubits : list[int]
            The qubits on which the gate should act.
        Raises
        ------
        IndexError
            If the index is out of range.
        AssertionError
            If the gate being added is not a part of qiskit's standard gates.
        """
        if index < -1 or index >= len(self.current_ansatz.data) + 1:
            raise IndexError("Gate index out of range.")
        
        assert gate_name in INSTRUCTION_MAP, f"Gate {gate_name} is not a recognized standard gate."

        instruction = INSTRUCTION_MAP[gate_name]
        
        # Resize parameter list if needed, always adding a new index
        if instruction.params:
            new_param = self._new_parameters(1)[0]
            # Create gate instruction
            instruction_with_params = instruction.copy()
            instruction_with_params.params = [new_param]
        else:
            instruction_with_params = instruction.copy()

        qubit_tuple = tuple(qubits)
        
        # Insert into circuit data
        self.current_ansatz.data.insert(index, CircuitInstruction(
            operation=instruction_with_params,
            qubits=qubit_tuple,
            clbits=[]
        ))
        
        # Save state if tracking history
        self._save_state()
        
    def add_block_at_index(
        self,
        block_name: str,
        index: int,
        ansatz_qubits: list[int] | list[Qubit],
    ) -> None:
        """
        Insert a parametrized block from the block pool at a specific circuit index.

        Parameters
        ----------
        block_name : str
            Name of the block in `self.block_pool`.
        index : int
            Circuit-data index at which the block should be inserted.
        ansatz_qubits : list[int] | list[Qubit]
            Qubits from the ansatz on which the block acts.

        Raises
        ------
        IndexError
            If the index is out of range.
        AssertionError
            If the block name is unknown.
        ValueError
            If the wrong number of qubits is provided.
        """

        if index < 0 or index > len(self.current_ansatz.data):
            raise IndexError("Block index out of range.")

        assert block_name in self.block_pool, (
            f"Block '{block_name}' is not part of the available block pool: "
            f"{list(self.block_pool.keys())}."
        )

        block = self.block_pool[block_name]

        if len(ansatz_qubits) != block.num_qubits:
            raise ValueError(
                f"Block '{block_name}' acts on {block.num_qubits} qubits, "
                f"but got {len(ansatz_qubits)}."
            )

        new_params = self._new_parameters(block.num_parameters)
        block_circuit = block.build(new_params)

        local_to_target = {
            block_circuit.qubits[i]: ansatz_qubits[i] for i in range(block.num_qubits)
        }

        for offset, inst in enumerate(block_circuit.data):
            mapped_qubits = tuple(local_to_target[q] for q in inst.qubits)
            self.current_ansatz.data.insert(
                index + offset,
                CircuitInstruction(
                    operation=inst.operation.copy(),
                    qubits=mapped_qubits,
                    clbits=list(inst.clbits),
                ),
            )

        self._save_state()
            
    def add_random_gate(self) -> tuple[str, ]:
        """
        Add a randomly selected gate from the operator pool at a random index.
        """
        if not self.current_ansatz.data:
            index = 0
        else:
            index = random.randint(0, len(self.current_ansatz.data))
        
        gate_name = random.choice(self.operator_pool)
        qubits = random.sample(self.current_ansatz.qubits, INSTRUCTION_MAP[gate_name].num_qubits)
        
        self.add_gate_at_index(gate_name, index, qubits)
        
        return gate_name, qubits, index
    
    def add_random_block(self) -> tuple[str, list[Qubit], int]:
        """
        Insert a random block from the block pool at a random position.
        """
        if not self.current_ansatz.data:
            index = 0
        else:
            index = random.randint(0, len(self.current_ansatz.data))

        block_name = random.choice(list(self.block_pool.keys()))
        block = self.block_pool[block_name]
        qubits = random.sample(self.current_ansatz.qubits, block.num_qubits)

        self.add_block_at_index(block_name, index, qubits)
        return block_name, qubits, index
        
    def remove_gate_by_index(self, indices: int | list[int]) -> None:
        """
        Remove one or multiple gates from the ansatz at specified indices and adjust the 
        `Parameter` list accordingly.

        Parameters
        ----------
        indices : int or list[int]
            The index or indices of the gates in the circuit data to remove.

        Raises
        ------
        IndexError
            If any index is out of range.
        """
        if isinstance(indices, int):
            indices = [indices]

        if any(idx < 0 or idx >= len(self.current_ansatz.data) for idx in indices):
            raise IndexError("One or more gate indices are out of range.")
        
        # Sort indices in descending order to prevent shifting issues when deleting
        indices.sort(reverse=True)
        
        removed_param_names = set()
        
        for index in indices:
            operation = self.current_ansatz.data[index].operation
            
            for param in operation.params:
                # Case 1: plain Parameter
                if isinstance(param, Parameter):
                    removed_param_names.add(param.name)
                # Case 2: ParameterExpression or similar symbolic object
                elif hasattr(param, "parameters"):
                    for subparam in param.parameters:
                        removed_param_names.add(subparam.name)
                # Case 3: numeric value (float, int, etc.) -> ignore
            
            # Remove the gate from the circuit
            del self.current_ansatz.data[index]
            
        
        # Update parameters
        self.update_params()
        # Save state if tracking history
        self._save_state()

    def remove_random_gate(self) -> None:
        """
        Remove a gate randomly.
        """
        # TODO: Implement
        pass

    def _save_state(self) -> None:
        """Save the current ansatz to history if tracking is enabled."""
        if self.track_history:
            self.history.append(self.current_ansatz.copy())

    def rollback(self, steps: int = 1) -> None:
        """
        Roll back the ansatz to a previous version.

        Parameters
        ----------
        steps : int, optional
            Number of steps to roll back (default: 1).

        Raises
        ------
        ValueError
            If attempting to roll back beyond stored history.
        """
        if not self.track_history or len(self.history) < steps + 1:
            raise ValueError("Cannot rollback beyond available history.")
        self.current_ansatz = self.history[-(steps + 1)].copy()
        self.history = self.history[:-(steps)]
        self.update_params()

    def get_current_ansatz(self) -> QuantumCircuit:
        """
        Get the current ansatz.

        Returns
        -------
        QuantumCircuit
            The current version of the quantum circuit.
        """
        return self.current_ansatz
    
    def update_ansatz(self, new_ansatz: QuantumCircuit) -> None:
        """
        Update the current ansatz.
        
        Parameters
        ----------
        new_ansatz : QuantumCircuit
            The new ansatz.
        """
        
        self.current_ansatz = new_ansatz.copy()

    def update_params(self) -> None:
        """
        Synchronize `self.params` with the parameters currently present in `current_ansatz`.
        """
        self.params = sorted(
            list(self.current_ansatz.parameters),
            key=lambda p: int(p.name.split("_")[1])
    )

    def copy(self) -> "AdaptiveAnsatz":
        """
        Return a deep copy of the AdaptiveAnsatz.

        Returns
        -------
        AdaptiveAnsatz
            Independent copy of the current adaptive ansatz object.
        """
        new_obj = AdaptiveAnsatz(
            initial_ansatz=self.current_ansatz.copy(),
            track_history=self.track_history,
            operator_pool=list(self.operator_pool),
            block_pool=dict(self.block_pool),
        )

        # Preserve history independently.
        if self.track_history:
            new_obj.history = [qc.copy() for qc in self.history]
        else:
            new_obj.history = []

        return new_obj
