import random, logging

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector, CircuitInstruction, Qubit
from qiskit.transpiler.passes import RemoveBarriers
from qiskit.circuit.library.standard_gates import get_standard_gate_name_mapping

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
        operator_pool: list[str] = None
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
        self.track_history = track_history
        
        ansatz_no_barriers = RemoveBarriers()(initial_ansatz)
        # Extract existing parameters
        existing_params = ansatz_no_barriers.parameters
        num_params = len(existing_params)

        # Create a ParameterVector and replace parameters
        self.param_vector = ParameterVector("θ", num_params)
        param_map = {p: self.param_vector[i] for i, p in enumerate(existing_params)}
        
        self.current_ansatz = ansatz_no_barriers.assign_parameters(param_map)
        self.history: list[QuantumCircuit] = [self.current_ansatz] if track_history else []
        
    def add_gate_at_index(self, gate_name: str, index: int, qubits: list[int] | list[Qubit]) -> None:
        """
        Add a gate from the instruction map at a specific index in the circuit data and adjust
        ParameterVector accordingly.

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
        
        # Resize parameter vector if needed
        if instruction.params:
            self.param_vector.resize(len(self.param_vector) + 1)
            new_param = self.param_vector[-1]

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
        
    def remove_gate_by_index(self, indices: int | list[int]) -> None:
        """
        Remove one or multiple gates from the ansatz at specified indices and adjust ParameterVector 
        accordingly.

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
        
        param_indices_to_remove = []
        
        for index in indices:
            instruction = INSTRUCTION_MAP[self.current_ansatz.data[index].name]
            params = instruction.params
            
            # Remove the gate from the circuit
            del self.current_ansatz.data[index]
            
            # Track parameter indices to remove
            if params:
                param_indices_to_remove.append(index)
        
        # Remove parameters in descending order to avoid re-indexing issues
        for _ in sorted(set(param_indices_to_remove), reverse=True):
            self.param_vector.resize(len(self.param_vector) - 1)
        
        # Save state if tracking history
        self._save_state()

    def remove_random_gate(self, index: int) -> None:
        """
        Remove a gate from the ansatz by index.

        Parameters
        ----------
        index : int
            The index of the gate in the circuit to remove.
        """
        self.current_ansatz.data.pop(index)
        self._save_state()

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
        self.history = self.history[:-(steps + 1)]

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
    
    def update_parameter_vector(self, new_vector: ParameterVector) -> None:
        """
        Update the parameter vector.

        Parameters
        ----------
        new_vector : ParameterVector
            The new vector to be used.
        """
        self.param_vector = new_vector
