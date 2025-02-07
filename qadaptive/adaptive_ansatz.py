from qiskit import QuantumCircuit
from qiskit.circuit.library.standard_gates import get_standard_gate_name_mapping

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
            operator_pool = ['rx', 'ry', 'rz', 'rzz']
        else:
            incorrect_gate_names = [gate_name for gate_name in operator_pool 
                                    if gate_name not in INSTRUCTION_MAP]
            assert not incorrect_gate_names,(
                f"The following gates are not part of the starndard gates: {incorrect_gate_names}"
            )
        
        self.current_ansatz = initial_ansatz.copy()
        self.track_history = track_history
        self.history: list[QuantumCircuit] = [initial_ansatz.copy()] if track_history else []

    def add_gate(self, gate, qubits: list[int]) -> None:
        """
        Add a gate to the ansatz.

        Parameters
        ----------
        gate : qiskit.circuit.Instruction
            The quantum gate to be added.
        qubits : list[int]
            The qubits on which the gate should act.
        """
        self.current_ansatz.append(gate, qubits)
        self._save_state()

    def remove_gate(self, index: int) -> None:
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
