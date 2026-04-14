import numpy as np

from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister
from qiskit.quantum_info import (
    Statevector,
    DensityMatrix,
    Statevector,
    Operator,
    SparsePauliOp,
    random_statevector,
    partial_trace,
    state_fidelity
)
from qiskit_aer import AerSimulator
from qiskit_aer.primitives import SamplerV2

loc_sim = AerSimulator(method="qasm_simulator")
sim_sampler = SamplerV2(loc_sim)

def prepare_seven_qubit_encoder() -> QuantumCircuit:
    """
    Prepare the 7-qubit Steane encoder circuit.

    This follows the encoder construction shown in Fig. 12 of
    Mondal and Parhi, where the logical input qubit is placed on
    the last wire and the first six qubits are initialized in |0>.
    The implementation uses the stabilizer-based construction
    described in the paper.

    Qubit layout
    ------------
    q[0], ..., q[5]
        Ancilla/data qubits initialized in |0>.
    q[6]
        Input logical qubit to be encoded.

    Returns
    -------
    QuantumCircuit
        A 7-qubit encoding circuit mapping an input state on q[6]
        into the Steane code space.
    """
    qc = QuantumCircuit(7, name="7q_encode")

    qc.cx(6, 4)
    qc.cx(6, 5)
    
    qc.h(0)
    qc.h(1)
    qc.h(2)
    
    qc.cx(0, 3)
    qc.cx(1, 3)
    qc.cx(0, 5)
    qc.cx(2, 3)
    qc.cx(1, 4)
    qc.cx(0, 6)
    qc.cx(2, 4)
    qc.cx(1, 6)
    qc.cx(2, 5)

    return qc

def encode_state_with_seven_qubit_encoder(state: QuantumCircuit) -> QuantumCircuit:
    """Apply the 7-qubit Steane encoder to an input state on q6."""
    encoder = prepare_seven_qubit_encoder()
    if not isinstance(state, QuantumCircuit):
        raise ValueError("Input must be a 7-qubit QuantumCircuit.")
    assert state.num_qubits == 7, "Input state must have 7 qubits with desired state on q6."
    return state.compose(encoder)

def prepare_zero_logical_steane() -> QuantumCircuit:
    """Prepares the logical |0> state for the Steane [[7,1,3]] code."""
    qc = QuantumCircuit(7, name="|0>_L Steane")
    return encode_state_with_seven_qubit_encoder(qc)

def prepare_one_logical_steane() -> QuantumCircuit:
    """Prepares logical |1> by applying transversal X to logical |0>."""
    qc = prepare_zero_logical_steane()
    qc.x(range(7)) # Transversal X
    qc.name = "|1>_L Steane"
    return qc

def prepare_plus_logical_steane() -> QuantumCircuit:
    """Prepares logical |+> by applying transversal H to logical |0>."""
    qc = prepare_zero_logical_steane()
    qc.h(range(7)) # Transversal H
    qc.name = "|+>_L Steane"
    return qc

def prepare_five_qubit_encoder(name="5q_encode") -> QuantumCircuit:
    """
    Prepare the encoder for the 5-qubit [[5,1,3]] code.

    This follows the modified encoder shown in Fig. 10 of the article.
    The logical input qubit is placed on the last wire q[4], and the
    first four qubits are initialized in |0>.

    Returns
    -------
    QuantumCircuit
        A 5-qubit encoding circuit.
    """
    qc = QuantumCircuit(7, name=name)

    qc.h(0)
    qc.s(0)
    qc.cy(0, 4)

    qc.h(1)
    qc.cx(1, 4)

    qc.h(2)
    qc.cz(2, 0)
    qc.cz(2, 1)
    qc.cx(2, 4)

    qc.h(3)
    qc.s(3)
    qc.cz(3, 0)
    qc.cz(3, 2)
    qc.cy(3, 4)

    return qc


def prepare_five_qubit_decoder() -> QuantumCircuit:
    """
    Prepare the decoder for the 5-qubit [[5,1,3]] code.

    This is the inverse of the encoder and maps the logical information
    back onto qubit 4.

    Returns
    -------
    QuantumCircuit
        A 5-qubit decoding circuit.
    """
    return prepare_five_qubit_encoder(name="5q_decod").inverse()

zero_physical = QuantumCircuit(7)
one_physical = QuantumCircuit(7)
one_physical.x(6)
plus_physical = QuantumCircuit(7)
plus_physical.h(6)

zero_logical_steane: QuantumCircuit = prepare_zero_logical_steane()
one_logical_steane: QuantumCircuit = prepare_one_logical_steane()
one_logical_steane_encoded: QuantumCircuit = encode_state_with_seven_qubit_encoder(one_physical)
plus_logical_steane: QuantumCircuit = prepare_plus_logical_steane()
plus_logical_steane_encoded: QuantumCircuit = encode_state_with_seven_qubit_encoder(plus_physical)

initial_logical_states = [zero_logical_steane, one_logical_steane, plus_logical_steane]

five_qubit_encoder: QuantumCircuit = prepare_five_qubit_encoder()
five_qubit_decoder: QuantumCircuit = prepare_five_qubit_decoder()

# Final rotations should be on q4, since that's where the decoded 
# state should be
final_rotation_0 = QuantumCircuit(7)
final_rotation_1 = QuantumCircuit(7)
final_rotation_1.x(4)
final_rotation_plus = QuantumCircuit(7)
final_rotation_plus.h(4)

final_rotations = [final_rotation_0, final_rotation_1, final_rotation_plus]

def verify_encoder_decoder(
    encoder: QuantumCircuit,
    decoder: QuantumCircuit,
    num_tests: int = 5,
    atol: float = 1e-8,
) -> None:
    """
    Verify that decoder composed with encoder acts as identity
    on the logical input qubit.

    Parameters
    ----------
    encoder : QuantumCircuit
        Encoding circuit on 5 qubits.
    decoder : QuantumCircuit
        Decoding circuit on 5 qubits.
    num_tests : int, optional
        Number of random single-qubit input states to test.
    atol : float, optional
        Absolute numerical tolerance for numerical comparisons.
    """
    for i in range(num_tests):
        psi = random_statevector(2)

        # Prepare |psi> on qubit 0 and |0000> on qubits 1-4
        prep = QuantumCircuit(5)
        prep.initialize(psi.data, 0)
        initial = Statevector.from_instruction(prep)

        # Apply encoder, then decoder
        final = initial.evolve(encoder).evolve(decoder)

        # Trace out qubits 1,2,3,4 and keep qubit 0
        reduced = partial_trace(final, [1, 2, 3, 4])

        # Expected one-qubit density matrix
        target = DensityMatrix(psi)

        if not np.allclose(reduced.data, target.data, atol=atol):
            raise AssertionError(
                f"Test {i} failed.\n"
                f"Input statevector:\n{psi.data}\n\n"
                f"Reduced decoded state:\n{reduced.data}\n\n"
                f"Expected state:\n{target.data}"
            )

    print(f"All {num_tests} tests passed.")

def check_eigenstate(state: Statevector, operator: Operator, atol=1e-8, sign=1) -> bool:
    """Check if state is +1 eigenstate of operator."""
    new_state = state.evolve(operator)
    overlap = np.vdot(state.data, new_state.data)
    return np.isclose(overlap, sign*1.0, atol=atol)


def verify_steane_state(prep_circuit: QuantumCircuit, state_name:str="0L", atol: float = 1e-8) -> dict:
    """
    Verify that the state prepared by ``prep_circuit`` is a valid Steane code state
    with the expected stabilizer and logical-operator eigenvalues.

    Parameters
    ----------
    prep_circuit : QuantumCircuit
        A circuit that prepares a candidate Steane code state on 7 qubits.
    state_name : str, optional
        Which logical state is expected. Must be one of:
        - "0L" : checks Z_L = +1
        - "1L" : checks Z_L = -1
        - "+L" : checks X_L = +1
    atol : float, optional
        Absolute tolerance used for eigenvalue checks.

    Returns
    -------
    dict
        Dictionary containing the measured expectation values.

    Raises
    ------
    ValueError
        If ``prep_circuit`` is not 7-qubit or if ``state_name`` is invalid.
    AssertionError
        If any stabilizer or logical eigenvalue check fails.

    Notes
    -----
    This function assumes the same Qiskit Pauli-string convention throughout:

    - In Qiskit, the *leftmost* character acts on the highest-index qubit.
    - For 7 qubits, the string "XXXXIII" means:
          X on q6, X on q5, X on q4, X on q3, I on q2, I on q1, I on q0

    The Steane stabilizers from https://arxiv.org/pdf/2309.11793 are therefore 
    encoded as:

        M1 = XXXXIII
        M2 = XXIIXXI
        M3 = XIXIXIX
        M4 = ZZZZIII
        M5 = ZZIIZZI
        M6 = ZIZIZIZ

    and logical operators as:

        X_L = XXXXXXX
        Z_L = ZZZZZZZ
    """
    if prep_circuit.num_qubits != 7:
        raise ValueError("prep_circuit must act on exactly 7 qubits.")

    valid_names = {"0L", "1L", "+L"}
    if state_name not in valid_names:
        raise ValueError(f"state_name must be one of {valid_names}, got {state_name!r}.")

    state = Statevector(prep_circuit)

    def expval(label: str) -> float:
        """Return the real expectation value of a Pauli string."""
        op = SparsePauliOp(label)
        val = state.expectation_value(op)
        return float(np.real_if_close(val))

    # Steane stabilizers in Qiskit label convention.
    stabilizers = {
        "M1": "IIIXXXX",
        "M2": "XXIIXXI",
        "M3": "XIXIXIX",
        "M4": "IIIZZZZ",
        "M5": "ZZIIZZI",
        "M6": "ZIZIZIZ",
    }

    # Logical operators.
    XL = "XXXXXXX"
    ZL = "ZZZZZZZ"

    results = {
        "state_name": state_name,
        "stabilizers": {},
        "logical": {},
    }

    # Check all stabilizers are +1.
    for name, label in stabilizers.items():
        value = expval(label)
        results["stabilizers"][name] = value
        assert np.isclose(value, 1.0, atol=atol), (
            f"Stabilizer {name} = {label} has expectation {value}, expected +1."
        )

    # Check the appropriate logical eigenvalue.
    if state_name == "0L":
        z_val = expval(ZL)
        results["logical"]["ZL"] = z_val
        assert np.isclose(z_val, 1.0, atol=atol), (
            f"Logical Z_L has expectation {z_val}, expected +1 for |0_L>."
        )

    elif state_name == "1L":
        z_val = expval(ZL)
        results["logical"]["ZL"] = z_val
        assert np.isclose(z_val, -1.0, atol=atol), (
            f"Logical Z_L has expectation {z_val}, expected -1 for |1_L>."
        )

    elif state_name == "+L":
        x_val = expval(XL)
        results["logical"]["XL"] = x_val
        assert np.isclose(x_val, 1.0, atol=atol), (
            f"Logical X_L has expectation {x_val}, expected +1 for |+_L>."
        )

    return results
    
def verify_steane_encoder_convention(
    encoder: QuantumCircuit,
    atol: float = 1e-8,
) -> None:
    """
    Check that the encoder maps |0> to |0_L> and |1> to |1_L>
    in the same convention as your verified logical-state routines.
    """
    # Encode |0>
    circ0 = QuantumCircuit(7)
    circ0.compose(encoder, inplace=True)
    sv_enc_0 = Statevector.from_instruction(circ0)

    sv_ref_0 = Statevector.from_instruction(prepare_zero_logical_steane())
    fid0 = state_fidelity(sv_enc_0, sv_ref_0)

    # Encode |1>
    circ1 = QuantumCircuit(7)
    circ1.x(6)
    circ1.compose(encoder, inplace=True)
    sv_enc_1 = Statevector.from_instruction(circ1)

    sv_ref_1 = Statevector.from_instruction(prepare_one_logical_steane())
    fid1 = state_fidelity(sv_enc_1, sv_ref_1)

    print(f"Fidelity encoded |0> vs |0_L>: {fid0:.12f}")
    print(f"Fidelity encoded |1> vs |1_L>: {fid1:.12f}")

    if fid0 < 1 - atol:
        raise AssertionError("Encoder does not match your |0_L> convention.")
    if fid1 < 1 - atol:
        raise AssertionError("Encoder does not match your |1_L> convention.")

    print("Steane encoder convention check passed.")
    
def validate_trained_code_switcher(
    trained_circ: QuantumCircuit, 
    use_random_states: bool = False,
    num_sampled_states: int = 10,
    atol: float = 1e-8
    ) -> list[tuple[QuantumCircuit, bool]]:
    """
    Validate that the trained code switcher circuit correctly maps arbitrary logical states
    of the 7-qubit Steane code to the corresponding states in the 5-qubit code.

    This function prepares ``num_sampled_states`` random one-qubit states, then encodes them 
    to the 7-qubit code, applies the trained code switcher, then decodes with the 5-qubit
    decoder and checks that the resulting single-qubit state (which lives on q4) matches
    the input states within the specified tolerance.

    Parameters
    ----------
    trained_circ : QuantumCircuit
        The trained code switcher circuit. It is assumed to act on 7 qubits and to be
        fully bound, i.e. it should not contain free parameters.
    use_random_states : bool, optional
        Whether to include random states in the validation. If False, only the logical |0>,
        |1>, and |+> states are tested.
    num_sampled_states : int, optional
        Number of random states to test.
    atol : float, optional
        Absolute tolerance for fidelity checks.
        
    Returns
    -------
    list[tuple[QuantumCircuit, bool]]
        A list of the tested input states to a boolean indicating whether 
        the code switcher passed the fidelity check for that state.
    """
    if trained_circ.num_qubits != 7:
        raise ValueError(
            f"'trained_circ' must act on 7 qubits, got {trained_circ.num_qubits}."
        )

    if trained_circ.parameters:
        raise ValueError(
            "'trained_circ' still contains free parameters. Pass a fully bound circuit."
        )

    steane_encoder = prepare_seven_qubit_encoder()
    five_qubit_decoder = prepare_five_qubit_decoder()

    results: list[tuple[QuantumCircuit, bool]] = []
    fidelities: list[float] = []

    def run_single_test(vec: np.ndarray, name: str):
        input_state_circ = QuantumCircuit(1, name=name)
        input_state_circ.initialize(vec, 0)

        full_circ = QuantumCircuit(7, name=f"validate_{name}")
        full_circ.initialize(vec, 6)
        full_circ.compose(steane_encoder, inplace=True)
        full_circ.compose(trained_circ, inplace=True)

        if five_qubit_decoder.num_qubits == 5:
            full_circ.compose(five_qubit_decoder, qubits=[0, 1, 2, 3, 4], inplace=True)
        elif five_qubit_decoder.num_qubits == 7:
            full_circ.compose(five_qubit_decoder, inplace=True)
        else:
            raise ValueError("Decoder must act on 5 or 7 qubits.")

        final_state = Statevector.from_instruction(full_circ)
        reduced_q4 = partial_trace(final_state, [0, 1, 2, 3, 5, 6])

        target_dm = DensityMatrix(Statevector(vec))
        fidelity = state_fidelity(reduced_q4, target_dm)

        fidelities.append(fidelity)
        results.append((input_state_circ, bool(np.isclose(fidelity, 1.0, atol=atol)), fidelity))

    # --- Deterministic tests ---
    run_single_test(np.array([1, 0], dtype=complex), "0")
    run_single_test(np.array([0, 1], dtype=complex), "1")
    run_single_test(np.array([1/np.sqrt(2), 1/np.sqrt(2)], dtype=complex), "+")

    if use_random_states:
        # --- Random tests ---
        for idx in range(num_sampled_states):
            vec = np.random.randn(2) + 1j * np.random.randn(2)
            vec = vec / np.linalg.norm(vec)
            run_single_test(vec, f"psi_{idx}")

    num_total = 3 + num_sampled_states if use_random_states else 3
    num_passed = sum(result[1] for result in results)

    print(
        f"Passed {num_passed}/{num_total} validation tests. "
        f"Min fidelity: {min(fidelities):.12f}, "
        f"Mean fidelity: {np.mean(fidelities):.12f}"
    )

    return results


def code_switch_cost(params: list | np.ndarray | dict, ansatz: QuantumCircuit):
    average_fidelity = 0.0
    if isinstance(params, (list, np.ndarray)):
        active_params = {param.name: params[i] for i, param in enumerate(ansatz.parameters)}
    elif isinstance(params, dict):
        active_params = {p: params[p.name] for p in ansatz.parameters if p.name in params}
    
    for init_circs, final_rotation in zip(initial_logical_states, final_rotations):
        full_circ = QuantumCircuit(7, 1) # Default classical register name is 'c'
        full_circ.compose(
            init_circs.compose(ansatz).compose(five_qubit_decoder.compose(final_rotation)), inplace=True
            )
        full_circ.measure(0, 0)
        
        pub = (full_circ, active_params, 200)
                                                         # classical register name
        counts = sim_sampler.run([pub]).result()[0].data.c.get_counts()
        
        fidelity = counts.get("0", 0) / 200
        average_fidelity += fidelity
             
    return -average_fidelity / len(initial_logical_states)
