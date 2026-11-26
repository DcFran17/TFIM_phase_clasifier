from qiskit.quantum_info import SparsePauliOp

def build_ising_hamiltonian(n_qubits, coupling, h):
    terms = []
    for i in range(n_qubits):
        op = ["I"] * n_qubits
        op[i] = "Z"
        op[(i + 1) % n_qubits] = "Z"
        terms.append(("".join(op), -coupling))
    for i in range(n_qubits):
        op = ["I"] * n_qubits
        op[i] = "X"
        terms.append(("".join(op), -h))
    return SparsePauliOp.from_list(terms).simplify()

#Data for testing the function
if __name__ == "__main__":
    print(build_ising_hamiltonian(5, 1.0, 0.5))

