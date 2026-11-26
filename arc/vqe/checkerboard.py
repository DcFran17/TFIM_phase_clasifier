from qiskit.circuit import QuantumCircuit, ParameterVector
from qiskit import QuantumCircuit
import matplotlib.pyplot as plt

def checkerboard_matchings(n_qubits):
    edges = [(i, (i + 1) % n_qubits) for i in range(n_qubits)]
    remaining = edges.copy()
    matchings = []
    while remaining:
        layer, used, rest = [], set(), []
        for q1, q2 in remaining:
            if q1 not in used and q2 not in used:
                layer.append((q1, q2))
                used.update((q1, q2))
            else:
                rest.append((q1, q2))
        matchings.append(layer)
        remaining = rest
    return matchings


def build_checkerboard_ansatz(n_qubits, layers=4):
    qc = QuantumCircuit(n_qubits)
    matchings = checkerboard_matchings(n_qubits)
    pairings = [matchings[layer % len(matchings)] for layer in range(layers)]
    parameter_count = 5 * sum(len(layer) for layer in pairings)
    theta = ParameterVector("theta", parameter_count)
    k = 0
    for layer in pairings:
        for q1, q2 in layer:
            qc.rx(theta[k], q1)
            qc.rx(theta[k + 1], q2)
            qc.rzz(theta[k + 2], q1, q2)

            qc.rz(theta[k + 3], q1)
            qc.rz(theta[k + 4], q2)
            k += 5
    return qc, theta


#For testing the function

n_qubits = 10
layer = 2

qc, theta = build_checkerboard_ansatz(n_qubits, layers=layer)

qc.draw("mpl", fold=-1, style={"backgroundcolor": "#EEEEEE"})
#Data for testing the function
if __name__ == "__main__":
    plt.show()

 

