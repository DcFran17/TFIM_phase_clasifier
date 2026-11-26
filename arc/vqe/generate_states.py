import sys
import time
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.sparse.linalg import eigsh
from qiskit.quantum_info import Statevector

from arc.vqe.hamiltonian import build_ising_hamiltonian
from arc.vqe.checkerboard import build_checkerboard_ansatz

N_QUBITS = 10
N_LAYERS = 4
J = 1.0
H_MIN = 0.0
H_MAX = 2.0
N_H = 100
RANDOM_SEED = 42
H_C = J

MAXITER = 300
TOL = 1e-6
FINITE_DIFF_EPS = 1e-6

DATA_DIR = ROOT_DIR / "data"
RESULTS_DIR = ROOT_DIR / "results"
DATA_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def state_and_energy(ansatz, values, hamiltonian):
    bound = ansatz.assign_parameters(values)
    state = Statevector.from_instruction(bound)
    energy = float(np.real(state.expectation_value(hamiltonian)))
    return state, energy


def objective(values, ansatz, hamiltonian):
    return state_and_energy(ansatz, values, hamiltonian)[1]


def exact_ground_energy(hamiltonian):
    try:
        matrix = hamiltonian.to_matrix(sparse=True)
        return float(np.real(eigsh(matrix, k=1, which="SA", return_eigenvectors=False, tol=1e-9)[0]))
    except (TypeError, ValueError, AttributeError):
        return float(np.real(np.linalg.eigvalsh(hamiltonian.to_matrix())[0]))


def run_vqe_sweep(h_list, direction, ansatz, theta_params, rng):
    sweep = {"states": [], "energies": [], "params": [], "success": [], "iterations": [], "messages": []}
    previous = None
    start = time.time()
    print(f"\nBarrido {direction.upper()}: {len(h_list)} puntos")

    for idx, h in enumerate(h_list):
        hamiltonian = build_ising_hamiltonian(N_QUBITS, J, float(h))
        x0 = rng.uniform(-0.1, 0.1, len(theta_params)) if previous is None else previous.copy()
        result = minimize(
            objective, x0, args=(ansatz, hamiltonian), method="L-BFGS-B",
            options={
                "maxiter": MAXITER, "ftol": TOL, "gtol": TOL,
                "eps": FINITE_DIFF_EPS, "maxls": 30
            }
        )
        previous = result.x.copy()
        state, energy = state_and_energy(ansatz, result.x, hamiltonian)
        state_data = np.asarray(state.data, dtype=np.complex128)
        state_data /= np.linalg.norm(state_data)

        sweep["states"].append(state_data)
        sweep["energies"].append(energy)
        sweep["params"].append(result.x.copy())
        sweep["success"].append(bool(result.success))
        sweep["iterations"].append(int(result.nit))
        sweep["messages"].append(str(result.message))

        if idx % 10 == 0 or idx == len(h_list) - 1:
            print(f"  {idx+1:3d}/{len(h_list)} | h/J={h/J:.4f} | E={energy:.8f} | iter={result.nit} | success={result.success}")

    print(f"Tiempo {direction}: {time.time()-start:.1f} s")
    return {key: np.asarray(value) for key, value in sweep.items()}


if __name__ == "__main__":
    vqe_file = DATA_DIR / f"vqe_checkerboard_states_{N_QUBITS}qubits.npz"

    # Si no existe en data/, lo calcula y lo guarda automáticamente en data/
    if not vqe_file.exists():
        print(f"No existe {vqe_file}. Generando estados VQE y guardando en data/...")
        rng = np.random.default_rng(RANDOM_SEED)
        h_values = np.linspace(H_MIN, H_MAX, N_H)
        ansatz, theta_params = build_checkerboard_ansatz(N_QUBITS, layers=N_LAYERS)

        forward = run_vqe_sweep(h_values, "ida", ansatz, theta_params, rng)
        backward_rev = run_vqe_sweep(h_values[::-1], "vuelta", ansatz, theta_params, rng)
        backward = {key: val[::-1] for key, val in backward_rev.items()}

        use_forward = forward["energies"] <= backward["energies"]
        states = np.where(use_forward[:, None], forward["states"], backward["states"])
        energies = np.where(use_forward, forward["energies"], backward["energies"])
        exact_energies = np.array([exact_ground_energy(build_ising_hamiltonian(N_QUBITS, J, float(h))) for h in h_values])
        energy_errors = np.abs(energies - exact_energies)
        probabilities = np.abs(states) ** 2

        np.savez(
            vqe_file,
            h_values=h_values,
            states=states,
            energies=energies,
            exact_energies=exact_energies,
            energy_errors=energy_errors,
            probabilities=probabilities,
        )
        print(f"Archivo guardado en: {vqe_file}")

    with np.load(vqe_file) as saved:
        h_values = np.asarray(saved["h_values"], dtype=float)
        states = np.asarray(saved["states"], dtype=np.complex128)
        energies = np.asarray(saved["energies"], dtype=float)
        exact_energies = np.asarray(saved["exact_energies"], dtype=float)
        energy_errors = np.asarray(saved["energy_errors"], dtype=float)
        probabilities = np.asarray(saved["probabilities"], dtype=float)

    if states.shape != (N_H, 2**N_QUBITS):
        raise ValueError(f"Dimensiones incompatibles: {states.shape}; se esperaba {(N_H, 2**N_QUBITS)}")
    print(f"Estados VQE cargados: {states.shape} desde {vqe_file}")
    print(f"Error energético medio: {energy_errors.mean():.6g} | máximo: {energy_errors.max():.6g}")
    print(f"Suma de probabilidades por estado: [{probabilities.sum(axis=1).min():.10f}, {probabilities.sum(axis=1).max():.10f}]")

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.plot(h_values / J, energy_errors / N_QUBITS, color="black", linewidth=1.4)
    ax.scatter(h_values / J, energy_errors / N_QUBITS, color="black", s=10)
    ax.axvline(H_C / J, color="0.45", linestyle="--", linewidth=1)
    ax.set(xlabel=r"$h/J$", ylabel=r"$|E_{VQE}-E_0|/N$")
    ax.grid(alpha=0.22)
    fig.tight_layout()
    plt.show()
