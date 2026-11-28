# Supervised Quantum Phase Classification of the Transverse-Field Ising Model via VQE and a Quantum Spiking Neural Network (QSNN)

This repository implements a hybrid quantum-classical pipeline to identify the quantum phase transition of the 1D periodic **Transverse-Field Ising Model (TFIM)**. Ground-state wavefunctions are approximated across varying transverse field strengths using a **Variational Quantum Eigensolver (VQE)** with a hardware-efficient checkerboard ansatz, and subsequently classified into ordered ($h/J < 1$) and disordered ($h/J \ge 1$) phases by a two-layer **Quantum Spiking Neural Network (QSNN)** built with custom Alpha spiking neurons and surrogate-gradient learning.

---

## Physical Model & Methodology

1. **Hamiltonian & Ground-State Generation (`arc/vqe/`)**
   * **Model:** 10-qubit 1D periodic Transverse-Field Ising Model ($J = 1.0$, critical point $h_c/J = 1.0$):
     $$H = -J \sum_{i=0}^{N-1} Z_i Z_{i+1} - h \sum_{i=0}^{N-1} X_i$$
   * **Ansatz & Optimization:** A 4-layer checkerboard variational ansatz optimized via L-BFGS-B across $N_h = 100$ transverse field points $h \in [0.0, 2.0]$, validated against exact diagonalization (`scipy.sparse.linalg.eigsh`).

2. **Quantum Spiking Neural Network (`arc/qsnn/`)**
   * **Input Encoding:** Measurement probability distributions $p_k = \vert{}\langle k \vert{} \psi(h) \rangle\vert{}^2 \in \mathbb{R}^{1024}$ computed from the VQE statevectors.
   * **Architecture:** Two spiking layers (`1024 -> 64 -> 2`) using stateful `Alpha` neurons simulated over $T = 200$ time steps, trained via AdamW and cross-entropy loss on scaled output firing rates.

---

## Repository Structure

```text
TFIM_phase_clasifier/
├── arc/
│   ├── vqe/
│   │   ├── hamiltonian.py        # Builds the periodic TFIM SparsePauliOp Hamiltonian
│   │   ├── checkerboard.py       # Constructs the 4-layer checkerboard variational ansatz
│   │   └── generate_states.py    # Runs VQE optimization and exports ground states (.npz)
│   ├── qsnn/
│   │   ├── alpha_neuron.py       # Alpha spiking neuron model and normalized surrogate gradient
│   │   ├── network.py            # Two-layer QSNN architecture and firing-rate readout
│   │   ├── entrada.py            # Data loading, probability extraction, and stratified splits
│   │   ├── train.py              # Training loop with ReduceLROnPlateau and early stopping
│   │   └── test.py               # Held-out test evaluation, metrics, plots, and CSV/JSON export
│   └── data_pipeline.py          # End-to-end data processing utilities
├── data/
│   └── vqe_checkerboard_states_10qubits.npz   # Precomputed VQE states, energies, and exact benchmarks
├── Figures/
│   ├── Figure_0.png              # VQE circuit / energy error diagnostics
│   ├── Figure_1.png              # Training and validation cross-entropy curves
│   ├── Figure_2.png              # QSNN phase-1 predicted probability vs. h/J
│   ├── Figure_3.png              # Development vs. held-out test phase boundary scatter
│   └── Figure_4.png              # Additional phase transition analysis plots
├── notebook/                     # Exploratory Jupyter notebooks for VQE & dual-output QSNN
├── results/
│   ├── qsnn_dual_alpha_state_dict.pt   # Trained PyTorch model weights
│   ├── qsnn_metrics_dual_alpha.json    # Summary test metrics and hyperparameters
│   └── qsnn