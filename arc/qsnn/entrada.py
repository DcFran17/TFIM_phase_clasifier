from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data"

N_QUBITS = 10
J = 1.0
H_C = J
RANDOM_SEED = 42
BATCH_SIZE = 8


vqe_file = DATA_DIR / f"vqe_checkerboard_states_{N_QUBITS}qubits.npz"
if not vqe_file.exists():
    raise FileNotFoundError(f"No existe {vqe_file}. Ejecuta primero las celdas del VQE.")

with np.load(vqe_file) as saved:
    h_values = np.asarray(saved["h_values"], dtype=float)
    states = np.asarray(saved["states"], dtype=np.complex128)
    energies = np.asarray(saved["energies"], dtype=float)
    exact_energies = np.asarray(saved["exact_energies"], dtype=float)
    energy_errors = np.asarray(saved["energy_errors"], dtype=float)

if states.ndim != 2 or states.shape != (len(h_values), 2**N_QUBITS):
    raise ValueError(f"Dimensiones de estados incompatibles: {states.shape}; se esperaba ({len(h_values)}, {2**N_QUBITS}).")

probabilities = np.abs(states)**2
probability_sums = probabilities.sum(axis=1)
if not np.allclose(probability_sums, 1.0, atol=1e-6):
    raise ValueError("Hay estados cuyas probabilidades no suman uno.")

labels = (h_values >= H_C).astype(np.int64)
X = torch.tensor(probabilities, dtype=torch.float32)
y = torch.tensor(labels, dtype=torch.long)
indices = np.arange(len(h_values))

dev_idx, test_idx = train_test_split(
    indices, test_size=0.20, random_state=RANDOM_SEED,
    shuffle=True, stratify=labels
)
fit_idx, val_idx = train_test_split(
    dev_idx, test_size=0.15, random_state=RANDOM_SEED + 1,
    shuffle=True, stratify=labels[dev_idx]
)

X_fit, y_fit = X[fit_idx], y[fit_idx]
X_val, y_val = X[val_idx], y[val_idx]
X_test, y_test = X[test_idx], y[test_idx]

print(f"Estados: {len(indices)} | ajuste: {len(fit_idx)} | validación: {len(val_idx)} | test: {len(test_idx)}")
print(f"Suma de probabilidades: [{probability_sums.min():.10f}, {probability_sums.max():.10f}]")
print("Conteos por clase (ajuste/validación/test):")
for name, idx in [("ajuste", fit_idx), ("validación", val_idx), ("test", test_idx)]:
    counts = np.bincount(labels[idx], minlength=2)
    print(f"  {name:10s}: fase 0={counts[0]:2d}, fase 1={counts[1]:2d}")

train_loader = DataLoader(
    TensorDataset(X_fit, y_fit), batch_size=BATCH_SIZE, shuffle=True,
    generator=torch.Generator().manual_seed(RANDOM_SEED)
)