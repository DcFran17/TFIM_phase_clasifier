import sys
import time
import copy
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from arc.qsnn.entrada import train_loader, X_fit, X_val, y_val, RANDOM_SEED
from arc.qsnn.network import QSNN, NUM_HIDDEN, RATE_LOGIT_SCALE, output_logits, set_seed

LEARNING_RATE = 3e-4
WEIGHT_DECAY = 5e-5
NUM_EPOCHS = 300
PATIENCE = 45

RESULTS_DIR = ROOT_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
set_seed(RANDOM_SEED)

net = QSNN(num_inputs=X_fit.shape[1], num_hidden=NUM_HIDDEN, num_outputs=2).to(device)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(net.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", factor=0.5, patience=8, min_lr=1e-5
)

history = {"train_ce": [], "val_ce": [], "learning_rate": []}
best_val_loss = float("inf")
best_state = None
best_epoch = 0
steps_without_improvement = 0
start_time = time.time()

for epoch in range(NUM_EPOCHS):
    net.train()
    batch_losses = []
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        _, output_spikes = net(inputs)
        logits = output_logits(output_spikes)
        loss = loss_fn(logits, targets)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
        optimizer.step()
        batch_losses.append(float(loss.detach().cpu()))

    net.eval()
    with torch.no_grad():
        _, val_spikes = net(X_val.to(device))
        val_logits = output_logits(val_spikes)
        val_loss = float(loss_fn(val_logits, y_val.to(device)).cpu())

    scheduler.step(val_loss)
    train_ce = float(np.mean(batch_losses))
    history["train_ce"].append(train_ce)
    history["val_ce"].append(val_loss)
    history["learning_rate"].append(optimizer.param_groups[0]["lr"])

    if np.isfinite(val_loss) and val_loss < best_val_loss - 1e-4:
        best_val_loss = val_loss
        best_state = copy.deepcopy(net.state_dict())
        best_epoch = epoch + 1
        steps_without_improvement = 0
    else:
        steps_without_improvement += 1

    if epoch == 0 or (epoch + 1) % 10 == 0:
        print(f"Época {epoch+1:03d}/{NUM_EPOCHS} | CE ajuste={train_ce:.4f} | CE val={val_loss:.4f} | escala={RATE_LOGIT_SCALE:g}")
    if steps_without_improvement >= PATIENCE:
        print(f"Parada temprana en época {epoch+1}; mejor época={best_epoch}.")
        break

if best_state is None:
    raise RuntimeError("No se obtuvo un estado válido; revisa la dinámica de spikes y los gradientes.")

net.load_state_dict(best_state)
train_time = time.time() - start_time
print(f"Tiempo: {train_time:.1f} s | mejor época: {best_epoch} | CE val: {best_val_loss:.5f}")

torch.save(net.state_dict(), RESULTS_DIR / "qsnn_dual_alpha_state_dict.pt")

fig, ax = plt.subplots(figsize=(7.2, 4.2))
ax.plot(np.arange(1, len(history["train_ce"]) + 1), history["train_ce"], label="CrossEntropy ajuste")
ax.plot(np.arange(1, len(history["val_ce"]) + 1), history["val_ce"], label="CrossEntropy validación")
ax.set(xlabel="Época", ylabel="Entropía cruzada")
ax.grid(alpha=0.22)
ax.legend(frameon=False)
fig.tight_layout()
plt.show()