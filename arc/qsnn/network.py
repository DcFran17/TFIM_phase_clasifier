import sys
import random
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import torch
import torch.nn as nn

from arc.qsnn.alpha_neuron import Alpha

NUM_HIDDEN = 64
NUM_STEPS = 200
RATE_LOGIT_SCALE = 100.0
RANDOM_SEED = 42


def set_seed(seed=RANDOM_SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class QSNN(nn.Module):
    """Dos capas Alpha: 64 neuronas ocultas y dos neuronas de salida."""
    def __init__(self, num_inputs, num_hidden=NUM_HIDDEN, num_outputs=2):
        super().__init__()
        self.num_steps = NUM_STEPS
        self.fcq1 = nn.Linear(num_inputs, num_hidden)
        self.fct1 = nn.Linear(num_inputs, num_hidden)
        self.lif1 = Alpha(size=num_hidden)
        self.fcq2 = nn.Linear(num_hidden, num_outputs)
        self.fct2 = nn.Linear(num_hidden, num_outputs)
        self.lif2 = Alpha(size=num_outputs)

    def forward(self, inputs):
        inputs = inputs.reshape(inputs.size(0), -1)
        self.lif1.reset_mem(inputs.size(0), inputs.device)
        self.lif2.reset_mem(inputs.size(0), inputs.device)
        hidden_record, output_record = [], []

        for _ in range(self.num_steps):
            z1 = self.fcq1(inputs)
            tau1 = self.fct1(inputs).abs() * 200.0
            x1 = (z1 > 0).to(z1.dtype)
            spk1, _ = self.lif1(z1, tau1, x1)

            z2 = self.fcq2(spk1)
            tau2 = self.fct2(spk1).abs() * 200.0
            x2 = (z2 > 0).to(z2.dtype)
            spk2, _ = self.lif2(z2, tau2, x2)
            hidden_record.append(spk1)
            output_record.append(spk2)

        return torch.stack(hidden_record), torch.stack(output_record)


def output_rates(output_spikes):
    # output_spikes: [pasos, lote, 2] -> [lote, 2]
    return output_spikes.mean(dim=0)


def output_logits(output_spikes):
    # La escala calibra las tasas como logits para softmax/CrossEntropy.
    return RATE_LOGIT_SCALE * output_rates(output_spikes)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(RANDOM_SEED)
    net = QSNN(num_inputs=2**10, num_hidden=NUM_HIDDEN, num_outputs=2).to(device)
    num_parameters = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print(net)
    print(f"Neuronas de salida Alpha: 2 | parámetros entrenables: {num_parameters:,}")