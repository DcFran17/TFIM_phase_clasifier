import torch
import torch.nn as nn

THRESHOLD = 0.75
T1 = 60.0
class SpikeSurrogate(torch.autograd.Function):
    """Spike binario en forward y derivada surrogate normalizada en backward."""
    @staticmethod
    def forward(ctx, alpha, threshold):
        ctx.save_for_backward(alpha)
        return (alpha > threshold).to(alpha.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        (alpha,) = ctx.saved_tensors
        surrogate_grad = 1.0 / (torch.pi * (1.0 + (torch.pi * alpha).pow(2)))
        return grad_output * surrogate_grad, None


class Alpha(nn.Module):
    def __init__(self, threshold=THRESHOLD, T1=T1, size=1):
        super().__init__()
        self.threshold = threshold
        self.T1 = T1
        self.size = size
        self.mem = None

    def reset_mem(self, batch_size, device):
        self.mem = torch.zeros(batch_size, self.size, dtype=torch.float32, device=device)

    def forward(self, z, tau, x):
        eps = 1e-7
        if self.mem is None or self.mem.shape != z.shape or self.mem.device != z.device:
            self.reset_mem(z.shape[0], z.device)

        mem = self.mem
        mem_safe = mem.clamp(min=0.0, max=1.0 - eps)
        mem_trig = mem_safe.clamp(min=eps)
        phi = torch.where(
            mem < self.threshold,
            2.0 * torch.asin(torch.sqrt(mem_trig)),
            torch.zeros_like(mem)
        )

        # Excitación monotónica y acotada: 0 <= theta_eff < pi - phi.
        theta_eff = (torch.pi - phi) * (1.0 - torch.exp(-torch.abs(z)))

        # Relajación alternativa: con x=0, sinapsis sola, mem(t+1) ~= mem(t) exp(-tau/T1).
        decay = torch.exp(-tau / self.T1)
        decayed_mem = (mem_trig * decay).clamp(min=eps, max=1.0 - eps)
        gamma_alt = 2.0 * torch.asin(torch.sqrt(decayed_mem)) - phi

        theta_term = torch.sin((theta_eff + phi) * x / 2.0).pow(2)
        gamma_term = torch.sin((gamma_alt + phi) * (1.0 - x) / 2.0).pow(2)
        alpha_candidate = torch.where(
            mem < self.threshold,
            gamma_term + theta_term,
            torch.zeros_like(mem)
        )

        spike = SpikeSurrogate.apply(alpha_candidate, self.threshold)
        mem_next = torch.where(alpha_candidate > 1.0, torch.zeros_like(alpha_candidate), alpha_candidate)
        self.mem = mem_next
        return spike, mem_next