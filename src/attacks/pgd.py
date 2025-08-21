import torch
from .fgsm import linf_clamp

def pgd_attack(pixel_values: torch.Tensor, loss_fn, epsilon: float, alpha: float, steps: int):
    x0 = pixel_values.clone().detach()
    x = x0.clone().detach().requires_grad_(True)
    for _ in range(steps):
        x.grad = None
        loss = loss_fn(x)
        loss.backward()
        with torch.no_grad():
            x = x + alpha * x.grad.sign()
            x = linf_clamp(x, x0, epsilon).clamp(-10, 10)
        x.requires_grad_(True)
    return x.detach()
