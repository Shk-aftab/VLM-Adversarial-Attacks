import torch

def linf_clamp(x, x0, eps):
    return torch.clamp(x, x0 - eps, x0 + eps)

def fgsm_attack(pixel_values: torch.Tensor, loss_fn, epsilon: float):
    x = pixel_values.clone().detach().requires_grad_(True)
    loss = loss_fn(x)
    loss.backward()
    with torch.no_grad():
        adv = x + epsilon * x.grad.sign()
    return adv.clamp(-10, 10)
