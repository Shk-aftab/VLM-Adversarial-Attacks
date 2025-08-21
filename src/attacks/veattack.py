import torch

def linf_clamp(x, x0, eps):
    return torch.clamp(x, x0 - eps, x0 + eps)

def veattack(
    pixel_values: torch.Tensor,
    loss_fn,
    epsilon: float,
    alpha: float,
    steps: int,
    mode: str = "max",  # 'max' to maximize loss, 'min' to minimize
    momentum: float = 0.9
):
    """
    VEAttack (Vision Encoder Attack) as iterative PGD with cosine similarity loss (or NLL) on features.
    """
    x0 = pixel_values.clone().detach()
    perturbation = torch.zeros_like(pixel_values).uniform_(-epsilon, epsilon).detach().requires_grad_(True)
    velocity = torch.zeros_like(pixel_values)
    for _ in range(steps):
        perturbation.requires_grad_(True)
        loss = loss_fn(x0 + perturbation)
        if mode == "max":
            grad = torch.autograd.grad(loss, perturbation, retain_graph=False)[0]
        elif mode == "min":
            grad = -torch.autograd.grad(loss, perturbation, retain_graph=False)[0]
        else:
            raise ValueError(f"Unknown mode: {mode}")
        velocity = momentum * velocity + grad.sign()
        with torch.no_grad():
            perturbation = perturbation + alpha * velocity.sign()
            perturbation = linf_clamp(perturbation, torch.zeros_like(x0), epsilon)
            perturbation = linf_clamp(x0 + perturbation, x0, epsilon) - x0
            perturbation = (x0 + perturbation).clamp(-10, 10) - x0
        perturbation = perturbation.detach().requires_grad_(True)
    # Return the adversarial tensor
    return (x0 + perturbation).detach()