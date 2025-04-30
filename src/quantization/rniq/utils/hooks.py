import torch.nn as nn
import torch


class ActivationHook:
    def __init__(self, module: nn.Module):
        self.feature_map = None
        self.hook = module.register_forward_hook(self.forward_hook)

    def forward_hook(self, module, input, output):
        self.feature_map = output

    def remove(self):
        self.hook.remove()


class NoiseGradHook:
    def __init__(self, module: nn.Module, noise_scale=1e-3, noise_prob=0.5):
        self.hook = module.register_full_backward_hook(self.backward_hook)
        self.noise_scale = noise_scale
        self.noise_prob = noise_prob

    def backward_hook(self, module, grad_input, grad_output):
        if isinstance(grad_input, tuple):
            B = torch.bernoulli(torch.full_like(grad_input[0], self.noise_prob))
            S = B.mul(2.0).sub(1.0)
            return (grad_input[0] + S * self.noise_scale,)

        else:
            B = torch.bernoulli(torch.full_like(grad_input, self.noise_prob))
            S = B.mul(2.0).sub(1.0)
            return grad_input + S * self.noise_scale

    def remove(self):
        self.hook.remove()
