import torch
import torch.nn as nn
import torch.nn.functional as F

class SoftPlusMaxPool2d(nn.Module):
    """
    Approximation of default MaxPool2d using softplus
    kernel_size, stride, padding - similar to nn.MaxPool2d
    beta - "coarse" of approximation;  beta->inf results with true max.
    """
    def __init__(self, kernel_size, stride=None, padding=0, beta: float = 1.0):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride if (stride is not None) else kernel_size
        self.padding = padding
        self.beta = beta

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N, C, H, W)
        # 1) unfold → patches with k×k dimensions
        patches = F.unfold(
            x,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding
        )
        # patches: (N, C* k*k, L), где L = number of sliding windows
        N, Ck2, L = patches.shape
        k2 = Ck2 // x.size(1)
        # 2) translate to (N, C, k*k, L) view
        patches = patches.view(N, x.size(1), k2, L)
        # 3) calculate (1/β)·logsumexp(β·patches) w.r.t k*k dimensions
        # therefore it's 2D softmax
        pooled = torch.logsumexp(self.beta * patches, dim=2) / self.beta
        # pooled: (N, C, L)
        # 4) decomposition back into tensor (N, C, H_out, W_out)
        H_out = (x.size(2) + 2*self.padding - self.kernel_size) // self.stride + 1
        W_out = (x.size(3) + 2*self.padding - self.kernel_size) // self.stride + 1
        return pooled.view(N, x.size(1), H_out, W_out)
