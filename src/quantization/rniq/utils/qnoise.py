import torch
from torch import Tensor
from torch.autograd import Function

class QNoise(Function):

    @staticmethod
    def forward(input, scale):
        # output = scale * (torch.round(input) - input)
        output = (torch.round(input) - input)
        return output

    @staticmethod
    def setup_context(ctx, inputs, output):
        input, scale = inputs
        ctx.save_for_backward(input, scale)

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        # This is a pattern that is very convenient - at the top of backward
        # unpack saved_tensors and initialize all gradients w.r.t. inputs to
        # None. Thanks to the fact that additional trailing Nones are
        # ignored, the return statement is simple even when the function has
        # optional inputs.
        input, scale = ctx.saved_tensors
        grad_input = grad_scale = None

        if ctx.needs_input_grad[0]:
            # grad_input = grad_output * 0
            grad_input = torch.zeros_like(grad_output)
        if ctx.needs_input_grad[1]:
            #grad_scale = grad_output * (torch.round(input) - input)
            #grad_scale = grad_output * (torch.randint(2, size=input.shape, dtype=input.dtype, device=input.device).sub(0.5))
            grad_scale = grad_output * torch.normal(0, 0.2888, size=input.shape, dtype=input.dtype, device=input.device) # 0.2888 is std for [-1, 1] normal distribution
            #grad_scale = grad_output * (torch.rand_like(input).sub_(0.5))

        return grad_input, grad_scale