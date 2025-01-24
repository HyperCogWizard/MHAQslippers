import torch
from torch import nn, inf

from src.aux.types import QScheme
from src.quantization.rniq.rniq import Quantizer
from src.quantization.rniq.utils.enums import QMode


class NoisyAct(nn.Module):
    def __init__(self, init_s=-10, init_q=10, channels=1, signed=True, noise_ratio=1, disable=False) -> None:
        super().__init__()
        self.disable = disable
        self.signed = signed
        self._act_b = torch.tensor([0]).float()
        self._log_act_s = torch.tensor([init_s]).float()
        self._log_act_q = torch.tensor([init_q]).float()
        self._noise_ratio = torch.tensor(noise_ratio)
        # self.log_act_q = torch.nn.Parameter(self._log_act_q, requires_grad=True)
        self.log_act_q = torch.nn.Parameter(torch.empty(channels).fill_(init_q), requires_grad=True)
        self.act_b = torch.nn.Parameter(torch.empty(channels).fill_(0), requires_grad=True)

        # if signed:
        #     self.act_b = torch.nn.Parameter(self._act_b, requires_grad=True)
        # else:
        #     self.act_b = torch.nn.Parameter(self._act_b, requires_grad=False)
        #     self.act_b = torch.nn.Parameter(self._act_b, requires_grad=False)

        self.log_act_s = torch.nn.Parameter(torch.empty(channels).fill_(init_s), requires_grad=True)
        # self.log_act_s = torch.nn.Parameter(self._log_act_s, requires_grad=True)
        self.Q = Quantizer(self, torch.exp2(self.log_act_s), 0, -inf, inf)
        self.bw = torch.tensor(0.0)

    def forward(self, x):
        if self.disable:
            return x
        s = torch.exp2(self.log_act_s)
        q = torch.exp2(self.log_act_q)
        
        self.Q.zero_point = self.act_b.view(1, -1, 1, 1)
        self.Q.min_val = self.act_b.view(1, -1, 1, 1)
        self.Q.max_val = (self.act_b + q - s).view(1, -1, 1, 1)
        self.Q.scale = s.view(1, -1, 1, 1)
        # self.Q.rnoise_ratio = self._noise_ratio
        self.Q.rnoise_ratio = torch.tensor(0)

        q = self.Q.quantize(x)
        if not self.training:
            self.bw = torch.log2(torch.Tensor([q.unique().numel()]))
        return self.Q.dequantize(q)
