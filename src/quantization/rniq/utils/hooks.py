import torch.nn as nn

class ActivationHook():
    def __init__(self, module: nn.Module):
        self.feature_map = None
        self.hook = module.register_forward_hook(self.forward_hook)
    
    def forward_hook(self, module, input, output):
        self.feature_map = output
    
    def remove(self):
        self.hook.remove()