import torch
import torch.nn as nn
import torch.nn.functional as F
"""
Exponential Moving Average Linear Layer. Shared-weight transformers are useful in that they can
have earlier activations affect later activations -- sort of a long-distance skip-residual. 
NOTE: I didn't use this in main project, just a cool idea
"""
class EMA_Linear(nn.Module):
    def __init__(self, in_dim, out_dim, detach = False):
        super().__init__()
        self.base_linear = nn.Linear(in_dim, out_dim)
        self.ema_logits = nn.Parameter(torch.ones(out_dim) * 5)
        self.register_buffer('held_act', None)
        #Use these types of functions to reduce the if/else in forward
        self.detach_func = lambda x: x.detach().clone() if detach else x.clone()
    
    def reset_state(self):
        self.held_act = None

    def forward(self, x):
        #EMA linear, allowing for size changs (CCOT)
        ema_sigmoid = F.sigmoid(self.ema_logits)
        x = self.base_linear(x) #out_dim
        if self.held_act is None:
            self.held_act = self.detach_func(x)
            return x
        if self.held_act.device != x.device or self.held_act.dtype != x.dtype:
            self.held_act = self.held_act.to(device=x.device, dtype=x.dtype)
        if any(b > t for b, t in zip(self.held_act.shape, x.shape)):
            raise ValueError(f"held_act larger than target: {tuple(self.held_act.shape)} vs {tuple(x.shape)}")
        if self.held_act.shape == x.shape:
            x = x * ema_sigmoid + self.held_act * (1 - ema_sigmoid)
        else:
            slices = tuple(slice(0, b) for b in self.held_act.shape)
            y = x.clone()
            y[slices] = x[slices] * ema_sigmoid + self.held_act * (1 - ema_sigmoid)
            x = y
        self.held_act = self.detach_func(x)
        return x
