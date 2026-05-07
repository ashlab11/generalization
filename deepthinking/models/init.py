"""Initialization utilities for models."""
from __future__ import annotations

from typing import Optional
import math

import torch
from torch import nn

_RESIDUAL_OUTPUT_SUFFIXES = (
    ".out_proj",  # MHA output projection
    ".conv2",     # ConvAttn second conv
    ".mlp.2",     # Final MLP Linear in AttentionBlock
)


def _init_weighted_module(module: nn.Module, method: str, name: str) -> None:
    if method == "xavier":
        nn.init.xavier_uniform_(module.weight)
    elif method == "xavier_small":
        nn.init.xavier_uniform_(module.weight, gain=0.5)
    elif method == 'orthogonal':
        nn.init.orthogonal(module.weight)
    elif method == "kaiming":
        nn.init.kaiming_uniform_(module.weight, a=math.sqrt(5))
    elif method == "residual_zero":
        if name.endswith(_RESIDUAL_OUTPUT_SUFFIXES):
            nn.init.zeros_(module.weight)
        else:
            nn.init.xavier_uniform_(module.weight)
    else:
        raise ValueError(f"Unknown init_method: {method}")

    if module.bias is not None:
        nn.init.zeros_(module.bias)


def _init_gru_cell(module: nn.GRUCell) -> None:
    with torch.no_grad():
        nn.init.xavier_uniform_(module.weight_ih)
        nn.init.orthogonal_(module.weight_hh)
        if module.bias_ih is not None:
            nn.init.zeros_(module.bias_ih)
            hidden = module.hidden_size
            module.bias_ih[hidden:2 * hidden].fill_(1.0)
        if module.bias_hh is not None:
            nn.init.zeros_(module.bias_hh)
            hidden = module.hidden_size
            module.bias_hh[hidden:2 * hidden].fill_(1.0)

def _init_lstm_cell(module: nn.LSTMCell) -> None:
    with torch.no_grad():
        nn.init.xavier_uniform_(module.weight_ih)
        nn.init.orthogonal_(module.weight_hh)
        if module.bias_ih is not None:
            nn.init.zeros_(module.bias_ih)
            hidden = module.hidden_size
            # Set forget gate bias to 1.0: bias structure is [input, forget, cell, output]
            module.bias_ih[hidden:2 * hidden].fill_(1.0)
        if module.bias_hh is not None:
            nn.init.zeros_(module.bias_hh)
            hidden = module.hidden_size
            # Set forget gate bias to 1.0: bias structure is [input, forget, cell, output]
            module.bias_hh[hidden:2 * hidden].fill_(1.0)

def apply_initialization(model: nn.Module, init_method: Optional[str]) -> None:
    """Initialize non-GRU layers. GRU init stays at default (module-created) values."""
    for name, module in model.named_modules():
        if isinstance(module, nn.GRUCell):
            _init_gru_cell(module)
        if isinstance(module, nn.LSTMCell):
            _init_lstm_cell(module)

    if init_method is None:
        return
    method = init_method.lower()
    if method == "default":
        return

    for name, module in model.named_modules():
        if isinstance(module, nn.GRUCell):
            continue
        if isinstance(module, (nn.Linear, nn.Conv1d)):
            _init_weighted_module(module, method, name)
