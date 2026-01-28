""" blocks.py
    Neural network blocks.

    Collaboratively developed
    by Avi Schwarzschild, Eitan Borgnia,
    Arpit Bansal, and Zeyad Emam.

    BasicBlocks borrowed from ResNet architechtures
    Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>

    Developed for DeepThinking project
    October 2021
"""

import torch
from torch import nn, einsum
import torch.nn.functional as F
from itertools import permutations
from torchtune.modules import RotaryPositionalEmbeddings
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.nn.attention.flex_attention import flex_attention, create_block_mask
from math import factorial

WINDOW_SIZE = 31

def get_sliding_window(causal, sinks):
    def sliding_window(b, h, q_idx, kv_idx):
        sink = kv_idx <= sinks - 1
        window_mask = (q_idx - (kv_idx - sinks)).abs() <= WINDOW_SIZE
        if causal:
            causal_mask = q_idx >= (kv_idx - sinks) 
            return (window_mask & causal_mask) | sink
        return window_mask | sink
    return sliding_window

def add_monotone_hook(x_t, x_tm1, lam=1.0, margin=0.0, reduce_over_tokens=True):
    delta = (x_t - x_tm1).detach()

    def hook(g):
        with torch.no_grad():
            if reduce_over_tokens and g.ndim == 3:
                s = (g.float() * delta.float()).sum(dim=-1).mean(dim=-1, keepdim=True)
                active = (s > -margin).to(g.dtype).view(-1, 1, 1)
            elif reduce_over_tokens and g.ndim == 4:
                s = (g.float() * delta.float()).sum(dim=-1).mean(dim=[0,-1], keepdim=True)
                s = s.squeeze(0)
                active = (s > -margin).to(g.dtype).view(-1, 1, 1)
            else:
                s = (g.float() * delta.float()).sum(dim=-1, keepdim=True)
                active = (s > -margin).to(g.dtype)
        return g * (1.0 + lam * active)

    x_t.register_hook(hook)

class ConvAttn(nn.Module):
    def __init__(self, hidden_dim, concatenated = False):
        super().__init__()
        self.concatenated = concatenated
        if concatenated:
            self.proj_down = nn.Linear(hidden_dim * 2, hidden_dim)
        self.conv1 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3,
                               stride=1, padding=1, bias=False)
    def forward(self, x):
        #[B, L, (2)D]
        if self.concatenated:
            x = self.proj_down(x)
            x = F.relu(x)
        x = x.transpose(1, 2) #[B, D (C), L]
        x = self.conv2(F.relu(self.conv1(x)))
        return x.transpose(1, 2) #[B, L, D]
    
class MHA(nn.Module):
    def __init__(self, hidden_dim, num_heads, attn_type = 'full', 
                 qk_normalization = False, concatenated = False, max_seq_len=None, num_sinks=1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.attn_type = attn_type
        self.num_sinks = num_sinks
        assert hidden_dim % num_heads == 0, f"hidden_dim {hidden_dim} must be divisible by num_heads {num_heads}"
        
        if concatenated:
            self.qkv = nn.Linear(hidden_dim * 2, hidden_dim * 3)
        else:
            self.qkv = nn.Linear(hidden_dim, hidden_dim * 3)
            
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.max_seq_len = 512 if max_seq_len is None else int(max_seq_len)
        self.rope = RotaryPositionalEmbeddings(dim=self.head_dim, max_seq_len=self.max_seq_len)
        self.qk_norm = qk_normalization
        
        if qk_normalization:
            self.q_norm = nn.RMSNorm(self.head_dim)
            self.k_norm = nn.RMSNorm(self.head_dim)
        
        if self.num_sinks > 0:
            self.sink_k = nn.Parameter(torch.zeros(num_sinks, num_heads, self.head_dim))
            self.sink_v = nn.Parameter(torch.zeros(num_sinks, num_heads, self.head_dim))
        
        # Cache for block masks at different sequence lengths
        self._mask_cache = {}
        if 'local' in attn_type:
            self.flex_attention_compiled = torch.compile(flex_attention)
        self._compute_attn_stats = False
        
    def forward(self, x):
        # x: (B, L, D)
        B, L, _ = x.shape
        current_max = getattr(self.rope, "max_seq_len", None)
        if current_max is None or L > current_max:
            # Grow RoPE cache to accommodate longer sequences (e.g., large mazes).
            self.max_seq_len = L
            self.rope = RotaryPositionalEmbeddings(dim=self.head_dim, max_seq_len=self.max_seq_len).to(x.device)
        qkv = self.qkv(x)  # (B, L, 3*D)
        q, k, v = qkv.chunk(3, dim=-1)  # Each: (B, L, D)
        
        # Reshape to (B, L, num_heads, head_dim)
        q = q.reshape(B, L, self.num_heads, self.head_dim)
        k = k.reshape(B, L, self.num_heads, self.head_dim)
        v = v.reshape(B, L, self.num_heads, self.head_dim)
        
        q = self.rope(q)
        k = self.rope(k)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        
        if self.qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)
        
        if self.num_sinks > 0:
            sink_k = self.sink_k.unsqueeze(0).expand(B, -1, -1, -1).transpose(1, 2).to(k.dtype)
            sink_v = self.sink_v.unsqueeze(0).expand(B, -1, -1, -1).transpose(1, 2).to(v.dtype)
            k = torch.cat([sink_k, k], dim=2)
            v = torch.cat([sink_v, v], dim=2)
        
        is_causal = 'causal' in self.attn_type
        
        if 'local' in self.attn_type:
            if L not in self._mask_cache:
                self._mask_cache[L] = create_block_mask(get_sliding_window(is_causal, self.num_sinks), B=None, H=None, Q_LEN=L, KV_LEN=L+self.num_sinks, _compile=True)
            
            out = self.flex_attention_compiled(q, k, v, block_mask=self._mask_cache[L])
        elif self.attn_type == 'linear':
            q_kernel = F.elu(q) + 1
            k_kernel = F.elu(k) + 1
            kv = torch.einsum('bhdl,bhle->bhde', k_kernel.transpose(-2, -1), v)
            num = torch.einsum('bhld,bhde->bhle', q_kernel, kv)
            denom = torch.einsum('bhld,bhd->bhl', q_kernel, k_kernel.sum(dim=-2)).unsqueeze(-1) + 1e-8
            out = num / denom
        else:
            if is_causal and self.num_sinks > 0:
                scale = self.head_dim ** -0.5
                attn = (q @ k.transpose(-2, -1)) * scale
                mask = torch.zeros(L, L + self.num_sinks, device=q.device, dtype=torch.bool)
                mask[:, :self.num_sinks] = True
                for i in range(L):
                    mask[i, self.num_sinks:self.num_sinks+i+1] = True
                attn = attn.masked_fill(~mask[None, None, :, :], float('-inf'))
                attn = F.softmax(attn, dim=-1)
                out = attn @ v
            else:
                use_flash = q.is_cuda and q.dtype in (torch.float16, torch.bfloat16)
                backend = SDPBackend.FLASH_ATTENTION if use_flash else SDPBackend.MATH
                with sdpa_kernel(backend):
                    out = F.scaled_dot_product_attention(q, k, v, is_causal=is_causal)
        
        if self._compute_attn_stats and self.attn_type in ['full', 'local', 'causal_full', 'causal_local']:
            scale = self.head_dim ** -0.5
            attn_weights = (q @ k.transpose(-2, -1)) * scale
            kv_len = L + self.num_sinks
            if 'local' in self.attn_type:
                q_idx = torch.arange(L, device=attn_weights.device)
                kv_idx = torch.arange(kv_len, device=attn_weights.device)
                mask = torch.zeros(L, kv_len, device=attn_weights.device, dtype=torch.bool)
                mask[:, :self.num_sinks] = True
                seq_kv_idx = kv_idx[self.num_sinks:] - self.num_sinks
                window_mask = (q_idx[:, None] - seq_kv_idx[None, :]).abs() <= WINDOW_SIZE
                mask[:, self.num_sinks:] = window_mask
                if is_causal:
                    causal_mask = q_idx[:, None] >= seq_kv_idx[None, :]
                    mask[:, self.num_sinks:] = mask[:, self.num_sinks:] & causal_mask
                mask = mask[None, None, :, :]
                attn_weights = attn_weights.masked_fill(~mask, -torch.inf)
            elif is_causal:
                causal_mask = torch.tril(torch.ones(L, kv_len, device=attn_weights.device, dtype=torch.bool))
                causal_mask[:, :self.num_sinks] = True
                attn_weights = attn_weights.masked_fill(~causal_mask[None, None, :, :], -torch.inf)
            self._last_attn_max = torch.max(attn_weights).item()
            attn_weights = F.softmax(attn_weights, dim=-1)
            self._last_attn_entropy = -(attn_weights * torch.log(attn_weights + 1e-9)).sum(dim=-1).mean().item()
            
            
        # Reshape back: (B, num_heads, L, head_dim) -> (B, L, num_heads, head_dim) -> (B, L, D)
        out = out.transpose(1, 2).reshape(B, L, self.hidden_dim)
        out = self.out_proj(out)
        return out 

class AttentionBlock1D(nn.Module):
    #Expects N, L, D
    def __init__(self, hidden_dim, injection_type, norm_type, 
                recall_inner, spectral = False, 
                qk_normalization = False, post_relu = False,
                residual_method = 'add', attn_type='full', max_seq_len=None, num_sinks=0):
        
        #Attn
        super().__init__()
        self.injection_type = injection_type
        self.norm_type = norm_type
        self.recall_inner = recall_inner
        self.residual_method = residual_method
        self.hidden_dim = hidden_dim
        self._init_residual()
        
        if attn_type == 'conv':
            self.attn = ConvAttn(hidden_dim, concatenated = False)
        else:
            self.attn = MHA(hidden_dim, num_heads=8, attn_type=attn_type, qk_normalization=qk_normalization,
                            max_seq_len=max_seq_len, num_sinks=num_sinks)
        self.norm1 = nn.RMSNorm(hidden_dim)
        self.norm2 = nn.RMSNorm(hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Linear(hidden_dim * 4, hidden_dim)
        )
        self.post_relu = nn.ReLU() if post_relu else nn.Identity()
        
        
        match self.injection_type:
            case 'add':
                self.injection_func = lambda x, h: x + h
            case 'linear' | 'concat':
                self.Wh = nn.Linear(hidden_dim, hidden_dim)
                self.Wx = SpectralLinear(hidden_dim) if spectral else nn.Linear(hidden_dim, hidden_dim)
                self.injection_func = lambda x, h: self.Wx(x) + self.Wh(h)
            case 'none':
                self.injection_func = lambda x, h: x
            case _:
                raise ValueError(f"Invalid injection type: {self.injection_type}. Must be 'add', 'linear', 'concat', or 'none'")
        
        match self.norm_type:
            case 'pre':
                self.pre_norm_func = self.norm1
                self.peri_norm_func = nn.Identity()
                self.post_norm_func = nn.Identity()
            case 'peri':
                self.pre_norm_func = self.norm1
                self.peri_norm_func = self.norm2
                self.post_norm_func = nn.Identity()
            case 'post':
                self.pre_norm_func = nn.Identity()
                self.peri_norm_func = nn.Identity()
                self.post_norm_func = self.norm2
            case _:
                raise ValueError(f"Invalid norm_type: {self.norm_type}. Must be 'pre', 'peri', or 'post'")
                
    #Possible norms:
    #Pre
    #Post
    #Peri
    def forward(self, x, h):
        if not self.recall_inner:
            x = self.injection_func(x, h)
        
        shortcut = x
        if self.recall_inner: 
            x = self.injection_func(x, h)
        x = self.pre_norm_func(x)
        x = self.attn(x)
        x = self.post_norm_func(self.residual_func(shortcut, self.peri_norm_func(x)))
        
        shortcut = x
        if self.recall_inner:
            x = self.injection_func(x, h)
        x = self.pre_norm_func(x)
        x = self.mlp(x)
        x = self.post_norm_func(self.residual_func(shortcut, self.peri_norm_func(x)))
        x = self.post_relu(x)
        return x

    def _init_residual(self):
        match self.residual_method:
            case 'add':
                self.residual_func = lambda h, u: h + u
            case 'gru':
                self.gru = nn.GRUCell(self.hidden_dim, self.hidden_dim)
                def residual_func(h, u):
                    B, L, D = h.shape
                    merged = self.gru(u.reshape(B * L, D), h.reshape(B * L, D))
                    return merged.reshape(B, L, D)
                self.residual_func = residual_func
            case 'gate':
                self.gate_h = nn.Linear(self.hidden_dim, self.hidden_dim)
                self.gate_u = nn.Linear(self.hidden_dim, self.hidden_dim)
                def residual_func(h, u):
                    z_h = F.sigmoid(self.gate_h(h))
                    z_u = F.sigmoid(self.gate_u(u))
                    return z_h * h + z_u * u
                self.residual_func = residual_func

class ConcatAttentionBlock(nn.Module):
    def __init__(self, hidden_dim, norm_type, qk_normalization = False,
                residual_method = 'add', attn_type='full', max_seq_len=None, num_sinks=0):
        super().__init__()
        
        self.norm_type = norm_type
        self.residual_method = residual_method
        self.hidden_dim = hidden_dim
        if attn_type == 'conv':
            self.attn = ConvAttn(hidden_dim, concatenated = True)
        else:
            self.attn = MHA(hidden_dim, num_heads=8, attn_type=attn_type, qk_normalization=qk_normalization, 
                            concatenated = True, max_seq_len=max_seq_len, num_sinks=num_sinks)
        self.norm1 = nn.RMSNorm(hidden_dim)
        self.norm2 = nn.RMSNorm(hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim * 4),
            nn.GELU(),
            nn.Linear(hidden_dim * 4, hidden_dim)
        )
        
        self._init_residual()
        
        match self.norm_type:
            case 'pre':
                self.pre_norm_func = self.norm1
                self.peri_norm_func = nn.Identity()
                self.post_norm_func = nn.Identity()
            case 'peri':
                self.pre_norm_func = self.norm1
                self.peri_norm_func = self.norm2
                self.post_norm_func = nn.Identity()
            case 'post':
                self.pre_norm_func = nn.Identity()
                self.peri_norm_func = nn.Identity()
                self.post_norm_func = self.norm2
            case _:
                raise ValueError(f"Invalid norm_type: {self.norm_type}. Must be 'pre', 'peri', or 'post'")
            
    def forward(self, x, h):
        #Attn
        shortcut = x
        x = self.pre_norm_func(x)
        x = torch.cat([x, h], dim = -1)
        x = self.post_norm_func(self.residual_func(shortcut, self.peri_norm_func(self.attn(x))))
        
        #MLP
        shortcut = x
        x = self.pre_norm_func(x)
        x = torch.cat([x, h], dim = -1)
        x = self.post_norm_func(self.residual_func(shortcut, self.peri_norm_func(self.mlp(x))))
        return x

    def _init_residual(self):
        match self.residual_method:
            case 'add':
                self.residual_func = lambda h, u: h + u
            case 'gru':
                self.gru = nn.GRUCell(self.hidden_dim, self.hidden_dim)
                def residual_func(h, u):
                    B, L, D = h.shape
                    merged = self.gru(u.reshape(B * L, D), h.reshape(B * L, D))
                    return merged.reshape(B, L, D)
                self.residual_func = residual_func
            case 'gate':
                self.gate_h = nn.Linear(self.hidden_dim, self.hidden_dim)
                self.gate_u = nn.Linear(self.hidden_dim, self.hidden_dim)
                nn.init.zeros_(self.gate_u.weight)
                def residual_func(h, u):
                    z_h = F.sigmoid(self.gate_h(h))
                    z_u = F.sigmoid(self.gate_u(u))
                    return z_h * h + z_u * u
                self.residual_func = residual_func

class mHCAttentionBlock(nn.Module):
    def __init__(self, hidden_dim, norm_type,
                qk_normalization = False, lanes=3, attn_type='full', max_seq_len=None, num_sinks=0):
        
        #Attn
        super().__init__()
        self.norm_type = norm_type
        self.hidden_dim = hidden_dim
        self.lanes = lanes
        
        if attn_type == 'conv':
            self.attn = ConvAttn(hidden_dim, concatenated=True)
        else:
            self.attn = MHA(hidden_dim, num_heads=8, attn_type=attn_type, qk_normalization=qk_normalization, 
                            concatenated=True, max_seq_len=max_seq_len, num_sinks=num_sinks)
        self.norm1 = nn.RMSNorm(hidden_dim)
        self.norm2 = nn.RMSNorm(hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim * 4),
            nn.GELU(),
            nn.Linear(hidden_dim * 4, hidden_dim)
        )
        
        match self.norm_type:
            case 'pre':
                self.pre_norm_func = self.norm1
                self.peri_norm_func = nn.Identity()
                self.post_norm_func = nn.Identity()
            case 'peri':
                self.pre_norm_func = self.norm1
                self.peri_norm_func = self.norm2
                self.post_norm_func = nn.Identity()
            case 'post':
                self.pre_norm_func = nn.Identity()
                self.peri_norm_func = nn.Identity()
                self.post_norm_func = self.norm2
            case _:
                raise ValueError(f"Invalid norm_type: {self.norm_type}. Must be 'pre', 'peri', or 'post'")

        perms = list(permutations(range(lanes)))
        perm_matrices = torch.zeros(len(perms), lanes, lanes)
        for i, perm in enumerate(perms):
            perm_matrices[i] = torch.eye(lanes)[:, perm]
        self.register_buffer("perm_matrices", perm_matrices)
        
        #mHC logits
        #Currently n! + 2n parameters
        #Could do it nn!d + 2n^2 parameters, for dynamic? 
        #for N = 3, that's ~18d, for N = 4 that's ~100d
        self.perm_logits = nn.Parameter(torch.zeros(factorial(lanes)))
        self.in_scalars = nn.Parameter(torch.ones(lanes) / lanes)
        self.out_scalars = nn.Parameter(torch.zeros(lanes))
        
    def forward(self, x, h):
        #x should be #[Lanes, B, L, D]
        mixing_M = einsum('p,pij->ij', F.softmax(self.perm_logits, dim=0), self.perm_matrices) #[lanes, lanes]
        
        shortcut = x
        input = self.pre_norm_func(einsum('k,k...->...', self.in_scalars, x))
        #Concatenating with h
        input = torch.cat([input, h], dim = -1)
        layer_output = self.attn(input).unsqueeze(0).repeat(self.lanes, 1, 1, 1)
        layer_output = self.peri_norm_func(layer_output)
        layer_output = einsum('k,k...->k...', self.out_scalars, layer_output)
        shortcut_mixing = einsum('kl,l...->k...', mixing_M, shortcut)
        x = self.post_norm_func(layer_output + shortcut_mixing)
        
        shortcut = x
        input = self.pre_norm_func(einsum('k,k...->...', self.in_scalars, x))
        input = torch.cat([input, h], dim = -1)
        layer_output = self.mlp(input).unsqueeze(0).repeat(self.lanes, 1, 1, 1)
        layer_output = self.peri_norm_func(layer_output)
        layer_output = einsum('k,k...->k...', self.out_scalars, layer_output)
        shortcut_mixing = einsum('kl,l...->k...', mixing_M, shortcut)
        x = self.post_norm_func(layer_output + shortcut_mixing)
        
        return x

class SpectralLinear(nn.Module):
    #Linear layer with spectral norm < 1
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.lin = nn.Linear(dim, dim)
        self.lin = nn.utils.parametrizations.spectral_norm(self.lin)
        self.alpha_logit = nn.Parameter(torch.tensor(0.0))
    def forward(self, x):
        return F.sigmoid(self.alpha_logit) * self.lin(x)

class BasicBlock1D(nn.Module):
    """Basic residual block class 1D"""

    expansion = 1

    #Kernel size used to be 3
    def __init__(self, in_planes, planes, stride=1, group_norm=False, relu=True):
        super().__init__()
        self.conv1 = nn.Conv1d(in_planes, planes, kernel_size=11,
                               stride=stride, padding=5, bias=False)
        self.gn1 = nn.GroupNorm(4, planes, affine=False) if group_norm else nn.Sequential()
        self.conv2 = nn.Conv1d(planes, planes, kernel_size=11,
                               stride=1, padding=5, bias=False)
        self.gn2 = nn.GroupNorm(4, planes, affine=False) if group_norm else nn.Sequential()
        self.relu = relu

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(nn.Conv1d(in_planes, self.expansion * planes,
                                                    kernel_size=1, stride=stride, bias=False))

    def forward(self, x):
        out = F.relu(self.gn1(self.conv1(x)))
        out = self.gn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out) if self.relu else out
        return out


class BasicBlock2D(nn.Module):
    """Basic residual block class 2D"""

    expansion = 1

    def __init__(self, in_planes, planes, stride=1, group_norm=False):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.gn1 = nn.GroupNorm(4, planes, affine=False) if group_norm else nn.Sequential()
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.gn2 = nn.GroupNorm(4, planes, affine=False) if group_norm else nn.Sequential()

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(nn.Conv2d(in_planes, self.expansion * planes,
                                                    kernel_size=1, stride=stride, bias=False))

    def forward(self, x):
        out = F.relu(self.gn1(self.conv1(x)))
        out = self.gn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out
