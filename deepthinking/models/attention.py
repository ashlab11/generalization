from sympy import Q, false
import math
import torch
from torch import nn, einsum
import torch.nn.functional as F
from itertools import permutations
from math import factorial
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.nn.attention.flex_attention import flex_attention, create_block_mask
from torchtune.modules import RotaryPositionalEmbeddings
from .ropend import RoPENd

#Sliding window for local attention
WINDOW_SIZE = 5
def get_sliding_window(sinks, spatial_dims):
    """General sliding window for arbitrary spatial dimensions using Manhattan distance."""
    
    # Compute cumulative products for coordinate conversion (row-major flattening)
    # cumprods[i] = product of spatial_dims[i+1:] 
    # For (H, W): cumprods = [W, 1] so row = idx // W, col = idx % W
    # For (D1, D2, D3): cumprods = [D2*D3, D3, 1]
    cumprods_list = []
    for i in range(len(spatial_dims)):
        prod = math.prod(spatial_dims[i+1:]) if i < len(spatial_dims) - 1 else 1
        cumprods_list.append(prod)
    cumprods = torch.tensor(cumprods_list, dtype=torch.long)
    spatial_dims_tensor = torch.tensor(spatial_dims, dtype=torch.long)
    
    def idx_to_coords(idx):
        """Convert flattened index (after sinks) to multi-dimensional coordinates.
        Returns tensor of shape (num_dims, ...) where first dim is coordinate dimension."""
        true_idx = idx - sinks
        # Expand for broadcasting: (num_dims, 1) and (1, ...)
        cumprods_expanded = cumprods.view(-1, *([1] * idx.dim()))
        spatial_dims_expanded = spatial_dims_tensor.view(-1, *([1] * idx.dim()))
        true_idx_expanded = true_idx.unsqueeze(0)
        
        # Compute all coordinates at once: (num_dims, ...)
        coords = (true_idx_expanded // cumprods_expanded) % spatial_dims_expanded
        return coords
    
    def sliding_window(b, h, q_idx, kv_idx):
        sink = kv_idx <= sinks - 1
        
        # Convert indices to coordinates: (num_dims, ...)
        q_coords = idx_to_coords(q_idx)
        kv_coords = idx_to_coords(kv_idx)
        
        # Compute Manhattan distance (L1 norm) across all dimensions
        diff = torch.abs(q_coords.float() - kv_coords.float()).sum(dim=0)
        
        window_mask = diff <= WINDOW_SIZE
        return sink | window_mask
    
    return sliding_window

def get_sliding_window_1d(sinks):
    """1D sliding window (kept for backward compatibility)."""
    def sliding_window(b, h, q_idx, kv_idx):
        sink = kv_idx <= sinks - 1
        window_mask = (q_idx - (kv_idx - sinks)).abs() <= WINDOW_SIZE
        return window_mask | sink
    return sliding_window

#Convolutional attention
class ConvAttn(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.is_setup = False
        self.lin = nn.Linear(input_dim, output_dim)
        self.conv1 = nn.Conv1d(output_dim, output_dim, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.conv2 = nn.Conv1d(output_dim, output_dim, kernel_size=3,
                               stride=1, padding=1, bias=False)
    
    def setup(self, dimensions, device=None):
        #Sets up on first try
        match dimensions:
            case 1:
                conv_func = nn.Conv1d
            case 2:
                conv_func = nn.Conv2d
            case 3: 
                conv_func = nn.Conv3d
        
        self.conv1 = conv_func(self.output_dim, self.output_dim, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.conv2 = conv_func(self.output_dim, self.output_dim, kernel_size=3,
                               stride=1, padding=1, bias=False)
        if device is not None:
            self.conv1 = self.conv1.to(device)
            self.conv2 = self.conv2.to(device)
        self.is_setup = True
        
    def forward(self, x):
        #Lazy initialization
        if not self.is_setup:
            self.setup(len(x.shape[1:-1]), device=x.device)

        #[B, ..., 2D]
        x = F.relu(self.lin(x)) #[B, ..., D]
        x = x.permute(0, -1, *range(1, x.dim() - 1)) #[B, D, ...]
        x = self.conv2(F.relu(self.conv1(x)))
        return x.permute(0, *range(2, x.dim()), 1) #[B, ..., D]
    
#Multi-headed attention with different attention methods
class MHA(nn.Module):
    def __init__(self, input_dim, output_dim, num_heads, attn_type = 'full', 
                 qk_normalization = False, num_sinks=1, max_seq_len = 512):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_heads = num_heads
        self.head_dim = output_dim // num_heads
        self.attn_type = attn_type
        self.num_sinks = num_sinks
        assert output_dim % num_heads == 0, f"hidden_dim {output_dim} must be divisible by num_heads {num_heads}"
        
        self.qkv = nn.Linear(input_dim, output_dim * 3)            
        self.out_proj = nn.Linear(output_dim, output_dim)
        
        self.q_norm = nn.RMSNorm(self.head_dim) if qk_normalization else nn.Identity()
        self.k_norm = nn.RMSNorm(self.head_dim) if qk_normalization else nn.Identity()
        self.is_setup = False
        
        if self.num_sinks > 0:
            self.sink_k = nn.Parameter(torch.zeros(num_sinks, num_heads, self.head_dim))
            self.sink_v = nn.Parameter(torch.zeros(num_sinks, num_heads, self.head_dim))
        
        # Cache for block masks at different sequence lengths
        self._mask_cache = {}
        if 'local' in attn_type:
            self.flex_attention_compiled = torch.compile(flex_attention)
        self._compute_attn_stats = False
        
    def setup(self, x_shape, device=None):
        self.rope = RoPENd((*x_shape, self.head_dim))
        if device is not None:
            self.rope = self.rope.to(device)
        self.shape = x_shape
        self.is_setup = True
    
    def forward(self, x):
        # x: (B, ..., D), and self-attention requires (B, N, L, D)
        if not self.is_setup or self.shape != x.shape[1:-1]:
            self.setup(x.shape[1:-1], device=x.device)
        spatial_dims = x.shape[1:-1]
        L = math.prod(spatial_dims)
        B = x.shape[0] 
        qkv = self.qkv(x)  # (B, ..., 3*D)
        q, k, v = qkv.chunk(3, dim=-1)  # Each: (B, ..., D)
        
        # Reshape to (B, ..., N, D) and then permute to (B, N, ..., D).
        # Build axes from spatial rank to avoid using stale q-shape during assignment.
        permute_dims = (0, -2, *range(1, len(spatial_dims) + 1), -1)
        q = q.reshape(B, *spatial_dims, self.num_heads, self.head_dim).permute(*permute_dims)
        k = k.reshape(B, *spatial_dims, self.num_heads, self.head_dim).permute(*permute_dims)
        v = v.reshape(B, *spatial_dims, self.num_heads, self.head_dim).permute(*permute_dims)
        
        q = self.rope(q) #(B, N, ..., D)
        k = self.rope(k)
        
        q, k = self.q_norm(q), self.k_norm(k)
        q = q.reshape(B, self.num_heads, L, self.head_dim)
        k = k.reshape(B, self.num_heads, L, self.head_dim)
        v = v.reshape(B, self.num_heads, L, self.head_dim)
        
        if self.num_sinks > 0:
            sink_k = self.sink_k.unsqueeze(0).expand(B, -1, -1, -1).transpose(1, 2).to(k.dtype)
            sink_v = self.sink_v.unsqueeze(0).expand(B, -1, -1, -1).transpose(1, 2).to(v.dtype)
            k = torch.cat([sink_k, k], dim=2)
            v = torch.cat([sink_v, v], dim=2)
        
        match self.attn_type:
            case 'local':
                # Cache by full spatial signature -- probably unnecessary, but who knows about the future
                mask_key = (tuple(spatial_dims), int(self.num_sinks))
                if mask_key not in self._mask_cache:
                    self._mask_cache[mask_key] = create_block_mask(get_sliding_window(self.num_sinks, spatial_dims), B=None, H=None, Q_LEN=L, KV_LEN=L + self.num_sinks, _compile=True)
                out = self.flex_attention_compiled(q, k, v, block_mask=self._mask_cache[mask_key])
            case 'full':
                use_flash = q.is_cuda and q.dtype in (torch.float16, torch.bfloat16)
                backend = SDPBackend.FLASH_ATTENTION if use_flash else SDPBackend.MATH
                with sdpa_kernel(backend):
                    out = F.scaled_dot_product_attention(q, k, v)
            case 'linear':
                q_kernel, k_kernel = F.elu(q) + 1, F.elu(k) + 1
                kv = torch.einsum('bhdl,bhle->bhde', k_kernel.transpose(-2, -1), v)
                num = torch.einsum('bhld,bhde->bhle', q_kernel, kv)
                denom = torch.einsum('bhld,bhd->bhl', q_kernel, k_kernel.sum(dim=-2)).unsqueeze(-1) + 1e-8
                out = num / denom
            
        if self._compute_attn_stats and self.attn_type != 'linear' and len(self.shape) == 1: #Only compute for 1D attention, others are too hard
            self._compute_attention_stats(q, k, L)
            
        # Reshape back: (B, num_heads, L, head_dim) -> (B, L, num_heads, head_dim) -> (B, *spatial_dims, D)
        out = out.transpose(1, 2).reshape(B, *spatial_dims, self.output_dim)
        out = self.out_proj(out)
        return out 

    def _compute_attention_stats(self, q, k, seq_len):
        # Stats are computed assuming non-causal attention.
        scale = self.head_dim ** -0.5
        attn_weights = (q @ k.transpose(-2, -1)) * scale
        kv_len = seq_len + self.num_sinks
        if 'local' in self.attn_type:
            q_idx = torch.arange(seq_len, device=attn_weights.device)
            kv_idx = torch.arange(kv_len, device=attn_weights.device)
            mask = torch.zeros(seq_len, kv_len, device=attn_weights.device, dtype=torch.bool)
            mask[:, :self.num_sinks] = True
            seq_kv_idx = kv_idx[self.num_sinks:] - self.num_sinks
            window_mask = (q_idx[:, None] - seq_kv_idx[None, :]).abs() <= WINDOW_SIZE
            mask[:, self.num_sinks:] = window_mask
            mask = mask[None, None, :, :]
            attn_weights = attn_weights.masked_fill(~mask, -torch.inf)
        self._last_attn_max = torch.max(attn_weights).item()
        attn_weights = F.softmax(attn_weights, dim=-1)
        self._last_attn_entropy = -(attn_weights * torch.log(attn_weights + 1e-9)).sum(dim=-1).mean().item()

#Core block
class AttentionBlock(nn.Module):
    def __init__(self, hidden_dim, lanes = 1,
                injection_type = 'none', norm_type = 'peri', 
                recall_inner = False, qk_normalization = False,
                residual_method = 'add', attn_type='full', max_seq_len=None, num_sinks=0,
                post_relu = False):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.lanes = lanes
        self.injection_type = injection_type
        self.norm_type = norm_type
        self.recall_inner = recall_inner
        self.qk_normalization = qk_normalization
        self.residual_method = residual_method
        self.attn_type = attn_type
        self.max_seq_len = max_seq_len
        self.num_sinks = num_sinks
        self.post_relu = nn.ReLU() if post_relu else nn.Identity()
        
        # For LSTM: register buffer to store cell state (will be reset each forward pass)
        if residual_method == 'lstm':
            self.register_buffer('_lstm_cell_state', None)
        
        assert not (lanes > 1 and residual_method != 'mhc'), "if there are multiple lanes, residual method must be mhc"
        assert not (lanes <= 1 and residual_method == 'mhc') and lanes < 6 and int(lanes) == lanes, "mhc must have integer lanes between 2-5"
        assert recall_inner or injection_type != 'concat', 'concat injection requires recall inside each subfunc'
        assert recall_inner or residual_method != 'mhc', 'mhc requires recall_inner'
        
        match self.norm_type:
            case 'pre':
                self.pre_norm_func = nn.RMSNorm(hidden_dim)
                self.peri_norm_func = nn.Identity()
                self.post_norm_func = nn.Identity()
            case 'peri':
                self.pre_norm_func = nn.RMSNorm(hidden_dim)
                self.peri_norm_func = nn.RMSNorm(hidden_dim)
                self.post_norm_func = nn.Identity()
            case 'post':
                self.pre_norm_func = nn.Identity()
                self.peri_norm_func = nn.Identity()
                self.post_norm_func = nn.RMSNorm(hidden_dim)
            case 'sandwich':
                self.pre_norm_func = nn.RMSNorm(hidden_dim)
                self.peri_norm_func = nn.Identity()
                self.post_norm_func = nn.RMSNorm(hidden_dim)
            case _:
                raise ValueError(f"Invalid norm_type: {self.norm_type}. Must be 'pre', 'peri', 'post', or 'sandwich'")
            
        match self.injection_type:
            case 'add':
                self.injection_func = lambda x, h: x + h
                input_hidden_dim = hidden_dim
            case 'linear':
                self.Wh = nn.Linear(hidden_dim, hidden_dim)
                self.Wx = nn.Linear(hidden_dim, hidden_dim)
                self.injection_func = lambda x, h: self.Wx(x) + self.Wh(h)
                input_hidden_dim = hidden_dim
            case 'concat':
                self.injection_func = lambda x, h: torch.cat([x, h], dim = -1)
                input_hidden_dim = 2 * hidden_dim
            case 'none':
                self.injection_func = lambda x, h: x
                input_hidden_dim = hidden_dim
            case _:
                raise ValueError(f"Invalid injection type: {self.injection_type}. Must be 'add', 'linear', 'concat', or 'none'")
            
        if self.attn_type == 'conv':
            self.attn = ConvAttn(input_hidden_dim, hidden_dim)
        else:
            self.attn = MHA(
                input_hidden_dim, hidden_dim,
                num_heads=8,
                attn_type=self.attn_type,
                qk_normalization=self.qk_normalization,
                num_sinks=self.num_sinks,
            )
                
        self.mlp = nn.Sequential(
            nn.Linear(input_hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Linear(hidden_dim * 4, hidden_dim)
        )
        
        if self.lanes > 1:
            #By limiting lanes to 5 max, we use Birkhoff's theorem to calculate directly rather than iteratively
            perms = list(permutations(range(lanes)))
            perm_matrices = torch.zeros(len(perms), lanes, lanes)
            for i, perm in enumerate(perms):
                perm_matrices[i] = torch.eye(lanes)[:, perm]
            self.register_buffer("perm_matrices", perm_matrices)
            
            self.perm_logits = nn.Parameter(torch.zeros(factorial(lanes)))
            self.in_scalars = nn.Parameter(torch.ones(lanes) / lanes)
            self.out_scalars = nn.Parameter(torch.zeros(lanes))
            
        self._init_residual()
        
    def _init_residual(self):
        match self.residual_method:
            case 'add':
                self.residual_func = lambda h, u: h + u
            case 'relu':
                self.residual_func = lambda h, u: F.relu(h + u)
            case 'gru':
                self.gru = nn.GRUCell(self.hidden_dim, self.hidden_dim)
                def residual_func(h, u):
                    h_flat = h.reshape(-1, h.shape[-1])
                    u_flat = u.reshape(-1, u.shape[-1])
                    merged = self.gru(u_flat, h_flat)
                    return merged.reshape(h.shape)
                self.residual_func = residual_func
            case 'lstm':
                self.lstm = nn.LSTMCell(self.hidden_dim, self.hidden_dim)
                def residual_func(h, u):
                    #[B, ..., D]
                    h_flat = h.reshape(-1, h.shape[-1])
                    u_flat = u.reshape(-1, u.shape[-1])
                    if self._lstm_cell_state is None:
                        c_flat = torch.zeros_like(h_flat)
                    else:
                        c_flat = self._lstm_cell_state
                    h_out, c_out = self.lstm(u_flat, (h_flat, c_flat))
                    self._lstm_cell_state = c_out
                    return h_out.reshape(h.shape)
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
            case 'mhc':
                pass
    
    def forward(self, x, h):
        #x, h = [B, ..., C] depending on problem, but we don't need to deal with it in full block, just the individual mechanisms
        
        # Reset LSTM cell state at start of each forward pass
        if self.residual_method == 'lstm':
            self._lstm_cell_state = None
            
        if self.lanes > 1:
            return self._forward_mhc(x, h)

        if not self.recall_inner:
            x = self.injection_func(x, h)
            
        def run_block(block, x):
            shortcut = x
            x = self.pre_norm_func(x)
            if self.recall_inner:
                x = self.injection_func(x, h)
            x = block(x)
            x = self.post_norm_func(self.residual_func(shortcut, self.peri_norm_func(x)))
            return x
        
        x = run_block(self.attn, x)
        x = run_block(self.mlp, x)
        return self.post_relu(x)

    def _forward_mhc(self, x, h):
        mixing_M = einsum('p,pij->ij', F.softmax(self.perm_logits, dim=0), self.perm_matrices)

        def run_block(block, x_in):
            shortcut = x_in
            input = self.pre_norm_func(einsum('k,k...->...', self.in_scalars, x_in))
            input = self.injection_func(input, h)
            layer_output = block(input).unsqueeze(0).repeat(self.lanes, *([1] * input.dim()))
            layer_output = self.peri_norm_func(layer_output)
            layer_output = einsum('k,k...->k...', self.out_scalars, layer_output)
            shortcut_mixing = einsum('kl,l...->k...', mixing_M, shortcut)
            return self.post_norm_func(layer_output + shortcut_mixing)

        x = run_block(self.attn, x)
        x = run_block(self.mlp, x)
        return x
        
        
