import torch
from torch import nn, einsum
import torch.nn.functional as F
from itertools import permutations
from math import factorial
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.nn.attention.flex_attention import flex_attention, create_block_mask

#Sliding window for local attention
WINDOW_SIZE = 5
def get_sliding_window(sinks):
    def sliding_window(b, h, q_idx, kv_idx):
        sink = kv_idx <= sinks - 1
        window_mask = (q_idx - (kv_idx - sinks)).abs() <= WINDOW_SIZE
        return window_mask | sink
    return sliding_window

#Sliding window for 2d
def get_sliding_window_2d(sinks, width):
    def get_width_height(x):
        true_x = x - sinks
        x_width = true_x % width
        x_height = true_x // width
        return x_width, x_height
    def sliding_window(b, h, q_idx, kv_idx):
        sink = kv_idx <= sinks - 1
        q_w, q_h = get_width_height(q_idx)
        k_w, k_h = get_width_height(kv_idx)
        diff = torch.abs(q_w - k_w) + torch.abs(q_h - k_h)
        return sink | diff <= WINDOW_SIZE
    return sliding_window

#Convolutional attention
class ConvAttn(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.conv1 = nn.Conv1d(input_dim, output_dim, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.conv2 = nn.Conv1d(output_dim, output_dim, kernel_size=3,
                               stride=1, padding=1, bias=False)
    def forward(self, x):
        #[B, L, (2)D]
        x = x.transpose(1, 2) #[B, D (C), L]
        x = self.conv2(F.relu(self.conv1(x)))
        return x.transpose(1, 2) #[B, L, D]
    
#Multi-headed attention with different attention methods
class MHA(nn.Module):
    def __init__(self, input_dim, output_dim, num_heads, attn_type = 'full', 
                 qk_normalization = False, num_sinks=1):
        super().__init__()
        self.input_dim = output_dim
        self.num_heads = num_heads
        self.head_dim = output_dim // num_heads
        self.attn_type = attn_type
        self.num_sinks = num_sinks
        assert output_dim % num_heads == 0, f"hidden_dim {output_dim} must be divisible by num_heads {num_heads}"
        
        self.qkv = nn.Linear(input_dim, output_dim * 3)            
        self.out_proj = nn.Linear(output_dim, output_dim)
        
        self.q_norm = nn.RMSNorm(self.head_dim) if qk_normalization else nn.Identity()
        self.k_norm = nn.RMSNorm(self.head_dim) if qk_normalization else nn.Identity()
        
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
        qkv = self.qkv(x)  # (B, L, 3*D)
        q, k, v = qkv.chunk(3, dim=-1)  # Each: (B, L, D)
        
        # Reshape to (B, H, L, D)
        q = q.reshape(B, self.num_heads, L, self.head_dim)
        k = k.reshape(B, self.num_heads, L, self.head_dim)
        v = v.reshape(B, self.num_heads, L, self.head_dim)
        
        q, k = self.q_norm(q), self.k_norm(k)
        
        if self.num_sinks > 0:
            sink_k = self.sink_k.unsqueeze(0).expand(B, -1, -1, -1).transpose(1, 2).to(k.dtype)
            sink_v = self.sink_v.unsqueeze(0).expand(B, -1, -1, -1).transpose(1, 2).to(v.dtype)
            k = torch.cat([sink_k, k], dim=2)
            v = torch.cat([sink_v, v], dim=2)
        
        match self.attn_type:
            case 'local':
                if L not in self._mask_cache:
                    self._mask_cache[L] = create_block_mask(get_sliding_window(self.num_sinks), B=None, H=None, Q_LEN=L, KV_LEN=L+self.num_sinks, _compile=True)
                out = self.flex_attention_compiled(q, k, v, block_mask=self._mask_cache[L])
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
            
        if self._compute_attn_stats and self.attn_type != 'linear':
            self._compute_attention_stats(q, k, L)
            
        # Reshape back: (B, num_heads, L, head_dim) -> (B, L, num_heads, head_dim) -> (B, L, D)
        out = out.transpose(1, 2).reshape(B, L, self.hidden_dim)
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
            case 'mhc':
                pass
    
    def forward(self, x, h):
        if self.lanes > 1:
            return self._forward_mhc(x, h)

        if not self.recall_inner:
            x = self.injection_func(x, h)
            
        def run_block(block, x_in):
            shortcut = x_in
            if self.recall_inner:
                x = self.injection_func(x, h)
            x = self.pre_norm_func(x)
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
            layer_output = block(input).unsqueeze(0).repeat(self.lanes, 1, 1, 1)
            layer_output = self.peri_norm_func(layer_output)
            layer_output = einsum('k,k...->k...', self.out_scalars, layer_output)
            shortcut_mixing = einsum('kl,l...->k...', mixing_M, shortcut)
            return self.post_norm_func(layer_output + shortcut_mixing)

        x = run_block(self.attn, x)
        x = run_block(self.mlp, x)
        return x
        
        
