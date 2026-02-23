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
import natten

#Convolutional attention
class ConvAttn(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size = 3, *, spatial_dims):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.is_setup = False
        self.lin = nn.Linear(input_dim, output_dim)
        self.kernel_size = kernel_size
        self.setup(spatial_dims)
    
    def setup(self, dimensions, device=None):
        if self.is_setup:
            return
        if dimensions not in (1, 2, 3):
            raise ValueError(f"spatial_dims must be 1, 2, or 3; got {dimensions}")
        #Sets up on first try
        match dimensions:
            case 1:
                conv_func = nn.Conv1d
            case 2:
                conv_func = nn.Conv2d
            case 3: 
                conv_func = nn.Conv3d
        
        self.conv1 = conv_func(self.output_dim, self.output_dim, kernel_size=self.kernel_size,
                               stride=1, padding=1, bias=False)
        self.conv2 = conv_func(self.output_dim, self.output_dim, kernel_size=self.kernel_size,
                               stride=1, padding=1, bias=False)
        if device is not None:
            self.conv1 = self.conv1.to(device)
            self.conv2 = self.conv2.to(device)
        self.is_setup = True
        
    def forward(self, x, ccot_tokens):
        #[B, ..., 2D]
        x = F.relu(self.lin(x)) #[B, ..., D]
        x = x.permute(0, -1, *range(1, x.dim() - 1)) #[B, D, ...]
        x = self.conv2(F.relu(self.conv1(x)))
        return x.permute(0, *range(2, x.dim()), 1), ccot_tokens #[B, ..., D]

#Multi-headed attention with different attention methods
class MHA(nn.Module):
    def __init__(self, input_dim, output_dim, num_heads, attn_type = 'full', 
                 qk_normalization = False, num_sinks=1, max_seq_len = 512, kernel_size=5):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_heads = num_heads
        self.head_dim = output_dim // num_heads
        self.attn_type = attn_type
        self.num_sinks = num_sinks
        self.kernel_size = kernel_size
        assert output_dim % num_heads == 0, f"hidden_dim {output_dim} must be divisible by num_heads {num_heads}"
        
        self.qkv = nn.Linear(input_dim, output_dim * 3)            
        self.out_proj = nn.Linear(output_dim, output_dim)
        
        self.q_norm = nn.RMSNorm(self.head_dim) if qk_normalization else nn.Identity()
        self.k_norm = nn.RMSNorm(self.head_dim) if qk_normalization else nn.Identity()
        self.is_setup = False
        
        if self.num_sinks > 0:
            self.sink_k = nn.Parameter(torch.zeros(num_sinks, num_heads, self.head_dim))
            self.sink_v = nn.Parameter(torch.zeros(num_sinks, num_heads, self.head_dim))
        
        self._compute_attn_stats = False
        
    def setup(self, x_shape, device=None):
        self.rope = RoPENd((*x_shape, self.head_dim))
        if device is not None:
            self.rope = self.rope.to(device)
        self.shape = x_shape
        self.is_setup = True
    
    def forward(self, x, ccot_tokens):
        # x: (B, ..., D), and self-attention requires (B, N, L, D)
        # ccot tokens are B, I, D
        if not self.is_setup or self.shape != x.shape[1:-1]:
            self.setup(x.shape[1:-1], device=x.device)
        spatial_dims = x.shape[1:-1]
        L = math.prod(spatial_dims)
        B = x.shape[0] 
        qkv = self.qkv(x)  # (B, ..., 3*D)
        
        num_ccot = ccot_tokens.shape[1]
        q_ccot = k_ccot = v_ccot = None
        #Working with chain of thought
        if num_ccot > 0:
            qkv_ccot = self.qkv(ccot_tokens)
            q_ccot, k_ccot, v_ccot = qkv_ccot.chunk(3, dim = -1) #Each (I, D)
            q_ccot = q_ccot.reshape(B, num_ccot, self.num_heads, self.head_dim).transpose(1, 2)
            k_ccot = k_ccot.reshape(B, num_ccot, self.num_heads, self.head_dim).transpose(1, 2)
            v_ccot = v_ccot.reshape(B, num_ccot, self.num_heads, self.head_dim).transpose(1, 2)
            q_ccot, k_ccot = self.q_norm(q_ccot), self.k_norm(k_ccot)
            
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
        if q.dtype != v.dtype:
            q = q.to(v.dtype)
        if k.dtype != v.dtype:
            k = k.to(v.dtype)
           
        match self.attn_type:
            case 'local':
                #Two steps: first get non-CoT, then get CoT, add them together.
                #[B, N, ..., D] -> [B, ..., N, D]
                nspatial = len(spatial_dims)
                q = q.permute(0, *range(2, 2 + nspatial), 1, -1).contiguous()
                k = k.permute(0, *range(2, 2 + nspatial), 1, -1).contiguous()
                v = v.permute(0, *range(2, 2 + nspatial), 1, -1).contiguous()
                attn_funcs = [natten.na1d, natten.na2d, natten.na3d]
                attn_func = attn_funcs[nspatial - 1] #Get the correct attention dim
                
                #Additional KV: CoT + sinks 
                add_k = []
                add_v = []
                
                if self.num_sinks > 0:
                    #Natten wants [B, L, N, D] - keys start as [L, N, D]
                    add_k.append(self.sink_k.unsqueeze(0).expand(B, -1, -1, -1).to(k.dtype))
                    add_v.append(self.sink_v.unsqueeze(0).expand(B, -1, -1, -1).to(v.dtype))
                if num_ccot > 0:
                    # k_ccot/v_ccot are (B, H, L, D) -- natten wants (B, L, H, D)
                    add_k.append(k_ccot.transpose(1, 2).contiguous())
                    add_v.append(v_ccot.transpose(1, 2).contiguous())
                
                additional_k = torch.cat(add_k, dim=1) if add_k else None
                additional_v = torch.cat(add_v, dim=1) if add_v else None
                
                # Optional NATTEN tuning for Blackwell (B200). Falls back automatically otherwise.
                natten_kwargs = {}
                if q.is_cuda and torch.cuda.get_device_capability(q.device) in {(10, 0), (10, 3)}:
                    natten_kwargs["backend"] = "blackwell-fna"
                    natten_kwargs["run_persistent_kernel"] = True
                    natten_kwargs["attention_kwargs"] = {
                        "backend": "blackwell-fmha",
                        "run_persistent_kernel": True,
                    }
                    match nspatial:
                        case 1:
                            natten_kwargs['q_tile_shape'] = (16, 16)
                            natten_kwargs['kv_tile_shape'] = (8, 16)
                            natten_kwargs['backward_q_tile_shape'] = (16, 8)
                            natten_kwargs['backward_kv_tile_shape'] = (8, 16)
                        case 2:
                            natten_kwargs['q_tile_shape'] = (256,)
                            natten_kwargs['kv_tile_shape'] = (128,)
                            natten_kwargs['backward_q_tile_shape'] = (128,)
                            natten_kwargs['backward_kv_tile_shape'] = (128,)
                    
                out = attn_func(q, k, v, kernel_size = self.kernel_size, additional_keys = additional_k,
                                        additional_values = additional_v, **natten_kwargs)
                out = out.reshape(B, *spatial_dims, self.output_dim)
                out = self.out_proj(out) #Result for tokens seeing from sinks and CCOT
                
                #Now getting CCOT looking from tokens
                if num_ccot > 0:
                    #K and V need to be [sinks, tokens, ccot], Q is just [ccot]
                    #Everything needs to be [B, H, L, D]
                    #Already have q_ccot, k_ccot, v_ccot from earlier
                    add_k.append(k.reshape(B, -1, self.num_heads, self.head_dim))
                    add_v.append(v.reshape(B, -1, self.num_heads, self.head_dim))
                    add_k = torch.cat(add_k, dim = 1)
                    add_v = torch.cat(add_v, dim = 1)
                    #At this point K/V are shape [B, L, H, D]
                    add_k = add_k.transpose(1, 2)
                    add_v = add_v.transpose(1, 2)
                    
                    with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
                        ccot_tokens = F.scaled_dot_product_attention(q_ccot, add_k, add_v)
                    ccot_tokens = ccot_tokens.transpose(1, 2).reshape(B, num_ccot, self.output_dim)
                    ccot_tokens = self.out_proj(ccot_tokens)
    
            case 'full':
                q = q.reshape(B, self.num_heads, L, self.head_dim)
                k = k.reshape(B, self.num_heads, L, self.head_dim)
                v = v.reshape(B, self.num_heads, L, self.head_dim)
                if self.num_sinks > 0:
                    sink_k = self.sink_k.unsqueeze(0).expand(B, -1, -1, -1).transpose(1, 2).to(k.dtype)
                    sink_v = self.sink_v.unsqueeze(0).expand(B, -1, -1, -1).transpose(1, 2).to(v.dtype)
                    k = torch.cat([sink_k, k], dim = 2)
                    v = torch.cat([sink_v, v], dim = 2)
                if num_ccot > 0:
                    q = torch.cat([q, q_ccot], dim = 2)
                    k = torch.cat([k, k_ccot], dim = 2)
                    v = torch.cat([v, v_ccot], dim = 2)
                use_flash = q.is_cuda and q.dtype in (torch.float16, torch.bfloat16)
                backend = SDPBackend.FLASH_ATTENTION if use_flash else SDPBackend.MATH
                with sdpa_kernel(backend):
                    out = F.scaled_dot_product_attention(q, k, v) #[B, N, L + C, H]
                
                # Compute attention stats only when explicitly enabled by diagnostics.
                if self._compute_attn_stats and len(self.shape) == 1:
                    with torch.no_grad():
                        self._compute_attention_stats(q[:, :, :L, :], k)
                out = out.transpose(1, 2).reshape(B, -1, self.output_dim)
                out = self.out_proj(out)
                ccot_tokens = out[:, L:, :]
                out = out[:, :L, :].reshape(B, *spatial_dims, self.output_dim)
                
        return out, ccot_tokens

    def _compute_attention_stats(self, q, k):
        # Stats are computed assuming non-causal attention.
        scale = self.head_dim ** -0.5
        attn_weights = (q @ k.transpose(-2, -1)) * scale
        self._last_attn_max = torch.max(attn_weights).item()
        attn_weights = F.softmax(attn_weights, dim=-1)
        self._last_attn_entropy = -(attn_weights * torch.log(attn_weights + 1e-9)).sum(dim=-1).mean().item()

#Core block
class AttentionBlock(nn.Module):
    def __init__(self, hidden_dim, lanes = 1,
                injection_type = 'none', norm_type = 'peri', 
                recall_inner = False, qk_normalization = False,
                residual_method = 'add', attn_type='full', max_seq_len=None, num_sinks=0, kernel_size=5,
                post_relu = False, *, spatial_dims):
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
        self.kernel_size = kernel_size
        self.post_relu = nn.ReLU() if post_relu else nn.Identity()
        
        # For LSTM: register buffer to store cell state (will be reset each forward pass)
        if residual_method == 'lstm':
            self.register_buffer('_lstm_cell_state', None)
        
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
            self.attn = ConvAttn(input_hidden_dim, hidden_dim, kernel_size=kernel_size, spatial_dims=spatial_dims)
        else:
            self.attn = MHA(
                input_hidden_dim, hidden_dim,
                num_heads=8,
                attn_type=self.attn_type,
                qk_normalization=self.qk_normalization,
                num_sinks=self.num_sinks,
                kernel_size=self.kernel_size,
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
    
    def forward(self, x, h, ccot_tokens):
        #x, h = [B, ..., C] depending on problem, but we don't need to deal with it in full block, just the individual mechanisms
        B, D = x.shape[0], x.shape[-1]
        
        # Reset LSTM cell state at start of each forward pass
        spatial_dims = x.shape[1:-1]
        L = math.prod(spatial_dims)
        if self.residual_method == 'lstm':
            self._lstm_cell_state = None
            
        if self.lanes > 1:
            return self._forward_mhc(x, h), ccot_tokens #ccot tokens will not be used, but kept for backwards compatibility

        if not self.recall_inner:
            x = self.injection_func(x, h)
        
        def run_block(block_name, x, ccot_tokens):
            #Both x and ccot are [B, *, D]
            shortcut_x, shortcut_ccot = x, ccot_tokens
            x, ccot_tokens = self.pre_norm_func(x), self.pre_norm_func(ccot_tokens)
            if self.recall_inner:
                x = self.injection_func(x, h)
                if self.injection_type == 'concat': #No "context" for ccot
                    ccot_tokens = torch.cat([ccot_tokens, torch.zeros_like(ccot_tokens)], dim = -1)
            if block_name == 'attn':
                x, ccot_tokens = self.attn(x, ccot_tokens) 
            if block_name == 'mlp':
                x, ccot_tokens = self.mlp(x), self.mlp(ccot_tokens)
            x = self.post_norm_func(self.residual_func(shortcut_x, self.peri_norm_func(x)))
            if shortcut_ccot.size(1) > 0:
                ccot_tokens = self.post_norm_func(self.residual_func(shortcut_ccot, self.peri_norm_func(ccot_tokens)))
            return x, ccot_tokens
        
        x, ccot_tokens = run_block('attn', x, ccot_tokens)
        x, ccot_tokens = run_block('mlp', x, ccot_tokens)
        
        return self.post_relu(x), self.post_relu(ccot_tokens)

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
        
        
