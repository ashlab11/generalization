""" dt_net_1d.py
    DeepThinking 1D convolutional neural network.

    Collaboratively developed
    by Avi Schwarzschild, Eitan Borgnia,
    Arpit Bansal, and Zeyad Emam.

    Developed for DeepThinking project
    October 2021
"""

import torch
from torch import nn
import torch.nn.functional as F
import math

from .attention import AttentionBlock
from .ema_linear import EMA_Linear

#Overarching DT Transformer class. Should work with 1D data as well as 2D data.
class DTTransformer(nn.Module):
    """DeepThinking Transformer model class"""

    def __init__(self, hidden_dim, num_blocks=1,
                 injection_type='concat', norm_type='peri', norm_before_head=True,
                 recall_inner=False, qk_normalization = False,
                 post_relu = False, residual_method = 'add', lanes = 1, attn_type='full',
                 in_channels = 1, out_channels = 2, num_sinks=0, kernel_size=5, local_attn_pad=False, ema_act = False, ccot = 'none', num_ccot_tokens = 10,
                 noise_prob = 0.0, noise_scale = 0.01, velocity = 0, *, spatial_dims, compile = False, 
                 init_norm = False, **kwargs):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_blocks = num_blocks
        self.ema_act = ema_act
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.ccot = ccot
        self.velocity = velocity
        self.compile = compile
        self.attn_type = attn_type
        self.kernel_size = kernel_size
        self.local_attn_pad = bool(local_attn_pad)
        self.init_norm = nn.RMSNorm(hidden_dim) if init_norm else nn.Identity()
        if spatial_dims not in (1, 2, 3):
            raise ValueError(f"spatial_dims must be 1, 2, or 3; got {spatial_dims}")
        
        assert self.ccot == 'none' or residual_method != 'lstm', "lstm cannot be used with continuous chain of thought"
        assert self.ccot == 'none' or lanes == 1, 'good luck getting ccot to work with mhc'
        assert not (lanes > 1 and residual_method != 'mhc'), "if there are multiple lanes, residual method must be mhc"
        assert not (lanes <= 1 and residual_method == 'mhc') and lanes < 6 and int(lanes) == lanes, "mhc must have integer lanes between 2-5"
        assert recall_inner or injection_type != 'concat', 'concat injection requires recall inside each subfunc'
        assert recall_inner or residual_method != 'mhc', 'mhc requires recall_inner'
        assert not (ccot == 'iterative' and attn_type == 'local'), "iterative ccot currently doesn't work with local attn"
        assert ccot == 'none' or attn_type != 'conv', "conv doesn't work with ccot (dimension mismatch)"
        assert velocity == 0 or (norm_type in ['peri', 'pre'] and residual_method == 'add'), 'velocity requires strictly additive residual updates'
        assert not (self.compile and ccot == 'iterative'), "can't use compilation with ccot"
               
        match self.ccot:
            case 'none':
                self.register_buffer('ccot_tokens', torch.empty(0, hidden_dim)) #I, D
            case 'fixed':
                self.ccot_tokens = nn.Parameter(torch.randn((num_ccot_tokens, hidden_dim)))
            case 'iterative':
                self.base_ccot = nn.Parameter(torch.randn(1, hidden_dim))
                self.get_new_ccot = lambda batch_size: self.base_ccot.clone().unsqueeze(0).repeat(batch_size, 1, 1)
                self.register_buffer('ccot_tokens', torch.empty(0, hidden_dim)) #I, D
            
        #Core blocks
        self.recur_blocks = nn.ModuleList([AttentionBlock(
                                           hidden_dim = hidden_dim,
                                           lanes = lanes,
                                           injection_type = injection_type,
                                           norm_type = norm_type,
                                           recall_inner = recall_inner,
                                           qk_normalization = qk_normalization, 
                                           residual_method = residual_method,
                                           attn_type = attn_type,
                                           num_sinks=num_sinks,
                                           kernel_size=kernel_size,
                                           local_attn_pad=local_attn_pad,
                                           post_relu=post_relu,
                                           spatial_dims=spatial_dims
                                           ) 
                                           for _ in range(num_blocks)])
        self.head = nn.Sequential(
            nn.RMSNorm(hidden_dim) if norm_before_head else nn.Identity(),
            nn.Linear(hidden_dim, hidden_dim), 
            nn.GELU(), 
            nn.Linear(hidden_dim, self.out_channels)
        )

        #Small parameters
        self.init_norm = nn.RMSNorm(hidden_dim)
        
        self.projection = nn.Linear(self.in_channels, hidden_dim)
        self.is_mhc = residual_method == 'mhc'
        self.lane_combine = nn.Parameter(torch.ones(lanes)) if self.is_mhc else None
        
        #Noise for path dependence
        self.noise_prob = float(noise_prob)
        self.noise_scale = float(noise_scale)
        
        #If using ema act, replace all linear layers with EMA_Linear layers
        if self.ema_act:
            def replace_linear(m):
                for n, c in m.named_children():
                    if isinstance(c, nn.Linear):
                        ema = EMA_Linear(c.in_features, c.out_features)
                        ema.base_linear.load_state_dict(c.state_dict())
                        setattr(m, n, ema)
                    else:
                        replace_linear(c)
            replace_linear(self)
            
            def reset_ema_buffers(m):
                for _, c in m.named_children():
                    if isinstance(c, EMA_Linear):
                        c.reset_state()
                    else:
                        reset_ema_buffers(c)
            self.reset = reset_ema_buffers
        
    def _single_iter_compilable(self, initial_thought, interim_thought):
        #We deal with spatial problems on the inside. Model receives interim_thought and initial_thought as [B, ..., D]
            prev_interim = interim_thought
            
            #Add noise
            if self.noise_prob > 0.0 and torch.rand((), device=interim_thought.device) < self.noise_prob:
                scale = self.noise_scale * (interim_thought.detach().std() + 1e-6)
                interim_thought = interim_thought + torch.randn_like(interim_thought) * scale
            
            #CORE BLOCKS
            for block in self.recur_blocks:
                interim_thought, ccot_tokens = block(interim_thought, initial_thought, ccot_tokens)

            #Velocity designed to avoid early convergence / falling into local minima  
            if self.velocity > 0:
                velocity = self.velocity * velocity + (interim_thought - prev_interim)
                interim_thought = prev_interim + velocity
            
            return interim_thought
    
    def forward(self, x, iters_to_do, interim_thought=None, return_all = False, **kwargs):
        ccot_tokens = self.ccot_tokens.unsqueeze(0).repeat(x.shape[0], 1, 1) #[B, I, D]
        
        # Normalize input to (B, *spatial_dims, C)
        if x.dim() >= 2 and x.size(1) == self.in_channels:
            # (B, C, *spatial_dims) -> (B, *spatial_dims, C)
            x = x.permute(0, *range(2, x.dim()), 1)
        elif x.dim() == 2:
            # (B, L) -> (B, L, 1)
            x = x.unsqueeze(-1)

        initial_thought = self.projection(x)
        initial_thought = self.init_norm(initial_thought)

        if interim_thought is None:
            interim_thought = initial_thought
        elif interim_thought.dim() >= 3 and interim_thought.size(1) == self.hidden_dim:
            interim_thought = interim_thought.permute(0, *range(2, interim_thought.dim()), 1)
        
        if self.is_mhc and interim_thought.dim() >= 3:
            interim_thought = interim_thought.unsqueeze(0).repeat(self.lanes, *([1] * interim_thought.dim()))
        
        # Infer spatial dimensions from x: x is (B, *spatial_dims, C) after transformation
        spatial_dims = x.shape[1:-1]  # All dims except batch and channel
        needs_all_outputs = return_all or not self.training
        if needs_all_outputs:
            all_outputs = torch.empty((x.size(0), iters_to_do, self.out_channels, *spatial_dims),
                                      device=x.device, dtype=initial_thought.dtype)
        track_norm_ratio = getattr(self, "_compute_h_norm_ratio", False)
        track_convergence = getattr(self, "_compute_convergence", False) and not self.compile
        
        if track_norm_ratio:
            first_h_norm = None
            last_h_norm = None
        
        if track_convergence:
            self._first_convergence_iter = iters_to_do
        
        penult_interim = None
        prev_interim = None
        velocity = torch.zeros_like(interim_thought) if self.velocity > 0 else None
        for i in range(iters_to_do):
            if self.compile:
                interim_thought = self._single_iter_compilable(initial_thought, interim_thought)
                if track_norm_ratio and (i == 0 or i == iters_to_do - 1):
                    h_flat = (interim_thought.mean(dim=0) if self.is_mhc else interim_thought).detach().to(torch.float32)
                    h_flat = h_flat.reshape(interim_thought.size(1) if self.is_mhc else interim_thought.size(0), -1)
                    h_norm = h_flat.norm(dim=-1).mean().item()
                    if i == 0:
                        first_h_norm = h_norm
                    if i == iters_to_do - 1:
                        last_h_norm = h_norm
                head_input = torch.einsum('k,kbld->bld', F.softmax(self.lane_combine, dim=0), interim_thought) if self.is_mhc else interim_thought
                out = self.head(head_input)
                # out is (B, *spatial_dims, out_channels), need (B, out_channels, *spatial_dims) for all_outputs
                out = out.permute(0, -1, *range(1, out.dim() - 1))
                if needs_all_outputs:
                    all_outputs[:, i] = out
                continue
            
            #We deal with spatial problems on the inside. Model receives interim_thought and initial_thought as [B, ..., D]
            prev_interim = interim_thought
            
            #Add noise
            if self.noise_prob > 0.0 and torch.rand((), device=interim_thought.device) < self.noise_prob:
                scale = self.noise_scale * (interim_thought.detach().std() + 1e-6)
                interim_thought = interim_thought + torch.randn_like(interim_thought) * scale
            
            #CORE BLOCKS
            for block in self.recur_blocks:
                interim_thought, ccot_tokens = block(interim_thought, initial_thought, ccot_tokens)
                if self.ccot == 'iterative' and i % 5 == 0: #Only add iterations every five
                    ccot_tokens = torch.cat([ccot_tokens, self.get_new_ccot(x.shape[0])], dim = 1)

            #Velocity designed to avoid early convergence / falling into local minima  
            if self.velocity > 0:
                velocity = self.velocity * velocity + (interim_thought - prev_interim)
                interim_thought = prev_interim + velocity
            
                
            #Track convergence
            if track_convergence:
                if i == iters_to_do - 2:
                    penult_interim = interim_thought
                if prev_interim is not None and self._first_convergence_iter == iters_to_do:
                    prev_flat = (prev_interim.mean(dim=0) if self.is_mhc else prev_interim).reshape(prev_interim.size(1) if self.is_mhc else prev_interim.size(0), -1)
                    curr_flat = (interim_thought.mean(dim=0) if self.is_mhc else interim_thought).reshape(interim_thought.size(1) if self.is_mhc else interim_thought.size(0), -1)
                    cos_sim = F.cosine_similarity(prev_flat, curr_flat, dim=-1).mean().item()
                    if cos_sim > 0.99:
                        self._first_convergence_iter = i
                
            head_input = torch.einsum('k,kbld->bld', F.softmax(self.lane_combine, dim=0), interim_thought) if self.is_mhc else interim_thought
            out = self.head(head_input)
            # out is (B, *spatial_dims, out_channels), need (B, out_channels, *spatial_dims) for all_outputs
            out = out.permute(0, -1, *range(1, out.dim() - 1))
            if needs_all_outputs:
                all_outputs[:, i] = out
            if track_norm_ratio and (i == 0 or i == iters_to_do - 1):
                h_flat = (interim_thought.mean(dim=0) if self.is_mhc else interim_thought).detach().to(torch.float32)
                h_flat = h_flat.reshape(interim_thought.size(1) if self.is_mhc else interim_thought.size(0), -1)
                h_norm = h_flat.norm(dim=-1).mean().item()
                if i == 0:
                    first_h_norm = h_norm
                if i == iters_to_do - 1:
                    last_h_norm = h_norm

        #Resets state each forward step if using EMA on the activations
        if self.ema_act:
            self.reset(self)
            
        if self.training:
            if not return_all:
                return out, interim_thought
            else:
                return all_outputs
            
        if track_norm_ratio:
            if first_h_norm is not None and last_h_norm is not None and first_h_norm != 0:
                self._last_h_norm_ratio = last_h_norm / first_h_norm
            else:
                self._last_h_norm_ratio = 0.0
        if track_convergence:
            if penult_interim is not None:
                penult_flat = (penult_interim.mean(dim=0) if self.is_mhc else penult_interim).reshape(penult_interim.size(1) if self.is_mhc else penult_interim.size(0), -1)
                final_flat = (interim_thought.mean(dim=0) if self.is_mhc else interim_thought).reshape(interim_thought.size(1) if self.is_mhc else interim_thought.size(0), -1)
                self._convergence_cosine = F.cosine_similarity(penult_flat, final_flat, dim=-1).mean().item()
            else:
                self._convergence_cosine = 0.0
            
        return all_outputs


def dt_transformer(width, num_blocks=1, injection_type='concat', norm_type='peri', 
                   norm_before_head=True, recall_inner=False, attn_type='full', num_sinks=0, **kwargs):
    return DTTransformer(AttentionBlock, width, num_blocks, 
                         injection_type=injection_type, norm_type=norm_type,
                         norm_before_head=norm_before_head, recall_inner=recall_inner, 
                         attn_type=attn_type, num_sinks=num_sinks, **kwargs)
