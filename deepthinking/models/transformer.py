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

from .blocks import AttentionBlock1D as AttentionBlock, ConcatAttentionBlock, mHCAttentionBlock, add_monotone_hook

class DTTransformer(nn.Module):
    """DeepThinking Transformer model class"""

    def __init__(self, block, hidden_dim, num_blocks, 
                 injection_type, norm_type, norm_before_head,
                 recall_inner, spectral = False, 
                 hidden_dropout = 0, qk_normalization = False,
                 monotone_lambda = 0.0, monotone_margin = 0.0, 
                 post_relu = False, full_concat = False,
                 residual_method = 'add', lanes = 3, attn_type='full',
                 in_channels: int = 1, max_seq_len=None, num_sinks=0,
                 noise_prob: float = 0.0, noise_scale: float = 0.01, **kwargs):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_blocks = num_blocks
        self.spectral = spectral
        
        #Allows to alternate injection (e.g. injection every two blocks, as in huginn)
        if residual_method == 'mhc':
            self.recur_blocks_inject = nn.ModuleList([
                mHCAttentionBlock(
                    hidden_dim,
                    norm_type,
                    qk_normalization=qk_normalization,
                    lanes=lanes,
                    attn_type=attn_type,
                    max_seq_len=max_seq_len,
                    num_sinks=num_sinks,
                )
                for _ in range(num_blocks)
            ])
            self.recur_blocks_no_inject = nn.ModuleList([])
            self.lanes = lanes
            self.lane_combine = nn.Parameter(torch.ones(self.lanes))
        elif full_concat:
            self.recur_blocks_inject = nn.ModuleList([
                ConcatAttentionBlock(
                    hidden_dim,
                    norm_type,
                    qk_normalization=qk_normalization,
                    residual_method=residual_method,
                    attn_type=attn_type,
                    max_seq_len=max_seq_len,
                    num_sinks=num_sinks,
                )
                for _ in range(num_blocks)
            ])
            self.recur_blocks_no_inject = nn.ModuleList([])
        else:
            self.recur_blocks_inject = nn.ModuleList([AttentionBlock(hidden_dim, injection_type, norm_type, recall_inner, 
                                                    spectral, qk_normalization=qk_normalization, post_relu = post_relu,
                                                    residual_method=residual_method, attn_type=attn_type,
                                                    max_seq_len=max_seq_len, num_sinks=num_sinks)])
            self.recur_blocks_no_inject = nn.ModuleList([AttentionBlock(hidden_dim, 'none',
                                                            norm_type, recall_inner, qk_normalization=qk_normalization,
                                                            residual_method=residual_method, attn_type=attn_type,
                                                            max_seq_len=max_seq_len, num_sinks=num_sinks) for _ in range(num_blocks - 1)])
            
        #self.init_norm = nn.RMSNorm(hidden_dim)
        self.hidden_dropout = nn.Dropout(hidden_dropout)
        self.in_channels = int(in_channels)
        self.projection = nn.Linear(self.in_channels, hidden_dim)
        self.monotone_lambda = float(monotone_lambda)
        self.monotone_margin = float(monotone_margin)
        self.is_mhc = (residual_method == 'mhc')
        self.noise_prob = float(noise_prob)
        self.noise_scale = float(noise_scale)
        self.head = nn.Sequential(
            nn.RMSNorm(hidden_dim) if norm_before_head else nn.Identity(),
            nn.Linear(hidden_dim, hidden_dim), 
            nn.GELU(), 
            nn.Linear(hidden_dim, 2)
        )
        
    def forward(self, x, iters_to_do, interim_thought=None, return_all = False, noise_prob=None, **kwargs):
        spatial_shape = None
        if x.dim() == 4:
            # (B, C, H, W) -> (B, H*W, C)
            spatial_shape = (x.size(2), x.size(3))
            x = x.permute(0, 2, 3, 1).reshape(x.size(0), -1, x.size(1))
        elif x.dim() == 3 and x.size(1) == self.in_channels:
            # (B, C, L) -> (B, L, C)
            x = x.transpose(1, 2)
        elif x.dim() == 2:
            # (B, L) -> (B, 1, L) -> (B, L, 1) for 1D sequences without channel dimension
            x = x.unsqueeze(1).transpose(1, 2)

        initial_thought = self.projection(x)
        #initial_thought = self.init_norm(initial_thought)

        if interim_thought is None:
            interim_thought = initial_thought
        elif interim_thought.dim() == 3 and interim_thought.size(1) == self.hidden_dim:
            interim_thought = interim_thought.transpose(1, 2)
        
        if self.is_mhc and interim_thought.dim() == 3:
            interim_thought = interim_thought.unsqueeze(0).repeat(self.lanes, 1, 1, 1)
            
        if spatial_shape is None:
            all_outputs = torch.zeros((x.size(0), iters_to_do, 2, x.size(1))).to(x.device)
        else:
            all_outputs = torch.zeros((x.size(0), iters_to_do, 2, spatial_shape[0], spatial_shape[1])).to(x.device)
        track_norm_ratio = getattr(self, "_compute_h_norm_ratio", False)
        track_convergence = getattr(self, "_compute_convergence", False)
        
        if track_norm_ratio:
            h_norms = []
        
        if track_convergence:
            self._first_convergence_iter = iters_to_do
        
        penult_interim = None
        prev_interim = None
        noise_prob = self.noise_prob if noise_prob is None else float(noise_prob)
        for i in range(iters_to_do):
            prev_interim = interim_thought
            interim_thought = self.hidden_dropout(interim_thought)
            if noise_prob > 0.0 and torch.rand((), device=interim_thought.device) < noise_prob:
                scale = self.noise_scale * (interim_thought.detach().std() + 1e-6)
                interim_thought = interim_thought + torch.randn_like(interim_thought) * scale
            for block in self.recur_blocks_inject:
                interim_thought = block(interim_thought, initial_thought)
            for block in self.recur_blocks_no_inject:                    
                interim_thought = block(interim_thought, initial_thought)
            if (self.training and self.monotone_lambda > 0.0 and prev_interim is not None
                    and i == iters_to_do - 1):
                add_monotone_hook(
                    interim_thought,
                    prev_interim,
                    lam=self.monotone_lambda,
                    margin=self.monotone_margin,
                    reduce_over_tokens=True,
                )
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
            if spatial_shape is None:
                all_outputs[:, i] = out.transpose(1, 2)
            else:
                all_outputs[:, i] = out.transpose(1, 2).reshape(x.size(0), 2, spatial_shape[0], spatial_shape[1])
            if track_norm_ratio:
                h_flat = (interim_thought.mean(dim=0) if self.is_mhc else interim_thought).detach().to(torch.float32).reshape(interim_thought.size(1) if self.is_mhc else interim_thought.size(0), -1)
                h_norms.append(h_flat.norm(dim=-1).mean().item())

        if self.training:
            if not return_all:
                if spatial_shape is None:
                    return out.transpose(1, 2), interim_thought
                return out.transpose(1, 2).reshape(x.size(0), 2, spatial_shape[0], spatial_shape[1]), interim_thought
            else:
                return all_outputs
            
        if track_norm_ratio:
            if len(h_norms) >= 2 and h_norms[0] != 0:
                self._last_h_norm_ratio = h_norms[-1] / h_norms[0]
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
