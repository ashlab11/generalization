"""
Taken from limefax on github, https://github.com/limefax/rope-nd
"""
import torch

class RoPENd(torch.nn.Module):
    """N-dimensional Rotary Positional Embedding."""
    def __init__(self, shape, base=10000, device=None):
        super(RoPENd, self).__init__()

        channel_dims, feature_dim = shape[:-1], shape[-1]
        k_max = feature_dim // (2 * len(channel_dims))

        assert feature_dim % k_max == 0, f'shape[-1] ({feature_dim}) is not divisible by 2 * len(shape[:-1]) ({2 * len(channel_dims)})'

        # tensor of angles to use
        theta_ks = 1 / (base ** (torch.arange(k_max, device=device) / k_max))

        # create a stack of angles multiplied by position
        angles = torch.cat([t.unsqueeze(-1) * theta_ks for t in
                            torch.meshgrid([torch.arange(d, device=device) for d in channel_dims], indexing='ij')], dim=-1)

        # convert to complex number to allow easy rotation
        rotations = torch.polar(torch.ones_like(angles), angles)

        # store in a buffer so it can be saved in model parameters
        self.register_buffer('rotations', rotations)

    def forward(self, x):
        # Apply RoPE in real space to avoid large float32 temporary allocations.
        x = x.reshape(*x.shape[:-1], -1, 2)
        x_real, x_imag = x[..., 0], x[..., 1]
        rot_real = self.rotations.real
        rot_imag = self.rotations.imag
        out_real = x_real * rot_real - x_imag * rot_imag
        out_imag = x_real * rot_imag + x_imag * rot_real
        return torch.stack((out_real, out_imag), dim=-1).flatten(-2)
