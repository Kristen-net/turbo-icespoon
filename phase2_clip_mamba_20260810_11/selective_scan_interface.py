"""
Pure PyTorch fallback for selective_scan_fn and selective_scan_ref.
Replaces the CUDA-optimized mamba_ssm ops with a pure PyTorch implementation.
Works on Windows without nvcc/CUDA toolkit compilation.
"""
import torch
import torch.nn.functional as F
import math


def selective_scan_ref(u, delta, A, B, C, z=None, delta_softplus=False, delta_bias=None):
    """
    Reference implementation of selective scan in pure PyTorch.
    
    Args:
        u: (batch, d_inner, length) input sequence
        delta: (batch, d_inner, length) time step
        A: (d_inner, d_state) state matrix (will be negated)
        B: (batch, d_state, length) or (d_state, length) input matrix
        C: (batch, d_state, length) or (d_state, length) output matrix
        z: (batch, d_inner, length) gate (optional)
        delta_softplus: whether to apply softplus to delta
        delta_bias: (d_inner,) bias for delta (optional)
    
    Returns:
        output: (batch, d_inner, length) if z is None, else (batch, d_inner, length)
        last_state: (batch, d_inner, d_state) last hidden state
    """
    batch, d_inner, length = u.shape
    
    if B.dim() == 2:
        B = B.unsqueeze(0).expand(batch, -1, -1)
    if C.dim() == 2:
        C = C.unsqueeze(0).expand(batch, -1, -1)
    
    d_state = B.shape[1]
    
    if delta_bias is not None:
        delta = delta + delta_bias.unsqueeze(0).unsqueeze(-1)
    
    if delta_softplus:
        delta = F.softplus(delta)
    
    # A is (d_inner, d_state), negate it
    A = -A  # (d_inner, d_state)
    
    # Discretize A and B
    # dA = exp(delta * A)  -> (batch, d_inner, d_state, length)
    delta_expanded = delta.unsqueeze(2)  # (batch, d_inner, 1, length)
    A_expanded = A.unsqueeze(0).unsqueeze(-1)  # (1, d_inner, d_state, 1)
    dA = torch.exp(delta_expanded * A_expanded)  # (batch, d_inner, d_state, length)
    
    # dB = delta * B  -> (batch, d_inner, d_state, length)
    B_expanded = B.unsqueeze(1)  # (batch, 1, d_state, length)
    delta_expanded2 = delta.unsqueeze(2)  # (batch, d_inner, 1, length)
    dB = delta_expanded2 * B_expanded  # (batch, d_inner, d_state, length)
    
    # u_expanded: (batch, d_inner, 1, length)
    u_expanded = u.unsqueeze(2)  # (batch, d_inner, 1, length)
    
    # dB_u = dB * u  -> (batch, d_inner, d_state, length)
    dB_u = dB * u_expanded  # (batch, d_inner, d_state, length)
    
    # Sequential scan (loop over time)
    h = torch.zeros(batch, d_inner, d_state, device=u.device, dtype=u.dtype)
    ys = []
    
    for t in range(length):
        h = dA[:, :, :, t] * h + dB_u[:, :, :, t]
        y_t = torch.einsum('bid,bid->bi', h, C[:, :, t].permute(0, 1, 0).reshape(batch, d_state, 1).expand(batch, d_inner, d_state))
        # Actually: y_t = sum over d_state of h * C
        # h: (batch, d_inner, d_state)
        # C_t: (batch, d_state) -> need to expand to (batch, d_inner, d_state)
        C_t = C[:, :, t].unsqueeze(1).expand(batch, d_inner, d_state)  # (batch, d_inner, d_state)
        y_t = (h * C_t).sum(dim=2)  # (batch, d_inner)
        ys.append(y_t)
    
    y = torch.stack(ys, dim=-1)  # (batch, d_inner, length)
    
    if z is not None:
        y = y * z
    
    return y, h


def selective_scan_fn(u, delta, A, B, C, z=None, delta_softplus=False, delta_bias=None):
    """
    Drop-in replacement for the CUDA selective_scan_fn.
    Falls back to the pure PyTorch reference implementation.
    
    Same interface as the original CUDA function for compatibility.
    """
    return selective_scan_ref(
        u, delta, A, B, C, 
        z=z, 
        delta_softplus=delta_softplus, 
        delta_bias=delta_bias
    )
