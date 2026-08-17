"""
Pure PyTorch fallback for selective_scan_fn and selective_scan_ref.
Replaces the CUDA-optimized mamba_ssm ops with a pure PyTorch implementation.
Works on Windows without nvcc/CUDA toolkit compilation.

Supports the full Mamba/SS2D calling convention including D (skip connection)
and return_last_state parameters.
"""
import torch
import torch.nn.functional as F
import math


def selective_scan_ref(u, delta, A, B, C, D=None, z=None,
                       delta_softplus=False, delta_bias=None,
                       return_last_state=False):
    """
    Reference implementation of selective scan in pure PyTorch.

    Args:
        u: (batch, d_inner, length) input sequence
        delta: (batch, d_inner, length) time step
        A: (d_inner, d_state) state matrix (already negated by caller)
        B: (batch, d_state, length) or (batch, K, d_state, length) input matrix
        C: (batch, d_state, length) or (batch, K, d_state, length) output matrix
        D: (d_inner,) skip connection parameter (optional)
        z: (batch, d_inner, length) gate (optional)
        delta_softplus: whether to apply softplus to delta
        delta_bias: (d_inner,) bias for delta (optional)
        return_last_state: whether to return the last hidden state

    Returns:
        output: (batch, d_inner, length)
        last_state: (batch, d_inner, d_state) if return_last_state, else None
    """
    batch, d_inner, length = u.shape

    # Handle B and C with optional K dimension
    # B can be: (batch, d_state, length) or (batch, K, d_state, length)
    if B.dim() == 4:
        # B: (batch, K, d_state, length) -> reshape to (batch, K*d_state, length)
        B = B.view(batch, -1, length)
    elif B.dim() == 2:
        B = B.unsqueeze(0).expand(batch, -1, -1)

    if C.dim() == 4:
        C = C.view(batch, -1, length)
    elif C.dim() == 2:
        C = C.unsqueeze(0).expand(batch, -1, -1)

    d_state = B.shape[1]

    # Handle A: (d_inner, d_state) or (K*d_inner, d_state)
    if A.dim() == 1:
        A = A.view(-1, d_state)

    # Apply delta bias
    if delta_bias is not None:
        if delta_bias.dim() == 1:
            delta = delta + delta_bias.unsqueeze(0).unsqueeze(-1)
        else:
            delta = delta + delta_bias

    # Apply softplus to delta
    if delta_softplus:
        delta = F.softplus(delta)

    # Discretize: dA = exp(delta * A)
    delta_expanded = delta.unsqueeze(2)  # (batch, d_inner, 1, length)
    A_expanded = A.unsqueeze(0).unsqueeze(-1)  # (1, d_inner, d_state, 1)
    dA = torch.exp(delta_expanded * A_expanded)  # (batch, d_inner, d_state, length)

    # dB = delta * B
    B_expanded = B.unsqueeze(1) if B.dim() == 3 else B  # (batch, 1, d_state, length)
    dB = delta.unsqueeze(2) * B_expanded  # (batch, d_inner, d_state, length)

    # dB_u = dB * u
    u_expanded = u.unsqueeze(2)  # (batch, d_inner, 1, length)
    dB_u = dB * u_expanded  # (batch, d_inner, d_state, length)

    # Sequential scan (loop over time)
    h = torch.zeros(batch, d_inner, d_state, device=u.device, dtype=u.dtype)
    ys = []

    for t in range(length):
        h = dA[:, :, :, t] * h + dB_u[:, :, :, t]
        # C_t: (batch, d_state) -> expand to (batch, d_inner, d_state)
        C_t = C[:, :, t].unsqueeze(1).expand(batch, d_inner, -1)
        y_t = (h * C_t).sum(dim=2)  # (batch, d_inner)
        ys.append(y_t)

    y = torch.stack(ys, dim=-1)  # (batch, d_inner, length)

    # Add skip connection D * u
    if D is not None:
        if D.dim() == 1:
            D_expanded = D.unsqueeze(0).unsqueeze(-1)  # (1, d_inner, 1)
        else:
            D_expanded = D
        y = y + D_expanded * u

    # Apply gate z
    if z is not None:
        y = y * z

    if return_last_state:
        return y, h
    else:
        return y


def selective_scan_fn(u, delta, A, B, C, D=None, z=None,
                      delta_softplus=False, delta_bias=None,
                      return_last_state=False):
    """
    Drop-in replacement for the CUDA selective_scan_fn.
    Falls back to the pure PyTorch reference implementation.

    Full interface compatible with Mamba's selective_scan_fn and SS2D's calling convention.
    """
    return selective_scan_ref(
        u, delta, A, B, C,
        D=D,
        z=z,
        delta_softplus=delta_softplus,
        delta_bias=delta_bias,
        return_last_state=return_last_state,
    )
