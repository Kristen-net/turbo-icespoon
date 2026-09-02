"""
Pure PyTorch fallback for selective_scan_fn and selective_scan_ref.
Replaces the CUDA-optimized mamba_ssm ops with a pure PyTorch implementation.
Works on Windows without nvcc/CUDA toolkit compilation.

Supports the full Mamba/SS2D calling convention including:
- D (skip connection)
- return_last_state
- K-direction (4-directional scan for SS2D)
"""
import torch
import torch.nn.functional as F
import math


def selective_scan_ref(u, delta, A, B, C, D=None, z=None,
                       delta_softplus=False, delta_bias=None,
                       return_last_state=False):
    """
    Reference implementation of selective scan in pure PyTorch.

    Handles both standard Mamba (B,C 3D) and SS2D (B,C 4D with K directions).

    Args:
        u: (batch, d_inner, length) or (batch, K*d_inner, length) input sequence
        delta: (batch, d_inner, length) or (batch, K*d_inner, length) time step
        A: (d_inner, d_state) or (K*d_inner, d_state) state matrix
        B: (batch, d_state, length) or (batch, K, d_state, length) input matrix
        C: (batch, d_state, length) or (batch, K, d_state, length) output matrix
        D: (d_inner,) or (K*d_inner,) skip connection parameter
        z: (batch, d_inner, length) or (batch, K*d_inner, length) gate
        delta_softplus: whether to apply softplus to delta
        delta_bias: (d_inner,) or (K*d_inner,) bias for delta
        return_last_state: whether to return the last hidden state

    Returns:
        output: (batch, d_inner, length) or (batch, K*d_inner, length)
        last_state: tuple if return_last_state
    """
    batch, d_inner_total, length = u.shape

    # Detect K-direction mode (SS2D)
    if B.dim() == 4:
        # SS2D mode: B is (batch, K, d_state, length)
        K = B.shape[1]
        d_state = B.shape[2]
        d_inner = d_inner_total // K

        # Split u, delta into K groups and process each
        outputs = []
        last_states = []

        for k in range(K):
            u_k = u[:, k * d_inner:(k + 1) * d_inner, :]  # (batch, d_inner, length)
            delta_k = delta[:, k * d_inner:(k + 1) * d_inner, :]  # (batch, d_inner, length)

            # A: (K*d_inner, d_state) -> take k-th block
            if A.shape[0] == d_inner_total:
                A_k = A[k * d_inner:(k + 1) * d_inner, :]  # (d_inner, d_state)
            else:
                A_k = A  # (d_inner, d_state)

            B_k = B[:, k, :, :]  # (batch, d_state, length)
            C_k = C[:, k, :, :]  # (batch, d_state, length)

            D_k = None
            if D is not None:
                if D.shape[0] == d_inner_total:
                    D_k = D[k * d_inner:(k + 1) * d_inner]  # (d_inner,)
                else:
                    D_k = D

            delta_bias_k = None
            if delta_bias is not None:
                if delta_bias.shape[0] == d_inner_total:
                    delta_bias_k = delta_bias[k * d_inner:(k + 1) * d_inner]
                else:
                    delta_bias_k = delta_bias

            z_k = None
            if z is not None and z.dim() == 3:
                if z.shape[1] == d_inner_total:
                    z_k = z[:, k * d_inner:(k + 1) * d_inner, :]

            # Process single direction (always get last state for SS2D mode)
            result = _selective_scan_single(
                u_k, delta_k, A_k, B_k, C_k,
                D=D_k, z=z_k,
                delta_softplus=delta_softplus,
                delta_bias=delta_bias_k,
                return_last_state=True
            )
            y_k, h_k = result
            outputs.append(y_k)
            last_states.append(h_k)

        y = torch.cat(outputs, dim=1)  # (batch, K*d_inner, length)

        if return_last_state:
            return y, last_states
        else:
            return y
    else:
        # Standard Mamba mode: B is (batch, d_state, length) or (d_state, length)
        return _selective_scan_single(
            u, delta, A, B, C,
            D=D, z=z,
            delta_softplus=delta_softplus,
            delta_bias=delta_bias,
            return_last_state=return_last_state
        )


def _selective_scan_single(u, delta, A, B, C, D=None, z=None,
                            delta_softplus=False, delta_bias=None,
                            return_last_state=False):
    """Process a single direction of selective scan."""
    batch, d_inner, length = u.shape

    # Ensure B and C are 3D: (batch, d_state, length)
    if B.dim() == 2:
        B = B.unsqueeze(0).expand(batch, -1, -1)
    if C.dim() == 2:
        C = C.unsqueeze(0).expand(batch, -1, -1)

    d_state = B.shape[1]

    # Ensure A is 2D: (d_inner, d_state)
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
    # delta: (batch, d_inner, length), A: (d_inner, d_state)
    delta_expanded = delta.unsqueeze(2)  # (batch, d_inner, 1, length)
    A_expanded = A.unsqueeze(0).unsqueeze(-1)  # (1, d_inner, d_state, 1)
    dA = torch.exp(delta_expanded * A_expanded)  # (batch, d_inner, d_state, length)

    # dB = delta * B
    # B: (batch, d_state, length) -> (batch, 1, d_state, length)
    B_expanded = B.unsqueeze(1)  # (batch, 1, d_state, length)
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
    """
    return selective_scan_ref(
        u, delta, A, B, C,
        D=D, z=z,
        delta_softplus=delta_softplus,
        delta_bias=delta_bias,
        return_last_state=return_last_state,
    )
