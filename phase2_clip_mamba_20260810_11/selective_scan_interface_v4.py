"""
Pure PyTorch fallback for selective_scan_fn and selective_scan_ref.
v4: Optimized with chunked cumulative computation.
- Uses float32 with NaN detection and fallback to sequential scan
- Chunk size 128 for balance of speed and stability
- SS2D K-direction (4-directional scan) support
"""
import torch
import torch.nn.functional as F
import math


def selective_scan_ref(u, delta, A, B, C, D=None, z=None,
                       delta_softplus=False, delta_bias=None,
                       return_last_state=False):
    """Reference implementation of selective scan in pure PyTorch."""
    batch, d_inner_total, length = u.shape

    if B.dim() == 4:
        K = B.shape[1]
        d_state = B.shape[2]
        d_inner = d_inner_total // K

        outputs = []
        last_states = []

        for k in range(K):
            u_k = u[:, k * d_inner:(k + 1) * d_inner, :]
            delta_k = delta[:, k * d_inner:(k + 1) * d_inner, :]

            if A.shape[0] == d_inner_total:
                A_k = A[k * d_inner:(k + 1) * d_inner, :]
            else:
                A_k = A

            B_k = B[:, k, :, :]
            C_k = C[:, k, :, :]

            D_k = None
            if D is not None:
                if D.shape[0] == d_inner_total:
                    D_k = D[k * d_inner:(k + 1) * d_inner]
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

        y = torch.cat(outputs, dim=1)

        if return_last_state:
            return y, last_states
        else:
            return y
    else:
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
    """Process a single direction of selective scan with chunked cumulative approach."""
    batch, d_inner, length = u.shape

    if B.dim() == 2:
        B = B.unsqueeze(0).expand(batch, -1, -1)
    if C.dim() == 2:
        C = C.unsqueeze(0).expand(batch, -1, -1)

    d_state = B.shape[1]

    if A.dim() == 1:
        A = A.view(-1, d_state)

    # Apply delta bias
    if delta_bias is not None:
        if delta_bias.dim() == 1:
            delta = delta + delta_bias.unsqueeze(0).unsqueeze(-1)
        else:
            delta = delta + delta_bias

    if delta_softplus:
        delta = F.softplus(delta)

    # A is already negated by the caller (SS2D does As = -exp(A_logs))
    # Compute dA = exp(delta * A) for all time steps (vectorized)
    # delta: (batch, d_inner, length), A: (d_inner, d_state)
    delta_A = delta.unsqueeze(2) * A.unsqueeze(0).unsqueeze(-1)
    dA = torch.exp(delta_A)  # (batch, d_inner, d_state, length)

    # Compute dB_u = delta * B * u
    dB = delta.unsqueeze(2) * B.unsqueeze(1)
    dB_u = dB * u.unsqueeze(2)  # (batch, d_inner, d_state, length)

    # Use chunked cumulative scan for efficiency
    # h[t] = dA[t] * h[t-1] + dB_u[t], h[-1] = 0
    # Solution: h[t] = cumdA[t] * sum_{j=0}^{t} (dB_u[j] / cumdA[j])
    CHUNK_SIZE = 128

    h = torch.zeros(batch, d_inner, d_state, device=u.device, dtype=u.dtype)
    ys = torch.empty(batch, d_inner, length, device=u.device, dtype=u.dtype)

    for chunk_start in range(0, length, CHUNK_SIZE):
        chunk_end = min(chunk_start + CHUNK_SIZE, length)
        chunk_len = chunk_end - chunk_start

        dA_chunk = dA[:, :, :, chunk_start:chunk_end]
        dBu_chunk = dB_u[:, :, :, chunk_start:chunk_end]
        C_chunk = C[:, :, chunk_start:chunk_end]

        # Cumulative product via log space for stability
        log_dA_chunk = torch.log(dA_chunk.clamp(min=1e-38))
        cum_log_dA = torch.cumsum(log_dA_chunk, dim=-1)
        cumdA = torch.exp(cum_log_dA)

        # dB_u / cumdA
        dBu_over_cumdA = dBu_chunk / cumdA.clamp(min=1e-38)

        # Cumulative sum
        cum_dBu = torch.cumsum(dBu_over_cumdA, dim=-1)

        # h[t] = cumdA[t] * (h_init + cum_dBu[t])
        h_expanded = h.unsqueeze(-1)
        h_chunk = cumdA * (h_expanded + cum_dBu)

        # Check for NaN/Inf - if found, fall back to sequential scan
        if torch.isnan(h_chunk).any() or torch.isinf(h_chunk).any():
            # Fallback: sequential scan for this chunk
            for t in range(chunk_len):
                h = dA_chunk[:, :, :, t] * h + dBu_chunk[:, :, :, t]
                C_t = C_chunk[:, :, t].unsqueeze(1).expand(batch, d_inner, -1)
                ys[:, :, chunk_start + t] = (h * C_t).sum(dim=2)
        else:
            # Vectorized output computation
            C_expanded = C_chunk.unsqueeze(1).expand(batch, d_inner, d_state, chunk_len)
            y_chunk = (h_chunk * C_expanded).sum(dim=2)
            ys[:, :, chunk_start:chunk_end] = y_chunk
            h = h_chunk[:, :, :, -1].contiguous()

    y = ys

    # Add skip connection D * u
    if D is not None:
        if D.dim() == 1:
            D_expanded = D.unsqueeze(0).unsqueeze(-1)
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
    """Drop-in replacement for the CUDA selective_scan_fn."""
    return selective_scan_ref(
        u, delta, A, B, C,
        D=D, z=z,
        delta_softplus=delta_softplus,
        delta_bias=delta_bias,
        return_last_state=return_last_state,
    )
