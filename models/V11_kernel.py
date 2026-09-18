"""V11 fused scan kernel: one Triton kernel per layer for the recurrence.

h_n = a_n * h_{n-1} + u_n (complex as real pairs), via tl.associative_scan
over a (BL, BN) tile loaded coalesced per (b, i) lane. Layout (B, I, L, N)
with N fastest; the tile is contiguous memory, so loads are coalesced.

Backward uses the conjugated multiplier (transpose of the forward complex
multiply): D_t = G_t + conj(a_{t+1}) * D_{t+1}, computed as the same scan
over time-reversed inputs with the multiplier shifted by one position.
  da_t = D_t * conj(h_{t-1});  du_t = D_t.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _rec_combine(Ar, Ai, Ur, Ui, Br, Bi, Vr, Vi):
    A_re = Ar * Br - Ai * Bi
    A_im = Ar * Bi + Ai * Br
    U_re = Br * Ur - Bi * Ui + Vr
    U_im = Br * Ui + Bi * Ur + Vi
    return A_re, A_im, U_re, U_im


@triton.jit
def _v11_fwd_kernel(a_re_ptr, a_im_ptr, u_re_ptr, u_im_ptr,
                    h_re_ptr, h_im_ptr, L, BL: tl.constexpr, BN: tl.constexpr):
    pid = tl.program_id(0)
    base = pid * L * BN
    offs_l = tl.arange(0, BL)
    offs_n = tl.arange(0, BN)
    ptrs = base + offs_l[:, None] * BN + offs_n[None, :]
    mask = (offs_l[:, None] < L)
    ar = tl.load(a_re_ptr + ptrs, mask=mask, other=1.0)
    ai = tl.load(a_im_ptr + ptrs, mask=mask, other=0.0)
    ur = tl.load(u_re_ptr + ptrs, mask=mask, other=0.0)
    ui = tl.load(u_im_ptr + ptrs, mask=mask, other=0.0)
    out = tl.associative_scan((ar, ai, ur, ui), 0, _rec_combine)
    tl.store(h_re_ptr + ptrs, out[2], mask=mask)
    tl.store(h_im_ptr + ptrs, out[3], mask=mask)


@triton.jit
def _v11_bwd_kernel(a_re_ptr, a_im_ptr, h_re_ptr, h_im_ptr, G_re_ptr, G_im_ptr,
                    da_re_ptr, da_im_ptr, du_re_ptr, du_im_ptr,
                    L, BL: tl.constexpr, BN: tl.constexpr):
    pid = tl.program_id(0)
    base = pid * L * BN
    offs_l = tl.arange(0, BL)
    offs_n = tl.arange(0, BN)
    # reversed index m = L-1-t; G' loaded at L-1-m; multiplier = conj(a_{t+1})
    # loaded at L-m (mask m>=1; m=0 -> multiplier zero since D_{L-1} = G_{L-1})
    mpos = offs_l
    ptrs_g = base + (L - 1 - mpos)[:, None] * BN + offs_n[None, :]
    ptrs_a = base + (L - mpos)[:, None] * BN + offs_n[None, :]
    mask = mpos[:, None] < L
    Gr = tl.load(G_re_ptr + ptrs_g, mask=mask, other=0.0)
    Gi = tl.load(G_im_ptr + ptrs_g, mask=mask, other=0.0)
    amask = mask & (mpos[:, None] >= 1)
    ar = tl.load(a_re_ptr + ptrs_a, mask=amask, other=0.0)
    ai = -tl.load(a_im_ptr + ptrs_a, mask=amask, other=0.0)  # conjugated
    out = tl.associative_scan((ar, ai, Gr, Gi), 0, _rec_combine)
    Dr, Di = out[2], out[3]
    # flip back: D_t stored at L-1-m
    tl.store(du_re_ptr + ptrs_g, Dr, mask=mask)
    tl.store(du_im_ptr + ptrs_g, Di, mask=mask)
    # da_t = D_t * conj(h_{t-1}); h_{t-1} at position t-1 = (L-1-m)-1
    ptrs_hp = base + (L - 2 - mpos)[:, None] * BN + offs_n[None, :]
    hmask = mask & ((L - 2 - mpos)[:, None] >= 0)
    hr = tl.load(h_re_ptr + ptrs_hp, mask=hmask, other=0.0)
    hi = tl.load(h_im_ptr + ptrs_hp, mask=hmask, other=0.0)
    da_re = Dr * hr + Di * hi
    da_im = Di * hr - Dr * hi
    tl.store(da_re_ptr + ptrs_g, da_re, mask=mask)
    tl.store(da_im_ptr + ptrs_g, da_im, mask=mask)


def _pow2(x: int) -> int:
    return max(2, 1 << (x - 1).bit_length())


def v11_scan_fwd(a_re, a_im, u_re, u_im):
    """(B, I, L, N) contiguous -> (B, I, L, N) h."""
    B, I, L, N = u_re.shape
    BL, BN = _pow2(L), _pow2(N)

    def pad(t):
        pl, pn = BL - L, BN - N
        if pl == 0 and pn == 0:
            return t.contiguous()
        return torch.nn.functional.pad(t, (0, pn, 0, pl)).contiguous()

    a_re, a_im, u_re, u_im = pad(a_re), pad(a_im), pad(u_re), pad(u_im)
    Lp, Np = BL, BN
    h_re = torch.empty_like(u_re)
    h_im = torch.empty_like(u_im)
    _v11_fwd_kernel[(B * I,)](a_re, a_im, u_re, u_im, h_re, h_im, Lp, BL=BL, BN=BN)
    return h_re[:, :, :L, :N], h_im[:, :, :L, :N]


def v11_scan_bwd(a_re, a_im, h_re, h_im, G_re, G_im):
    B, I, L, N = G_re.shape
    BL, BN = _pow2(L), _pow2(N)

    def pad(t):
        pl, pn = BL - L, BN - N
        if pl == 0 and pn == 0:
            return t.contiguous()
        return torch.nn.functional.pad(t, (0, pn, 0, pl)).contiguous()

    a_re, a_im = pad(a_re), pad(a_im)
    h_re, h_im = pad(h_re), pad(h_im)
    G_re, G_im = pad(G_re), pad(G_im)
    Lp, Np = BL, BN
    da_re = torch.empty_like(G_re); da_im = torch.empty_like(G_im)
    du_re = torch.empty_like(G_re); du_im = torch.empty_like(G_im)
    _v11_bwd_kernel[(B * I,)](a_re, a_im, h_re, h_im, G_re, G_im,
                              da_re, da_im, du_re, du_im, Lp, BL=BL, BN=BN)
    return (da_re[:, :, :L, :N], da_im[:, :, :L, :N],
            du_re[:, :, :L, :N], du_im[:, :, :L, :N])
