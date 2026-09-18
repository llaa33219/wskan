"""Triton two-kernel segmented scan for the V8 recurrence.

STATUS: verified correct (<=3e-6 vs serial loop, fwd and all gradients) but
NOT adopted - slower than the torch Hillis-Steele path in V8_WSKAN.py for
this project's shapes (scalar per-element access is latency-bound at our
lane counts). Kept as a reference implementation and a starting point for a
future Mamba-class fused kernel.

Layout: (B, I, N, L) contiguous, t fastest. Lanes: (b, i, n, segment).
Kernel A: per-segment serial scan with zero init; stores local h and the
  per-position within-segment prefix decay A_prefix (complex), plus segment
  end state and end prefix.
Kernel B: per (b, i, n) serial carry fixup across segments:
  h_final[t] = h_local[t] + carry * A_prefix[t]; carry = carry*A_end + h_end.

Backward mirrors this with the conjugated multiplier (transpose of the
forward complex multiply). All numerics: |a| <= 1, A_prefix decaying -
stable by construction.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _seg_scan_kernel(a_re_ptr, a_im_ptr, u_re_ptr, u_im_ptr,
                     h_re_ptr, h_im_ptr, A_re_ptr, A_im_ptr,
                     hEnd_re_ptr, hEnd_im_ptr, AEnd_re_ptr, AEnd_im_ptr,
                     L, C: tl.constexpr, NSEG: tl.constexpr):
    pid = tl.program_id(0)
    seg = pid % NSEG
    lane = pid // NSEG
    base = lane * L
    h_re = 0.0
    h_im = 0.0
    p_re = 1.0
    p_im = 0.0
    for j in range(C):
        t = seg * C + j
        ar = tl.load(a_re_ptr + base + t)
        ai = tl.load(a_im_ptr + base + t)
        pr = p_re * ar - p_im * ai
        pi = p_re * ai + p_im * ar
        p_re, p_im = pr, pi
        ur = tl.load(u_re_ptr + base + t)
        ui = tl.load(u_im_ptr + base + t)
        nhr = ar * h_re - ai * h_im + ur
        nhi = ar * h_im + ai * h_re + ui
        h_re, h_im = nhr, nhi
        tl.store(h_re_ptr + base + t, h_re)
        tl.store(h_im_ptr + base + t, h_im)
        tl.store(A_re_ptr + base + t, p_re)
        tl.store(A_im_ptr + base + t, p_im)
    tl.store(hEnd_re_ptr + pid, h_re)
    tl.store(hEnd_im_ptr + pid, h_im)
    tl.store(AEnd_re_ptr + pid, p_re)
    tl.store(AEnd_im_ptr + pid, p_im)


@triton.jit
def _carry_kernel(h_re_ptr, h_im_ptr, A_re_ptr, A_im_ptr,
                  hEnd_re_ptr, hEnd_im_ptr, AEnd_re_ptr, AEnd_im_ptr,
                  L, C: tl.constexpr, NSEG: tl.constexpr):
    pid = tl.program_id(0)
    base = pid * L
    c_re = 0.0
    c_im = 0.0
    for s in range(NSEG):
        ep = pid * NSEG + s
        he_re = tl.load(hEnd_re_ptr + ep)
        he_im = tl.load(hEnd_im_ptr + ep)
        Ae_re = tl.load(AEnd_re_ptr + ep)
        Ae_im = tl.load(AEnd_im_ptr + ep)
        for j in range(C):
            t = s * C + j
            Ar = tl.load(A_re_ptr + base + t)
            Ai = tl.load(A_im_ptr + base + t)
            hr = tl.load(h_re_ptr + base + t) + c_re * Ar - c_im * Ai
            hi = tl.load(h_im_ptr + base + t) + c_re * Ai + c_im * Ar
            tl.store(h_re_ptr + base + t, hr)
            tl.store(h_im_ptr + base + t, hi)
        nc_re = c_re * Ae_re - c_im * Ae_im + he_re
        nc_im = c_re * Ae_im + c_im * Ae_re + he_im
        c_re, c_im = nc_re, nc_im


@triton.jit
def _seg_scan_bwd_kernel(a_re_ptr, a_im_ptr, G_re_ptr, G_im_ptr,
                         D_re_ptr, D_im_ptr, Q_re_ptr, Q_im_ptr,
                         DEnd_re_ptr, DEnd_im_ptr, QEnd_re_ptr, QEnd_im_ptr,
                         L, L_real, C: tl.constexpr, NSEG: tl.constexpr):
    pid = tl.program_id(0)
    seg = pid % NSEG
    lane = pid // NSEG
    base = lane * L
    D_re = 0.0
    D_im = 0.0
    q_re = 1.0
    q_im = 0.0
    for j in range(C):
        t = seg * C + (C - 1 - j)
        nxt = t + 1
        in_seg = nxt < (seg + 1) * C
        valid = in_seg & (nxt < L_real)
        ar = tl.load(a_re_ptr + base + nxt, mask=valid, other=1.0)
        ai = tl.load(a_im_ptr + base + nxt, mask=valid, other=0.0)
        qr = q_re * ar + q_im * ai
        qi = q_im * ar - q_re * ai
        q_re, q_im = qr, qi
        Gr = tl.load(G_re_ptr + base + t)
        Gi = tl.load(G_im_ptr + base + t)
        nDr = Gr + ar * D_re + ai * D_im
        nDi = Gi + ar * D_im - ai * D_re
        D_re, D_im = nDr, nDi
        tl.store(D_re_ptr + base + t, D_re)
        tl.store(D_im_ptr + base + t, D_im)
        tl.store(Q_re_ptr + base + t, q_re)
        tl.store(Q_im_ptr + base + t, q_im)
    tl.store(DEnd_re_ptr + pid, D_re)
    tl.store(DEnd_im_ptr + pid, D_im)
    tl.store(QEnd_re_ptr + pid, q_re)
    tl.store(QEnd_im_ptr + pid, q_im)


@triton.jit
def _carry_bwd_kernel(a_re_ptr, a_im_ptr,
                      D_re_ptr, D_im_ptr, Q_re_ptr, Q_im_ptr,
                      DEnd_re_ptr, DEnd_im_ptr, QEnd_re_ptr, QEnd_im_ptr,
                      L, C: tl.constexpr, NSEG: tl.constexpr):
    pid = tl.program_id(0)
    base = pid * L
    c_re = 0.0
    c_im = 0.0
    for s_rev in range(NSEG):
        s = NSEG - 1 - s_rev
        ep = pid * NSEG + s
        De_re = tl.load(DEnd_re_ptr + ep)
        De_im = tl.load(DEnd_im_ptr + ep)
        Qe_re = tl.load(QEnd_re_ptr + ep)
        Qe_im = tl.load(QEnd_im_ptr + ep)
        # boundary factor: conj(a at the next segment's first position)
        bt = (s + 1) * C
        has_next = (s + 1) < NSEG
        abr = tl.load(a_re_ptr + base + bt, mask=has_next, other=0.0)
        abi = tl.load(a_im_ptr + base + bt, mask=has_next, other=0.0)
        cc_re = c_re * abr + c_im * abi
        cc_im = c_im * abr - c_re * abi
        for j in range(C):
            t = s * C + j
            Qr = tl.load(Q_re_ptr + base + t)
            Qi = tl.load(Q_im_ptr + base + t)
            Dr = tl.load(D_re_ptr + base + t) + cc_re * Qr - cc_im * Qi
            Di = tl.load(D_im_ptr + base + t) + cc_re * Qi + cc_im * Qr
            tl.store(D_re_ptr + base + t, Dr)
            tl.store(D_im_ptr + base + t, Di)
        nc_re = De_re + Qe_re * cc_re - Qe_im * cc_im
        nc_im = De_im + Qe_re * cc_im + Qe_im * cc_re
        c_re, c_im = nc_re, nc_im


def _n2(N: int) -> int:
    return 1 << (N - 1).bit_length()


def _padN(t, N2):
    if t.shape[-2] == N2:
        return t.contiguous()
    n = t.shape[-2]
    return torch.nn.functional.pad(t, (0, 0, 0, N2 - n)).contiguous()


def _pick_C(L: int) -> int:
    return 1 << max(4, (L.bit_length() - 3))


def scan_fwd(a_re, a_im, u_re, u_im):
    B, I, N, L = u_re.shape
    N2 = _n2(N)
    a_re = _padN(a_re, N2); a_im = _padN(a_im, N2)
    u_re = _padN(u_re, N2); u_im = _padN(u_im, N2)
    C = min(_pick_C(L), L)
    NSEG = (L + C - 1) // C
    Lp = NSEG * C
    def padL(t):
        if Lp == L:
            return t.contiguous()
        return torch.nn.functional.pad(t, (0, Lp - L)).contiguous()
    a_re, a_im, u_re, u_im = padL(a_re), padL(a_im), padL(u_re), padL(u_im)
    h_re = torch.empty_like(u_re); h_im = torch.empty_like(u_im)
    A_re = torch.empty_like(u_re); A_im = torch.empty_like(u_im)
    lanes = B * I * N2
    hEnd_re = torch.empty(lanes * NSEG, device=u_re.device); hEnd_im = torch.empty_like(hEnd_re)
    AEnd_re = torch.empty_like(hEnd_re); AEnd_im = torch.empty_like(hEnd_re)
    _seg_scan_kernel[(lanes * NSEG,)](a_re, a_im, u_re, u_im, h_re, h_im,
                                      A_re, A_im, hEnd_re, hEnd_im, AEnd_re, AEnd_im,
                                      Lp, C=C, NSEG=NSEG)
    _carry_kernel[(lanes,)](h_re, h_im, A_re, A_im, hEnd_re, hEnd_im, AEnd_re, AEnd_im,
                            Lp, C=C, NSEG=NSEG)
    return h_re[..., :N, :L], h_im[..., :N, :L]


def scan_bwd(a_re, a_im, h_re, h_im, G_re, G_im):
    B, I, N, L = h_re.shape
    N2 = _n2(N)
    a_re = _padN(a_re, N2); a_im = _padN(a_im, N2)
    h_re = _padN(h_re, N2); h_im = _padN(h_im, N2)
    G_re = _padN(G_re, N2); G_im = _padN(G_im, N2)
    C = min(_pick_C(L), L)
    NSEG = (L + C - 1) // C
    Lp = NSEG * C
    def padL(t):
        if Lp == L:
            return t.contiguous()
        return torch.nn.functional.pad(t, (0, Lp - L)).contiguous()
    a_re, a_im, G_re, G_im, h_re, h_im = padL(a_re), padL(a_im), padL(G_re), padL(G_im), padL(h_re), padL(h_im)
    D_re = torch.empty_like(G_re); D_im = torch.empty_like(G_im)
    Q_re = torch.empty_like(G_re); Q_im = torch.empty_like(G_im)
    lanes = B * I * N2
    DEnd_re = torch.empty(lanes * NSEG, device=G_re.device); DEnd_im = torch.empty_like(DEnd_re)
    QEnd_re = torch.empty_like(DEnd_re); QEnd_im = torch.empty_like(DEnd_re)
    _seg_scan_bwd_kernel[(lanes * NSEG,)](a_re, a_im, G_re, G_im, D_re, D_im, Q_re, Q_im,
                                          DEnd_re, DEnd_im, QEnd_re, QEnd_im, Lp, L_real=L, C=C, NSEG=NSEG)
    _carry_bwd_kernel[(lanes,)](a_re, a_im, D_re, D_im, Q_re, Q_im, DEnd_re, DEnd_im, QEnd_re, QEnd_im,
                                Lp, C=C, NSEG=NSEG)
    du_re = D_re[..., :N, :L]; du_im = D_im[..., :N, :L]
    h_prev_re = torch.nn.functional.pad(h_re[..., :N, :L - 1], (1, 0))  # h_prev[t] = h[t-1]
    h_prev_im = torch.nn.functional.pad(h_im[..., :N, :L - 1], (1, 0))
    da_re = du_re * h_prev_re + du_im * h_prev_im
    da_im = du_im * h_prev_re - du_re * h_prev_im
    return da_re, da_im, du_re, du_im
