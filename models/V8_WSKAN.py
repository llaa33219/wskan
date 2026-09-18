"""V8 Wavelet-State-KAN: identical math to V6/V7bc, GPU-native implementation.

The V3-V7 chunked SSD scan is compute-correct but launch-bound: hundreds of
small kernel launches per layer per step leave the GPU idle (measured: 0.7
s/step at the 1m tier, ~45% GPU utilization). V8 replaces it with a
log-depth parallel associative scan (Hillis-Steele / S5-style) for the
first-order linear recurrence

    h_n = a_n * h_{n-1} + u_n,   a_n = exp(lambda_tilde * Delta_n),

implemented on real/imag pairs (inductor cannot codegen complex ops),
then CUDA-graphed via torch.compile(mode="reduce-overhead").

The recurrence is mathematically identical to V6/V7bc: same parameters,
same outputs (verified to fp tolerance), same checkpoints load directly.

Stability: |a_n| <= 1 by construction (sigma > 0, Delta >= 0), so prefix
products never overflow; no log/exp hazards (the V1/V3 incidents' causes
are structurally absent here).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.V1_LM import WaveletStateKANLM
from models.V7_WSKAN import InterpretableWaveletStateKANLayer


def _scan_fwd(a_re: torch.Tensor, a_im: torch.Tensor,
              u_re: torch.Tensor, u_im: torch.Tensor):
    """Hillis-Steele parallel scan of h_n = a_n * h_{n-1} + u_n (complex as
    real pairs). All tensors (B, L, I, N); h_0 = u_0."""
    h_re, h_im = u_re, u_im
    p_re, p_im = a_re, a_im
    L = u_re.shape[-1]
    shift = 1
    while shift < L:
        t_re = p_re[..., shift:] * h_re[..., :-shift] - p_im[..., shift:] * h_im[..., :-shift]
        t_im = p_re[..., shift:] * h_im[..., :-shift] + p_im[..., shift:] * h_re[..., :-shift]
        h_re = torch.cat([h_re[..., :shift], h_re[..., shift:] + t_re], dim=-1)
        h_im = torch.cat([h_im[..., :shift], h_im[..., shift:] + t_im], dim=-1)
        # p[n] *= p[n - shift]  (prefix product so far, not the original a)
        pr_re = p_re[..., shift:] * p_re[..., :-shift] - p_im[..., shift:] * p_im[..., :-shift]
        pr_im = p_re[..., shift:] * p_im[..., :-shift] + p_im[..., shift:] * p_re[..., :-shift]
        p_re = torch.cat([p_re[..., :shift], pr_re], dim=-1)
        p_im = torch.cat([p_im[..., :shift], pr_im], dim=-1)
        shift *= 2
    return h_re, h_im


_scan_fwd_c = torch.compile(_scan_fwd, mode="reduce-overhead")


class _AssocScan(torch.autograd.Function):
    """Linear recurrence via lane-parallel Triton kernels (V8_triton_scan).

    forward:  h_n = a_n * h_{n-1} + u_n
    backward: D_n = G_n + conj(a_{n+1}) * D_{n+1}  (transpose of the forward
              complex multiply);  da_re = D_re*h_prev_re + D_im*h_prev_im;
              da_im = D_im*h_prev_re - D_re*h_prev_im;  du = D.
    Verified against the serial loop to <=2e-6.
    """

    @staticmethod
    def forward(ctx, a_re, a_im, u_re, u_im):
        h_re, h_im = _scan_fwd_c(a_re, a_im, u_re, u_im)
        ctx.save_for_backward(a_re, a_im, h_re, h_im)
        return h_re, h_im

    @staticmethod
    def backward(ctx, G_re, G_im):
        a_re, a_im, h_re, h_im = ctx.saved_tensors
        ar = a_re.flip(-1); ai = -a_im.flip(-1)
        Gr = G_re.flip(-1); Gi = G_im.flip(-1)
        ar_shift = torch.cat([torch.ones_like(ar[..., :1]), ar[..., :-1]], dim=-1)
        ai_shift = torch.cat([torch.zeros_like(ai[..., :1]), ai[..., :-1]], dim=-1)
        D_re, D_im = _scan_fwd_c(ar_shift, ai_shift, Gr, Gi)
        D_re = D_re.flip(-1); D_im = D_im.flip(-1)
        h_prev_re = torch.cat([torch.zeros_like(h_re[..., :1]), h_re[..., :-1]], dim=-1)
        h_prev_im = torch.cat([torch.zeros_like(h_im[..., :1]), h_im[..., :-1]], dim=-1)
        da_re = D_re * h_prev_re + D_im * h_prev_im
        da_im = D_im * h_prev_re - D_re * h_prev_im
        return da_re, da_im, D_re, D_im


def _assoc_scan(a_re, a_im, u_re, u_im):
    return _AssocScan.apply(a_re, a_im, u_re, u_im)


class FastWaveletStateKANLayer(InterpretableWaveletStateKANLayer):
    """wskan7bc layer with the parallel-scan recurrence (checkpoint-compatible)."""

    def forward_recurrent(self, x: torch.Tensor, idx: torch.Tensor | None = None) -> torch.Tensor:
        Bsz, L, I, N = x.shape[0], x.shape[1], self.in_dim, self.n_states
        self._compose_g()

        sigma = torch.exp(torch.clamp(self.log_sigma, self.log_sigma_min, self.log_sigma_max))
        rho = self._rho()
        sig_eff = sigma * rho                      # (i, N)
        om_eff = self.omega * rho
        dt = self._compute_dt(x)                   # (B, L, i)
        Bn = self._compute_B(x, idx)               # (B, L, i, N)
        Cn = self._compute_C(x, idx)
        u = Bn * x.unsqueeze(-1) * dt.unsqueeze(-1)  # ZOH write: (B, L, i, N)

        # a_n = exp(lambda_tilde * Delta_n), real pairs
        decay = torch.exp(-sig_eff * dt.unsqueeze(-1))      # (B, L, i, N)
        phase = om_eff * dt.unsqueeze(-1)
        a_re, a_im = decay * torch.cos(phase), decay * torch.sin(phase)

        # scan on the (B, I, N, L) layout: L-last keeps level slices coalesced
        a_re = (decay * torch.cos(phase)).permute(0, 2, 3, 1).contiguous()
        a_im = (decay * torch.sin(phase)).permute(0, 2, 3, 1).contiguous()
        u_re = u.permute(0, 2, 3, 1).contiguous()
        u_im = torch.zeros_like(u_re)

        h_re, h_im = _assoc_scan(a_re, a_im, u_re, u_im)  # (B, I, N, L)

        # readout: y_no = sum_ik C_nik * Re[g_iok * h_nik]
        read_re = (Cn.permute(0, 2, 3, 1) * h_re)  # (B, I, N, L)
        read_im = (Cn.permute(0, 2, 3, 1) * h_im)
        y = torch.einsum("bikl,iok->bol", read_re, self.a) - torch.einsum(
            "bikl,iok->bol", read_im, self.b)
        y = y.permute(0, 2, 1)  # (B, L, o)
        y = y * self._gate(x)
        return y + x @ self.w_base


class WaveletStateKANLMV8(WaveletStateKANLM):
    """V7 LM with V8 fast layers (identical parameter set; checkpoints
    interchange with wskan7bc)."""

    def __init__(self, vocab_size: int = 256, d_model: int = 32, n_layers: int = 3,
                 n_states: int = 6, chunk_size: int = 16,
                 use_feature_bc: bool = True, bc_rank: int = 32,
                 wz_diag: bool = True, g_rank: int | None = 32,
                 grad_checkpoint: bool = False, compile_chunk: bool = False,
                 oscillatory: bool = True):
        super().__init__(vocab_size, d_model, n_layers, n_states)
        self.layers = nn.ModuleList(
            FastWaveletStateKANLayer(d_model, d_model, n_states, chunk_size=chunk_size,
                                     use_feature_bc=use_feature_bc, bc_rank=bc_rank,
                                     wz_diag=wz_diag, g_rank=g_rank,
                                     oscillatory=oscillatory)
            for _ in range(n_layers)
        )
        self.prenorms = nn.ModuleList(nn.LayerNorm(d_model) for _ in range(n_layers))

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        x = self.tok_emb(idx)
        for norm, layer in zip(self.prenorms, self.layers):
            x = x + layer(norm(x), idx)
        return self.head(self.norm(x))
