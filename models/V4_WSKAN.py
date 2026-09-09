"""V4 Wavelet-State-KAN: free per-mode timescales.

Evidence-driven deltas from V3 (see V3_MULTISEED_ABLATION_REPORT.md and
V3_10M_PROBE_REPORT.md):

1. Freed decay: sigma bounds [e^-4, e^2] -> [e^-6, e^8]. The V1-era clamp was
   an underflow guard for the pow/log kernel path; V3/V4's chunked form never
   exponentiates positive arguments (decay matrix is causal, Re <= 0), so the
   tight ceiling only constrained the model - at 100k scale a third of modes
   were pinned at it.
2. Per-mode dynamic dilation: Delta becomes per (channel, mode) instead of
   per channel - each wavelet mode warps time independently. Low-rank
   projection (rank 8) keeps the parameter cost small.
3. ZOH-consistent write: u = B * Delta * x (the discretized input gain,
   matching Mamba's dt*B and the V1/V2 B_bar factor that V3 dropped).

Everything else (chunked SSD scan, gains, gates, pre-norm) is inherited
from V3 unchanged.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.V1_LM import WaveletStateKANLM
from models.V3_WSKAN import _DT_MAX, SelectiveWaveletStateKANLayer


class FreeWaveletStateKANLayer(SelectiveWaveletStateKANLayer):
    """V3 layer with per-mode dynamic dilation and freed sigma bounds."""

    def __init__(self, in_dim: int, out_dim: int, n_states: int = 6, chunk_size: int = 16,
                 eps: float = 1e-2, oscillatory: bool = True, grad_checkpoint: bool = False,
                 compile_chunk: bool = False, dt_rank: int = 8):
        super().__init__(in_dim, out_dim, n_states, chunk_size, eps, oscillatory,
                         grad_checkpoint, compile_chunk)
        self.log_sigma_min, self.log_sigma_max = -6.0, 8.0

        del self.W_dt  # per-channel dt is replaced by a low-rank per-mode dt
        self.W_dt1 = nn.Linear(in_dim, dt_rank, bias=False)
        self.W_dt2 = nn.Linear(dt_rank, in_dim * n_states)
        nn.init.normal_(self.W_dt1.weight, std=0.02)
        nn.init.normal_(self.W_dt2.weight, std=0.02)
        nn.init.constant_(self.W_dt2.bias, math.log(math.expm1(0.05)))  # dt ~= 0.05 at init

    def _compute_dt(self, x: torch.Tensor) -> torch.Tensor:
        Bsz, L, I = x.shape
        dt = F.softplus(self.W_dt2(self.W_dt1(x)))
        return torch.clamp(dt, max=_DT_MAX).view(Bsz, L, I, self.n_states)

    def _chunk(self, dtc, uc, Cc, h_re, h_im, sigma, mask):
        # dtc: (B, C, i, N) per-mode dilation; uc: (B, C, i, N) pre-write
        Bsz, C, I, N = uc.shape
        uc = uc * dtc  # ZOH-consistent write: u = B * Delta * x
        tau = torch.cumsum(dtc, dim=1)  # (B, C, i, N)
        diff = (tau.unsqueeze(2) - tau.unsqueeze(1)).clamp_min(0)  # (B, C, C, i, N)
        # clamp before exp: above-diagonal entries (masked away later)
        # have diff < 0 and would otherwise overflow exp(-sigma*diff).
        env = torch.exp(-sigma * diff) * mask[None, :, :, None, None]
        phase = self.omega * diff
        Dr, Di = env * torch.cos(phase), env * torch.sin(phase)
        Drf = Dr.permute(0, 3, 4, 1, 2).reshape(-1, C, C)
        Dif = Di.permute(0, 3, 4, 1, 2).reshape(-1, C, C)
        uf = uc.permute(0, 2, 3, 1).reshape(-1, C, 1)
        z_re = torch.bmm(Drf, uf).view(Bsz, I, N, C).permute(0, 3, 1, 2)
        z_im = torch.bmm(Dif, uf).view(Bsz, I, N, C).permute(0, 3, 1, 2)
        d_env = torch.exp(-sigma * tau)
        d_ph = self.omega * tau
        c_re, c_im = d_env * torch.cos(d_ph), d_env * torch.sin(d_ph)
        hc_re = z_re + c_re * h_re.unsqueeze(1) - c_im * h_im.unsqueeze(1)
        hc_im = z_im + c_re * h_im.unsqueeze(1) + c_im * h_re.unsqueeze(1)
        # readout per chunk (avoids materializing full-length hidden tensors)
        read_re = Cc * hc_re
        read_im = Cc * hc_im
        yc = torch.einsum("bcik,iok->bco", read_re, self.a) - torch.einsum(
            "bcik,iok->bco", read_im, self.b
        )
        return yc, hc_re[:, -1], hc_im[:, -1]

    def forward_recurrent_loop(self, x: torch.Tensor) -> torch.Tensor:
        """O(L) sequential reference for verification and streaming."""
        Bsz, L, I, N = x.shape[0], x.shape[1], self.in_dim, self.n_states
        sigma = torch.exp(torch.clamp(self.log_sigma, self.log_sigma_min, self.log_sigma_max))
        lam = torch.complex(-sigma, self.omega)
        dt = self._compute_dt(x)  # (B, L, i, N)
        Bn = self.W_B(x).view(Bsz, L, I, N)
        Cn = self.W_C(x).view(Bsz, L, I, N)
        h = torch.zeros(Bsz, I, N, dtype=torch.complex64, device=x.device)
        outs = []
        for n in range(L):
            a_n = torch.exp(lam * dt[:, n])
            h = a_n * h + (Bn[:, n] * dt[:, n] * x[:, n].unsqueeze(-1)).to(torch.complex64)
            r = Cn[:, n] * h
            y = torch.einsum("bik,iok->bo", r.real, self.a) - torch.einsum(
                "bik,iok->bo", r.imag, self.b
            )
            outs.append(y * F.silu(self.W_z(x[:, n])) + x[:, n] @ self.w_base)
        return torch.stack(outs, dim=1)


class WaveletStateKANLMV4(WaveletStateKANLM):
    """V3 LM with V4 free-timescale layers (pre-norm residuals kept)."""

    def __init__(self, vocab_size: int = 256, d_model: int = 32, n_layers: int = 3,
                 n_states: int = 6, oscillatory: bool = True, chunk_size: int = 16,
                 grad_checkpoint: bool = False, compile_chunk: bool = False):
        super().__init__(vocab_size, d_model, n_layers, n_states)
        self.layers = nn.ModuleList(
            FreeWaveletStateKANLayer(d_model, d_model, n_states, chunk_size=chunk_size,
                                     oscillatory=oscillatory, grad_checkpoint=grad_checkpoint,
                                     compile_chunk=compile_chunk)
            for _ in range(n_layers)
        )
        self.prenorms = nn.ModuleList(nn.LayerNorm(d_model) for _ in range(n_layers))

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        x = self.tok_emb(idx)
        for norm, layer in zip(self.prenorms, self.layers):
            x = x + layer(norm(x))
        return self.head(self.norm(x))
