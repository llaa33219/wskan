"""V6 Wavelet-State-KAN: factorized dilation - V4's timescale freedom on
V3's analyzable skeleton.

Design (see the V4 memory analysis and the factorization argument):

    Delta_{n,i,k} = Delta_dyn_{n,i} * rho_k

- Delta_dyn: input-dependent, per channel (V3 style) - the content warp.
- rho_k: static, learnable, one ladder per layer shared across channels -
  the multiresolution scale ladder (geometric in spirit, init rho=1).

With rho treated as eigenvalue scaling (lambda~ = rho * lambda), the write
u = B * Delta_dyn * x is exactly the first-order ZOH form of the per-mode
system, so ZOH consistency holds at channel granularity - no per-mode
write/timescale coupling.

What this preserves from V4: freed sigma bounds, multi-resolution timescale
ladder, long-memory slow modes (half-life_k = 0.693 / (rho_k sigma_k
Delta_dyn_i)), ZOH write.

What it restores from V3: a single warped time axis per channel, so the edge
function is a genuine wavelet on a content-warped axis and all V3 analysis
tools apply (with effective modes lambda~ = rho lambda).

Given up (measured-unused so far): per-mode content-driven timescale gating.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.V1_LM import WaveletStateKANLM
from models.V3_WSKAN import _DT_MAX, SelectiveWaveletStateKANLayer

_RHO_MIN, _RHO_MAX = -3.0, 3.0  # rho in [e^-3, e^3] ~= [0.05, 20]


class FactorizedWaveletStateKANLayer(SelectiveWaveletStateKANLayer):
    """V3 layer + static per-mode scale ladder rho + freed sigma + ZOH write."""

    def __init__(self, in_dim: int, out_dim: int, n_states: int = 6, chunk_size: int = 16,
                 eps: float = 1e-2, oscillatory: bool = True, grad_checkpoint: bool = False,
                 compile_chunk: bool = False):
        super().__init__(in_dim, out_dim, n_states, chunk_size, eps, oscillatory,
                         grad_checkpoint, compile_chunk)
        self.log_sigma_min, self.log_sigma_max = -6.0, 8.0  # V4's freed bounds
        self.log_rho = nn.Parameter(torch.zeros(n_states))  # rho = 1 at init

    def _rho(self) -> torch.Tensor:
        return torch.exp(torch.clamp(self.log_rho, _RHO_MIN, _RHO_MAX))

    def _chunk(self, dtc, uc, Cc, h_re, h_im, sigma, mask):
        # dtc: (B, C, i) per-channel dynamic warp; uc/Cc: (B, C, i, N)
        Bsz, C, I, N = uc.shape
        uc = uc * dtc.unsqueeze(-1)  # ZOH write at channel granularity: u = B * Delta * x
        tau = torch.cumsum(dtc, dim=1)  # (B, C, i) - single warped time per channel
        diff = (tau.unsqueeze(2) - tau.unsqueeze(1)).clamp_min(0)  # (B, C, C, i)
        # clamp before exp: above-diagonal entries (masked away later)
        # have diff < 0 and would otherwise overflow exp(-sigma*diff).
        sig_eff = sigma * self._rho()  # (i, N) effective decay = rho_k * sigma_ik
        om_eff = self.omega * self._rho()  # (i, N) effective frequency
        env = torch.exp(-sig_eff * diff.unsqueeze(-1)) * mask[None, :, :, None, None]
        phase = om_eff * diff.unsqueeze(-1)
        Dr, Di = env * torch.cos(phase), env * torch.sin(phase)
        Drf = Dr.permute(0, 3, 4, 1, 2).reshape(-1, C, C)
        Dif = Di.permute(0, 3, 4, 1, 2).reshape(-1, C, C)
        uf = uc.permute(0, 2, 3, 1).reshape(-1, C, 1)
        z_re = torch.bmm(Drf, uf).view(Bsz, I, N, C).permute(0, 3, 1, 2)
        z_im = torch.bmm(Dif, uf).view(Bsz, I, N, C).permute(0, 3, 1, 2)
        d_env = torch.exp(-sig_eff * tau.unsqueeze(-1))
        d_ph = om_eff * tau.unsqueeze(-1)
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
        rho = self._rho()
        lam = torch.complex(-sigma * rho, self.omega * rho)
        dt = self._compute_dt(x)  # (B, L, i) per-channel
        Bn = self.W_B(x).view(Bsz, L, I, N)
        Cn = self.W_C(x).view(Bsz, L, I, N)
        h = torch.zeros(Bsz, I, N, dtype=torch.complex64, device=x.device)
        outs = []
        for n in range(L):
            a_n = torch.exp(lam * dt[:, n].unsqueeze(-1))  # (B, i, N)
            write = (Bn[:, n] * dt[:, n].unsqueeze(-1) * x[:, n].unsqueeze(-1)).to(torch.complex64)
            h = a_n * h + write
            r = Cn[:, n] * h
            y = torch.einsum("bik,iok->bo", r.real, self.a) - torch.einsum(
                "bik,iok->bo", r.imag, self.b
            )
            outs.append(y * F.silu(self.W_z(x[:, n])) + x[:, n] @ self.w_base)
        return torch.stack(outs, dim=1)


class WaveletStateKANLMV6(WaveletStateKANLM):
    """V3 LM with V6 factorized-timescale layers (pre-norm residuals kept)."""

    def __init__(self, vocab_size: int = 256, d_model: int = 32, n_layers: int = 3,
                 n_states: int = 6, oscillatory: bool = True, chunk_size: int = 16,
                 grad_checkpoint: bool = False, compile_chunk: bool = False):
        super().__init__(vocab_size, d_model, n_layers, n_states)
        self.layers = nn.ModuleList(
            FactorizedWaveletStateKANLayer(d_model, d_model, n_states, chunk_size=chunk_size,
                                           oscillatory=oscillatory,
                                           grad_checkpoint=grad_checkpoint,
                                           compile_chunk=compile_chunk)
            for _ in range(n_layers)
        )
        self.prenorms = nn.ModuleList(nn.LayerNorm(d_model) for _ in range(n_layers))

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        x = self.tok_emb(idx)
        for norm, layer in zip(self.prenorms, self.layers):
            x = x + layer(norm(x))
        return self.head(self.norm(x))
