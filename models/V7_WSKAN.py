"""V7 Wavelet-State-KAN: interpretable-by-construction reparameterization.

Three changes over V6, each behind a flag, each turning a "probe me later"
component into a named-object component:

1. Feature-factorized B/C (use_feature_bc): the write/read gates are
       B_n = sum_f alpha_f(byte_n) * M_B[f] + lowrank(x_n)
   with FIXED named byte features alpha (space/newline/punct/upper/lower/
   digit/vowel/constant). The named part is interpretable by lookup - no
   probing. A small low-rank residual from the hidden state preserves
   context-dependent gating.

2. Diagonal output gate (wz_diag): y * SiLU(w_z * x) - per-channel gating,
   readable as scalars.

3. Low-rank edge gains (g_rank): g_iok = sum_r u_ir v_or m_rk, so each
   layer's edge functions are psi_io(t) = sum_r u_ir v_or phi_r(t) with R
   named temporal filters phi_r - the edge atlas collapses from 1,024x6
   edges to R filters plus loadings.

Design rule from V6's success: every reparameterization must pass the
neutrality test (parity with V6 on the 3-seed UltraChat protocol) or its
cost is documented as an interpretability/performance tradeoff.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.V1_LM import WaveletStateKANLM
from models.V6_WSKAN import FactorizedWaveletStateKANLayer

_N_BYTE_FEATURES = 8


def byte_features(idx: torch.Tensor) -> torch.Tensor:
    """idx: (B, L) int64 -> (B, L, F) fixed named features.

    F = [is_space, is_newline, is_punct, is_upper, is_lower, is_digit,
         is_vowel, 1]
    """
    b = idx
    feats = torch.stack([
        (b == 32).float(),
        (b == 10).float(),
        (~((b == 32) | (b == 10) | (b >= 48) & (b <= 57) | (b >= 65) & (b <= 90) | (b >= 97) & (b <= 122))).float(),
        ((b >= 65) & (b <= 90)).float(),
        ((b >= 97) & (b <= 122)).float(),
        ((b >= 48) & (b <= 57)).float(),
        ((b == 97) | (b == 101) | (b == 105) | (b == 111) | (b == 117)
         | (b == 65) | (b == 69) | (b == 73) | (b == 79) | (b == 85)).float(),
        torch.ones_like(b, dtype=torch.float32),
    ], dim=-1)
    return feats


class InterpretableWaveletStateKANLayer(FactorizedWaveletStateKANLayer):
    """V6 layer with feature-factorized B/C, diagonal W_z, low-rank g."""

    def __init__(self, in_dim: int, out_dim: int, n_states: int = 6, chunk_size: int = 16,
                 eps: float = 1e-2, oscillatory: bool = True, grad_checkpoint: bool = False,
                 compile_chunk: bool = False, use_feature_bc: bool = True, bc_rank: int = 32,
                 wz_diag: bool = True, g_rank: int | None = 32):
        super().__init__(in_dim, out_dim, n_states, chunk_size, eps, oscillatory,
                         grad_checkpoint, compile_chunk)
        I, N = in_dim, n_states

        if use_feature_bc:
            self.M_B = nn.Parameter(torch.randn(_N_BYTE_FEATURES, I, N) * (I * N) ** -0.5)
            self.M_C = nn.Parameter(torch.randn(_N_BYTE_FEATURES, I, N) * (I * N) ** -0.5)
            self.res_B = nn.Sequential(
                nn.Linear(in_dim, bc_rank, bias=False),
                nn.Linear(bc_rank, in_dim * n_states),
            )
            self.res_C = nn.Sequential(
                nn.Linear(in_dim, bc_rank, bias=False),
                nn.Linear(bc_rank, in_dim * n_states),
            )
            for m in (*self.res_B, *self.res_C):
                if isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, std=0.02)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
            del self.W_B
            del self.W_C

        if wz_diag:
            assert in_dim == out_dim, "diagonal output gate requires square layers"
            self.w_z_diag = nn.Parameter(torch.ones(out_dim))
            del self.W_z

        if g_rank is not None:
            R = g_rank
            self.u_g = nn.Parameter(torch.randn(I, R) * R**-0.5)
            self.v_g = nn.Parameter(torch.randn(out_dim, R) * R**-0.5)
            self.m_re = nn.Parameter(torch.randn(R, N) * 0.1)
            self.m_im = nn.Parameter(torch.randn(R, N) * 0.1)
            del self.a
            del self.b
            self.register_buffer("a", torch.zeros(I, out_dim, N), persistent=False)
            self.register_buffer("b", torch.zeros(I, out_dim, N), persistent=False)

    def _compose_g(self) -> None:
        if hasattr(self, "u_g"):
            self.a = torch.einsum("ir,or,rk->iok", self.u_g, self.v_g, self.m_re)
            self.b = torch.einsum("ir,or,rk->iok", self.u_g, self.v_g, self.m_im)

    def _compute_B(self, x: torch.Tensor, idx: torch.Tensor | None) -> torch.Tensor:
        Bsz, L, I, N = x.shape[0], x.shape[1], self.in_dim, self.n_states
        if hasattr(self, "M_B"):
            named = torch.einsum("blf,fin->blin", byte_features(idx), self.M_B)
            res = self.res_B(x).view(Bsz, L, I, N)
            return named + res
        return self.W_B(x).view(Bsz, L, I, N)

    def _compute_C(self, x: torch.Tensor, idx: torch.Tensor | None) -> torch.Tensor:
        Bsz, L, I, N = x.shape[0], x.shape[1], self.in_dim, self.n_states
        if hasattr(self, "M_C"):
            named = torch.einsum("blf,fin->blin", byte_features(idx), self.M_C)
            res = self.res_C(x).view(Bsz, L, I, N)
            return named + res
        return self.W_C(x).view(Bsz, L, I, N)

    def _gate(self, x: torch.Tensor) -> torch.Tensor:
        if hasattr(self, "w_z_diag"):
            return F.silu(self.w_z_diag * x)
        return F.silu(self.W_z(x))

    def forward_recurrent(self, x: torch.Tensor, idx: torch.Tensor | None = None) -> torch.Tensor:
        Bsz, L, I, N = x.shape[0], x.shape[1], self.in_dim, self.n_states
        self._compose_g()
        C = self.chunk_size
        pad = (C - L % C) % C
        if pad:
            x = F.pad(x, (0, 0, 0, pad))
        Lp = L + pad

        sigma = torch.exp(torch.clamp(self.log_sigma, self.log_sigma_min, self.log_sigma_max))
        dt = self._compute_dt(x)
        idx_p = F.pad(idx, (0, pad)) if (idx is not None and pad) else idx
        Bn = self._compute_B(x, idx_p)
        Cn = self._compute_C(x, idx_p)
        u = Bn * x.unsqueeze(-1)

        h_re = torch.zeros(Bsz, I, N, device=x.device)
        h_im = torch.zeros(Bsz, I, N, device=x.device)
        ys = []
        mask = torch.tril(torch.ones(C, C, dtype=x.dtype, device=x.device))
        for c0 in range(0, Lp, C):
            args = (dt[:, c0:c0+C], u[:, c0:c0+C], Cn[:, c0:c0+C], h_re, h_im, sigma, mask)
            if self.grad_checkpoint and self.training:
                yc, h_re, h_im = torch.utils.checkpoint.checkpoint(
                    self._chunk_fn, *args, use_reentrant=False)
            else:
                yc, h_re, h_im = self._chunk_fn(*args)
            ys.append(yc)
        y = torch.cat(ys, dim=1)[:, :L]
        y = y * self._gate(x[:, :L])
        return y + x[:, :L] @ self.w_base

    def forward_recurrent_loop(self, x: torch.Tensor, idx: torch.Tensor | None = None) -> torch.Tensor:
        Bsz, L, I, N = x.shape[0], x.shape[1], self.in_dim, self.n_states
        self._compose_g()
        sigma = torch.exp(torch.clamp(self.log_sigma, self.log_sigma_min, self.log_sigma_max))
        rho = self._rho()
        lam = torch.complex(-sigma * rho, self.omega * rho)
        dt = self._compute_dt(x)
        Bn = self._compute_B(x, idx)
        Cn = self._compute_C(x, idx)
        h = torch.zeros(Bsz, I, N, dtype=torch.complex64, device=x.device)
        outs = []
        for n in range(L):
            a_n = torch.exp(lam * dt[:, n].unsqueeze(-1))
            write = (Bn[:, n] * dt[:, n].unsqueeze(-1) * x[:, n].unsqueeze(-1)).to(torch.complex64)
            h = a_n * h + write
            r = Cn[:, n] * h
            y = torch.einsum("bik,iok->bo", r.real, self.a) - torch.einsum(
                "bik,iok->bo", r.imag, self.b)
            outs.append(y * self._gate(x[:, n:n+1]).squeeze(1) + x[:, n] @ self.w_base)
        return torch.stack(outs, dim=1)

    def forward(self, x: torch.Tensor, idx: torch.Tensor | None = None) -> torch.Tensor:
        if x.dim() == 2:
            self._compose_g()
            return self.forward_static(x)
        return self.forward_recurrent(x, idx)


class WaveletStateKANLMV7(WaveletStateKANLM):
    """V6 LM with V7 interpretable layers; layers receive byte ids."""

    def __init__(self, vocab_size: int = 256, d_model: int = 32, n_layers: int = 3,
                 n_states: int = 6, chunk_size: int = 16,
                 use_feature_bc: bool = True, bc_rank: int = 32,
                 wz_diag: bool = True, g_rank: int | None = 32):
        super().__init__(vocab_size, d_model, n_layers, n_states)
        self.layers = nn.ModuleList(
            InterpretableWaveletStateKANLayer(d_model, d_model, n_states, chunk_size=chunk_size,
                                              use_feature_bc=use_feature_bc, bc_rank=bc_rank,
                                              wz_diag=wz_diag, g_rank=g_rank)
            for _ in range(n_layers)
        )
        self.prenorms = nn.ModuleList(nn.LayerNorm(d_model) for _ in range(n_layers))

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        x = self.tok_emb(idx)
        for norm, layer in zip(self.prenorms, self.layers):
            x = x + layer(norm(x), idx)
        return self.head(self.norm(x))
