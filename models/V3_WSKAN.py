"""V3 Wavelet-State-KAN: fully selective wavelet-SSM edges.

V1: edge wavelet = LTI SSM impulse response (fixed kernel, FFT path).
V2: + static input/output gates around the LTI core.
V3: full Mamba-style selectivity - dt, B, C are functions of the input -
    while the wavelet skeleton (damped-oscillator eigenvalues lambda and
    complex edge gains g) stays learned and shared with the static mode.

The selective scan has an exact wavelet reading. With cumulative warped time
T_n = sum_j Delta_j:

    h_n = sum_{k<=n} exp(lambda * (T_n - T_k)) * B_k * x_k,

i.e. the edge applies its wavelet psi(t) = Re[g e^{lambda t}] to the input in
content-warped time: wavelet dilation becomes dynamic and input-driven. This
is the "perfect harmony" the project name promises: Mamba's Delta *is* a
learned, per-token wavelet dilation.

Architecture note (honest): the SSM state lives per (input channel, mode);
edges mix mode responses through learned complex gains g_{iok}. Per-edge
time-varying state is computationally infeasible at scale, so the edge
wavelet is psi_io(t) = sum_k Re[g_iok e^{lambda_ik t}] - a gain-weighted mix
of channel i's oscillator bank.

Computation: chunked SSD scan (Mamba-2 style). Within a chunk, the causal
decay matrix exp(lambda (T_n - T_k)) has Re(lambda)(T_n - T_k) <= 0 for
k <= n, so no overflow is possible; chunks carry state sequentially.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.V1_LM import WaveletStateKANLM

_LOG_MIN, _LOG_MAX = -4.0, 2.0  # legacy bounds for V1/V2 (pow/log underflow guard)
_DT_MAX = 1.0  # per-step dilation cap: keeps warped time bounded per token


class SelectiveWaveletStateKANLayer(nn.Module):
    """Selective wavelet-SSM edge layer.

    Static mode (2D input):    nominal wavelet psi_io(t), V1-style.
    Recurrent mode (3D input): selective scan with input-driven dt, B, C.

    Args:
        in_dim, out_dim: node counts.
        n_states: oscillator modes per input channel.
        chunk_size: SSD chunk length.
    """

    def __init__(self, in_dim: int, out_dim: int, n_states: int = 6, chunk_size: int = 16,
                 eps: float = 1e-2, oscillatory: bool = True, grad_checkpoint: bool = False,
                 compile_chunk: bool = False):
        super().__init__()
        self.in_dim, self.out_dim, self.n_states = in_dim, out_dim, n_states
        self.chunk_size = chunk_size
        self.eps = eps
        self.grad_checkpoint = grad_checkpoint
        self._chunk_fn = torch.compile(self._chunk) if compile_chunk else self._chunk
        self.log_sigma_min, self.log_sigma_max = _LOG_MIN, _LOG_MAX

        # Wavelet skeleton: damped-oscillator bank per input channel (S4D-Lin init)
        self.log_sigma = nn.Parameter(torch.full((in_dim, n_states), math.log(0.5)))
        if oscillatory:
            omega_init = math.pi * torch.arange(1, n_states + 1, dtype=torch.float32).repeat(in_dim, 1)
        else:
            omega_init = torch.zeros(in_dim, n_states)  # ablation: pure decay modes
        self.omega = nn.Parameter(omega_init, requires_grad=oscillatory)
        # Edge gains g_io = a + i*b: how edge (i,o) mixes channel i's modes
        self.a = nn.Parameter(torch.randn(in_dim, out_dim, n_states) * 0.1)
        self.b = nn.Parameter(torch.randn(in_dim, out_dim, n_states) * 0.1)

        # Static-mode dilation/translation per edge (unused in recurrent mode)
        self.mu = nn.Parameter(torch.zeros(in_dim, out_dim))
        self.log_s = nn.Parameter(torch.zeros(in_dim, out_dim))

        # Selectivity projections (input-dependent dt, B, C)
        self.W_dt = nn.Linear(in_dim, in_dim)
        nn.init.normal_(self.W_dt.weight, std=0.02)
        nn.init.constant_(self.W_dt.bias, math.log(math.expm1(0.05)))  # dt ~= 0.05 at init
        self.W_B = nn.Linear(in_dim, in_dim * n_states, bias=False)
        self.W_C = nn.Linear(in_dim, in_dim * n_states, bias=False)
        nn.init.normal_(self.W_B.weight, std=in_dim**-0.5)
        nn.init.normal_(self.W_C.weight, std=in_dim**-0.5)

        self.W_z = nn.Linear(in_dim, out_dim, bias=False)  # output gate
        nn.init.normal_(self.W_z.weight, std=in_dim**-0.5)
        self.w_base = nn.Parameter(
            torch.empty(in_dim, out_dim).uniform_(-1 / math.sqrt(in_dim), 1 / math.sqrt(in_dim))
        )
        self.w_wav = nn.Parameter(torch.ones(in_dim, out_dim))

    def _lambda(self) -> torch.Tensor:
        sigma = torch.exp(torch.clamp(self.log_sigma, self.log_sigma_min, self.log_sigma_max))
        return torch.complex(-sigma, self.omega)  # (in, N), Re < 0 always

    def _compute_dt(self, x: torch.Tensor) -> torch.Tensor:
        return torch.clamp(F.softplus(self.W_dt(x)), max=_DT_MAX)

    # ------------------------------------------------------------------ #
    # Static mode: nominal per-edge wavelet psi_io(t)                    #
    # ------------------------------------------------------------------ #
    def wavelet(self, t: torch.Tensor) -> torch.Tensor:
        """t: (B, in_dim, out_dim) -> psi: (B, in_dim, out_dim)."""
        sigma = torch.exp(torch.clamp(self.log_sigma, self.log_sigma_min, self.log_sigma_max))  # (in, N)
        m = torch.sqrt(t.unsqueeze(-1) ** 2 + self.eps)  # (B, in, out, 1)
        env = torch.exp(-sigma.unsqueeze(1) * m)  # (B, in, out, N)
        phase = self.omega.unsqueeze(1) * t.unsqueeze(-1)
        psi = (env * (self.a * torch.cos(phase) + self.b * torch.sin(phase))).sum(-1)

        # DC correction (same construction as V1, per-edge mean envelope)
        dc = (self.a * 2.0 * sigma.unsqueeze(1) / (sigma.unsqueeze(1) ** 2 + self.omega.unsqueeze(1) ** 2)).sum(-1)
        sigma_bar = sigma.mean(-1).unsqueeze(1)  # (in, 1)
        g_env = 0.5 * sigma_bar * torch.exp(-sigma_bar * m.squeeze(-1))
        return psi - dc * g_env

    def forward_static(self, x: torch.Tensor) -> torch.Tensor:
        s = torch.exp(torch.clamp(self.log_s, _LOG_MIN, _LOG_MAX))
        t = (x.unsqueeze(-1) - self.mu) / s
        wav = self.wavelet(t) / torch.sqrt(s)
        base = F.silu(x).unsqueeze(-1) * self.w_base
        return (base + self.w_wav * wav).sum(dim=1)

    # ------------------------------------------------------------------ #
    # Recurrent mode: chunked selective (SSD) scan                       #
    # ------------------------------------------------------------------ #
    def _chunk(self, dtc, uc, Cc, h_re, h_im, sigma, mask):
        Bsz, C, I, N = uc.shape
        tau = torch.cumsum(dtc, dim=1)  # local warped time (B, C, i)
        diff = (tau.unsqueeze(2) - tau.unsqueeze(1)).clamp_min(0)  # (B, C, C, i)
        # clamp before exp: above-diagonal entries (masked away later)
        # have diff < 0 and would otherwise overflow exp(-sigma*diff).
        env = torch.exp(-sigma * diff.unsqueeze(-1)) * mask[None, ..., None, None]
        phase = self.omega * diff.unsqueeze(-1)  # (B, C, C, i, N)
        Dr, Di = env * torch.cos(phase), env * torch.sin(phase)
        # bmm over (B, i, N): z = D @ u  ->  (B, C, i, N) complex as (re, im)
        Drf = Dr.permute(0, 3, 4, 1, 2).reshape(-1, C, C)
        Dif = Di.permute(0, 3, 4, 1, 2).reshape(-1, C, C)
        uf = uc.permute(0, 2, 3, 1).reshape(-1, C, 1)
        z_re = torch.bmm(Drf, uf).view(Bsz, I, N, C).permute(0, 3, 1, 2)
        z_im = torch.bmm(Dif, uf).view(Bsz, I, N, C).permute(0, 3, 1, 2)
        # state carry: exp(lam * tau) * h
        d_env = torch.exp(-sigma * tau.unsqueeze(-1))  # (B, C, i, N)
        d_ph = self.omega * tau.unsqueeze(-1)
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

    def forward_recurrent(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, L, in_dim) -> (B, L, out_dim)."""
        Bsz, L, I, N = x.shape[0], x.shape[1], self.in_dim, self.n_states
        C = self.chunk_size
        pad = (C - L % C) % C
        if pad:
            x = F.pad(x, (0, 0, 0, pad))
        Lp = L + pad

        sigma = torch.exp(torch.clamp(self.log_sigma, self.log_sigma_min, self.log_sigma_max))  # (i, N)
        dt = self._compute_dt(x)  # (B, Lp, i)
        Bn = self.W_B(x).view(Bsz, Lp, I, N)
        Cn = self.W_C(x).view(Bsz, Lp, I, N)
        u = Bn * x.unsqueeze(-1)  # write, real: (B, Lp, i, N)

        h_re = torch.zeros(Bsz, I, N, device=x.device)
        h_im = torch.zeros(Bsz, I, N, device=x.device)
        ys = []
        mask = torch.tril(torch.ones(C, C, dtype=x.dtype, device=x.device))
        for c0 in range(0, Lp, C):
            args = (dt[:, c0 : c0 + C], u[:, c0 : c0 + C], Cn[:, c0 : c0 + C], h_re, h_im, sigma, mask)
            if self.grad_checkpoint and self.training:
                yc, h_re, h_im = torch.utils.checkpoint.checkpoint(
                    self._chunk_fn, *args, use_reentrant=False
                )
            else:
                yc, h_re, h_im = self._chunk_fn(*args)
            ys.append(yc)
        y = torch.cat(ys, dim=1)[:, :L]
        y = y * F.silu(self.W_z(x[:, :L]))
        return y + x[:, :L] @ self.w_base

    def forward_recurrent_loop(self, x: torch.Tensor) -> torch.Tensor:
        """O(L) sequential reference for verification and streaming."""
        Bsz, L, I, N = x.shape[0], x.shape[1], self.in_dim, self.n_states
        lam = self._lambda()
        dt = self._compute_dt(x)
        Bn = self.W_B(x).view(Bsz, L, I, N)
        Cn = self.W_C(x).view(Bsz, L, I, N)
        h = torch.zeros(Bsz, I, N, dtype=torch.complex64, device=x.device)
        outs = []
        for n in range(L):
            a_n = torch.exp(lam * dt[:, n].unsqueeze(-1))  # (B, i, N)
            h = a_n * h + Bn[:, n] * x[:, n].unsqueeze(-1)
            r = Cn[:, n] * h
            y = torch.einsum("bik,iok->bo", r.real, self.a) - torch.einsum(
                "bik,iok->bo", r.imag, self.b
            )
            outs.append(y * F.silu(self.W_z(x[:, n])) + x[:, n] @ self.w_base)
        return torch.stack(outs, dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            return self.forward_static(x)
        if x.dim() == 3:
            return self.forward_recurrent(x)
        raise ValueError(f"expected 2D (static) or 3D (recurrent) input, got {x.dim()}D")

    @torch.no_grad()
    def spectral_signature(self) -> dict:
        sigma = torch.exp(torch.clamp(self.log_sigma, self.log_sigma_min, self.log_sigma_max))
        return {"sigma": sigma, "omega": self.omega}


class WaveletStateKANLMV3(WaveletStateKANLM):
    """V1 LM wrapper with V3 selective layers and pre-norm residual blocks.

    oscillatory=False freezes omega at 0 (pure decay modes) - the ablation
    that isolates the wavelet contribution from generic SSM selectivity.

    Pre-norm (x + layer(norm(x))) is load-bearing here, not cosmetic: the
    multiplicative gates make layer gain superlinear in the residual scale,
    and without inter-layer normalization the residual stream can blow up
    multiplicatively across layers (observed: 55 -> 2.5e6 -> 5.7e25 -> NaN).
    """

    def __init__(self, vocab_size: int = 256, d_model: int = 32, n_layers: int = 3,
                 n_states: int = 6, oscillatory: bool = True, chunk_size: int = 16,
                 grad_checkpoint: bool = False, compile_chunk: bool = False):
        super().__init__(vocab_size, d_model, n_layers, n_states)
        self.layers = nn.ModuleList(
            SelectiveWaveletStateKANLayer(d_model, d_model, n_states, chunk_size=chunk_size,
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
