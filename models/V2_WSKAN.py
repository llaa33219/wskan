"""V2 Wavelet-State-KAN: gated (selective) wavelet-SSM edges.

V1's recurrent mode is LTI: after training, every edge convolves its input
with a fixed wavelet kernel. Mamba's advantage comes from input-dependent
selectivity. V2 adds H3/Mamba-style multiplicative gating while keeping the
LTI state dynamics - and therefore the exact "wavelet = SSM impulse response"
identity and the O(L log L) FFT path:

    u_n = x_n * silu(W_g x_n)              (input gate: what enters memory)
    h_n = Abar h_{n-1} + Bbar u_n          (LTI wavelet-SSM, per edge)
    y_n = (g . h_n) * silu(W_z x_n) + W_base x_n   (output gate + direct pass)

What V2 deliberately does NOT do: per-step input-dependent Delta/B/C. That
would make the system time-varying, breaking the wavelet-kernel identity and
FFT acceleration; it is the V3 candidate. See V2_README.md.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.V1_LM import WaveletStateKANLM
from models.V1_WSKAN import WaveletStateKANLayer


class GatedWaveletStateKANLayer(WaveletStateKANLayer):
    """V1 wavelet-SSM edge layer + input/output multiplicative gates."""

    def __init__(self, in_dim: int, out_dim: int, n_states: int = 6, eps: float = 1e-2):
        super().__init__(in_dim, out_dim, n_states, eps)
        self.W_g = nn.Linear(in_dim, in_dim, bias=False)
        self.W_z = nn.Linear(in_dim, out_dim, bias=False)
        nn.init.normal_(self.W_g.weight, std=in_dim**-0.5)
        nn.init.normal_(self.W_z.weight, std=in_dim**-0.5)

    def forward_recurrent(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, L, in_dim) -> (B, L, out_dim)."""
        B, L, _ = x.shape
        n_fft = 2 * L
        k = self._ssm_kernel(L) * self.w_wav.unsqueeze(-1)  # (in, out, L)

        u = x * F.silu(self.W_g(x))  # input gate
        u_f = torch.fft.fft(u.transpose(1, 2), n=n_fft)
        k_f = torch.fft.fft(k, n=n_fft)
        y_f = torch.einsum("bin,ion->bon", u_f, k_f)
        y = torch.fft.ifft(y_f, n=n_fft)[:, :, :L].real.transpose(1, 2)

        y = y * F.silu(self.W_z(x))  # output gate
        return y + x @ self.w_base

    def forward_recurrent_loop(self, x: torch.Tensor) -> torch.Tensor:
        """O(L) sequential reference for verification and streaming."""
        B, L, _ = x.shape
        sigma = self._sigma()
        lam = torch.complex(-sigma, self.omega)
        s = self._scale().unsqueeze(-1)
        a_bar = torch.exp(lam * s)
        b_bar = (a_bar - 1.0) / lam
        g = torch.complex(self.a, self.b)

        u = x * F.silu(self.W_g(x))
        z = F.silu(self.W_z(x))
        h = torch.zeros(
            B, self.in_dim, self.out_dim, self.n_states,
            dtype=torch.complex64, device=x.device,
        )
        outs = []
        for n in range(L):
            un = u[:, n, :].unsqueeze(-1).unsqueeze(-1)
            h = a_bar * h + b_bar * un
            y = torch.einsum("ion,bion->bio", g, h).real
            outs.append(self.w_wav * y * z[:, n, :].unsqueeze(1))
        return torch.stack(outs, dim=1).sum(dim=2) + x @ self.w_base


class WaveletStateKANLMV2(WaveletStateKANLM):
    """V1 LM wrapper with V2 gated layers substituted in."""

    def __init__(self, vocab_size: int = 256, d_model: int = 32, n_layers: int = 3, n_states: int = 6):
        super().__init__(vocab_size, d_model, n_layers, n_states)
        self.layers = nn.ModuleList(
            GatedWaveletStateKANLayer(d_model, d_model, n_states) for _ in range(n_layers)
        )
