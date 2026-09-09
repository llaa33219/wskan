"""V1 Wavelet-State-KAN (WSKAN).

A Kolmogorov-Arnold Network layer whose edge wavelets are, by construction,
the impulse responses of a stable linear state-space model (SSM).

Mathematical core
-----------------
Each edge (i, j) carries N damped oscillator modes with complex eigenvalues

    lambda_k = -sigma_k + i * omega_k,   sigma_k = exp(log_sigma_k) > 0,

i.e. a diagonal continuous-time SSM

    h'(t) = A h(t) + B u(t),   A = diag(lambda_1..lambda_N).

Its impulse response is a damped-oscillation wavelet

    psi(t) = sum_k Re[ g_k * exp(lambda_k * t) ]
           = sum_k exp(-sigma_k * t) * (a_k cos(omega_k t) + b_k sin(omega_k t)),

with complex gain g_k = a_k + i * b_k (= C_k * B_k of the SSM).

Because psi is *literally* an SSM impulse response, the edge supports two
mathematically consistent modes:

1. Static mode  (WavKAN-style): psi is evaluated in closed form on scalar
   inputs, with learnable dilation s and translation mu:

       phi(x) = w_base * silu(x) + w_wav * psi((x - mu) / s) / sqrt(s).

2. Recurrent mode (SSM-style): the same (lambda, g) define a ZOH-discretized
   recurrence that scans a sequence:

       h_n = Abar h_{n-1} + Bbar x_n,   y_n = w_wav * Re[g . h_n] + w_base * x_n,

   where the wavelet scale s plays the role of the SSM step size Delta.

Stability is guaranteed in both modes: sigma_k > 0 implies |exp(lambda_k * s)| < 1.

Wavelet admissibility
---------------------
The sin part is odd and integrates to zero. The cos part integrates to
2 * sigma_k / (sigma_k^2 + omega_k^2). We subtract a closed-form DC correction
with a normalized envelope of the same family, so the mother wavelet has
(near-)zero mean; see V1_README.md for the derivation and its (stated)
approximation under the smoothed modulus.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class WaveletStateKANLayer(nn.Module):
    """One KAN layer with state-space-realizable wavelet edges.

    Args:
        in_dim:   number of input nodes.
        out_dim:  number of output nodes.
        n_states: number of damped oscillator modes (SSM state dim) per edge.
        eps:      smoothing for |t| (sqrt(t^2 + eps)); keeps psi differentiable at 0.

    Shapes:
        static mode:    (B, in_dim)    -> (B, out_dim)
        recurrent mode: (B, L, in_dim) -> (B, L, out_dim)
    """

    def __init__(self, in_dim: int, out_dim: int, n_states: int = 8, eps: float = 1e-2):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.n_states = n_states
        self.eps = eps

        # --- SSM state matrix A = diag(lambda), lambda = -sigma + i*omega ---
        # S4D-Lin style initialization: sigma = 1/2, omega_k = pi * k.
        log_sigma_init = math.log(0.5)
        self.log_sigma = nn.Parameter(
            torch.full((in_dim, out_dim, n_states), log_sigma_init)
        )
        self.omega = nn.Parameter(
            math.pi
            * torch.arange(1, n_states + 1, dtype=torch.float32)
            .view(1, 1, n_states)
            .repeat(in_dim, out_dim, 1)
        )

        # --- Complex wavelet gain g = a + i*b  (= C*B of the SSM) ---
        # Small random init; a carries the (even) cosine part, b the sine part.
        self.a = nn.Parameter(torch.randn(in_dim, out_dim, n_states) * 0.1)
        self.b = nn.Parameter(torch.randn(in_dim, out_dim, n_states) * 0.1)

        # --- Wavelet dilation / translation (static mode) ---
        # s = exp(log_s) > 0. In recurrent mode, s acts as the SSM step size.
        self.log_s = nn.Parameter(torch.zeros(in_dim, out_dim))
        self.mu = nn.Parameter(torch.zeros(in_dim, out_dim))

        # --- Edge output weights ---
        self.w_base = nn.Parameter(
            torch.empty(in_dim, out_dim).uniform_(-1 / math.sqrt(in_dim), 1 / math.sqrt(in_dim))
        )
        self.w_wav = nn.Parameter(torch.ones(in_dim, out_dim))

    # Numerical guards --------------------------------------------------
    # sigma = exp(log_sigma) and s = exp(log_s) are clamped to
    # [e^-4, e^2] ~= [0.018, 7.39]. Without this, exp(-sigma*s) can
    # underflow to exact 0.0 in float32, and the complex power
    # Abar**lags produces NaN gradients through log(0) in its backward.
    # This is a documented restriction of the learnable wavelet scales.
    _LOG_MIN, _LOG_MAX = -4.0, 2.0

    def _sigma(self) -> torch.Tensor:
        return torch.exp(torch.clamp(self.log_sigma, self._LOG_MIN, self._LOG_MAX))

    def _scale(self) -> torch.Tensor:
        return torch.exp(torch.clamp(self.log_s, self._LOG_MIN, self._LOG_MAX))

    # ------------------------------------------------------------------ #
    # Static mode: closed-form SSM impulse response as a wavelet basis   #
    # ------------------------------------------------------------------ #
    def wavelet(self, t: torch.Tensor) -> torch.Tensor:
        """Evaluate the DC-corrected mother wavelet psi(t).

        t: (B, in_dim, out_dim) -> psi: (B, in_dim, out_dim)
        """
        sigma = self._sigma()  # (in, out, N), in [e^-4, e^2]
        m = torch.sqrt(t.unsqueeze(-1) ** 2 + self.eps)  # smooth |t|, (B, in, out, 1)
        env = torch.exp(-sigma * m)  # (B, in, out, N)
        phase = self.omega * t.unsqueeze(-1)
        psi = (env * (self.a * torch.cos(phase) + self.b * torch.sin(phase))).sum(-1)

        # Closed-form DC correction for admissibility (zero mean).
        # Integral of exp(-sigma|t|) cos(omega t) over R is 2*sigma/(sigma^2+omega^2);
        # the sine part is odd and integrates to 0. Subtract dc * g_sigma_bar(t)
        # with g_sigma_bar a unit-mass envelope of the same family.
        dc = (self.a * 2.0 * sigma / (sigma**2 + self.omega**2)).sum(-1)  # (in, out)
        sigma_bar = sigma.mean(-1, keepdim=False)  # (in, out)
        g_env = 0.5 * sigma_bar * torch.exp(-sigma_bar * m.squeeze(-1))  # (B, in, out)
        return psi - dc * g_env

    def forward_static(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, in_dim) -> (B, out_dim)."""
        s = self._scale()  # (in, out)
        t = (x.unsqueeze(-1) - self.mu) / s  # (B, in, out)
        wav = self.wavelet(t) / torch.sqrt(s)
        base = F.silu(x).unsqueeze(-1) * self.w_base
        edge = base + self.w_wav * wav  # (B, in, out)
        return edge.sum(dim=1)  # (B, out)

    # ------------------------------------------------------------------ #
    # Recurrent mode: the edge IS a discretized SSM (ZOH)                #
    # ------------------------------------------------------------------ #
    def _ssm_kernel(self, L: int) -> torch.Tensor:
        """SSM convolution kernel of length L: K_l = Re[g * Bbar * Abar^l].

        This is the impulse response of the *discretized* SSM. For small
        Delta it approximates Delta * psi(l * Delta), the sampled wavelet —
        the ZOH factor Bbar and the absence of the static-mode DC correction
        make the two related but not identical (see V1_README.md).

        Returns: (in_dim, out_dim, L) real kernel.
        """
        sigma = self._sigma()  # (in, out, N)
        lam = torch.complex(-sigma, self.omega)  # (in, out, N)
        s = self._scale().unsqueeze(-1)  # (in, out, 1) = Delta
        a_bar = torch.exp(lam * s)  # |a_bar| < 1: stable by construction
        b_bar = (a_bar - 1.0) / lam  # ZOH input matrix (B_k = 1)
        g = torch.complex(self.a, self.b)  # (in, out, N)

        lags = torch.arange(L, device=sigma.device, dtype=sigma.dtype)
        a_pow = a_bar.unsqueeze(-1) ** lags  # (in, out, N, L)
        return torch.einsum("ion,ionl->iol", g * b_bar, a_pow).real

    def forward_recurrent(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, L, in_dim) -> (B, L, out_dim).

        LTI system => y equals the causal convolution of u with the SSM
        kernel (the sampled wavelet). Computed via FFT in O(L log L);
        `forward_recurrent_loop` below is the O(L) sequential reference,
        kept for streaming/online use and for numerical verification.
        """
        B, L, _ = x.shape
        n_fft = 2 * L  # linear (non-circular) causal convolution
        k = self._ssm_kernel(L) * self.w_wav.unsqueeze(-1)  # (in, out, L)

        u_f = torch.fft.fft(x.transpose(1, 2), n=n_fft)  # (B, in, n_fft)
        k_f = torch.fft.fft(k, n=n_fft)  # (in, out, n_fft)
        y_f = torch.einsum("bin,ion->bon", u_f, k_f)  # sums over input edges
        y = torch.fft.ifft(y_f, n=n_fft)[:, :, :L].real.transpose(1, 2)  # (B, L, out)

        return y + x @ self.w_base  # + direct pass D

    def forward_recurrent_loop(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, L, in_dim) -> (B, L, out_dim). O(L) sequential scan.

        Reference implementation: numerically equivalent to
        `forward_recurrent` (FFT path) but online-capable (constant memory,
        can stream tokens one at a time).
        """
        B, L, _ = x.shape
        sigma = self._sigma()  # (in, out, N)
        lam = torch.complex(-sigma, self.omega)  # (in, out, N)
        s = self._scale().unsqueeze(-1)  # (in, out, 1) = Delta
        a_bar = torch.exp(lam * s)  # |a_bar| < 1: stable by construction
        b_bar = (a_bar - 1.0) / lam  # ZOH input matrix (B_k = 1)
        g = torch.complex(self.a, self.b)  # (in, out, N) readout C*B

        h = torch.zeros(
            B, self.in_dim, self.out_dim, self.n_states,
            dtype=torch.complex64, device=x.device,
        )
        a_bar = a_bar.unsqueeze(0)  # (1, in, out, N)
        b_bar = b_bar.unsqueeze(0)
        outs = []
        for n in range(L):
            u = x[:, n, :].unsqueeze(-1).unsqueeze(-1)  # (B, in, 1, 1)
            h = a_bar * h + b_bar * u
            y = torch.einsum("ion,bion->bio", g, h).real
            y = self.w_wav * y + self.w_base * x[:, n, :].unsqueeze(-1)  # edge weights
            outs.append(y)
        return torch.stack(outs, dim=1).sum(dim=2)  # (B, L, out): sum over input edges

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            return self.forward_static(x)
        if x.dim() == 3:
            return self.forward_recurrent(x)
        raise ValueError(f"expected 2D (static) or 3D (recurrent) input, got {x.dim()}D")

    @torch.no_grad()
    def spectral_signature(self) -> dict:
        """Per-edge dominant (sigma, omega) - useful for interpretability reports."""
        sigma = torch.exp(self.log_sigma)
        idx = self.a.abs().argmax(-1)
        gather = lambda t: t.gather(-1, idx.unsqueeze(-1)).squeeze(-1)
        return {"sigma": gather(sigma), "omega": gather(self.omega)}


class WaveletStateKAN(nn.Module):
    """Multi-layer WSKAN.

    Args:
        widths:   e.g. [in_dim, h1, h2, out_dim].
        n_states: SSM state dimension per edge.
    """

    def __init__(self, widths: list[int], n_states: int = 8):
        super().__init__()
        self.layers = nn.ModuleList(
            WaveletStateKANLayer(w_in, w_out, n_states)
            for w_in, w_out in zip(widths[:-1], widths[1:])
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x
