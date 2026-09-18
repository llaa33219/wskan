"""V9 Wavelet-State-KAN: wavelet-generated FIR convolution (approximation tier).

The scan is replaced by a short causal convolution whose kernels ARE the edge
wavelets evaluated at integer lags:

    K_io[l] = psi_io(l) = sum_k Re[g_iok * exp(lambda_tilde_ik * l)],
    l = 0..K-1  (nominal Delta = 1)

    y_o = (K * u)_o * SiLU(W_z x) + x @ w_base,   u = x * SiLU(W_g x)

GPU story: one tensor-core conv1d per layer - no scan, no recurrence, no
chunk loops.

Approximation honesty: this drops the input-dependent Delta warping (the word
clock) and the long-memory tail beyond K lags. At the 100k regime kernels are
near-delta (half-life ~1-5 tokens), so truncation error is tiny; the clock
loss is the real cost (measured, see experiments/V9 report).

Interpretability: maximal - the convolution kernel literally is the edge
function; nothing to probe.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.V1_LM import WaveletStateKANLM
from models.V1_WSKAN import WaveletStateKANLayer

_LOG_MIN, _LOG_MAX = -4.0, 2.0


class WaveletConvKANLayer(WaveletStateKANLayer):
    """V1 wavelet machinery evaluated into a short causal conv kernel."""

    def __init__(self, in_dim: int, out_dim: int, n_states: int = 6, kernel_len: int = 32):
        super().__init__(in_dim, out_dim, n_states)
        self.kernel_len = kernel_len
        self.W_g = nn.Linear(in_dim, in_dim, bias=False)
        self.W_z = nn.Linear(in_dim, out_dim, bias=False)
        nn.init.normal_(self.W_g.weight, std=in_dim**-0.5)
        nn.init.normal_(self.W_z.weight, std=in_dim**-0.5)

    def conv_weight(self) -> torch.Tensor:
        """(out, in, K) causal conv kernel = the wavelet at integer lags."""
        K = self.kernel_len
        lags = torch.arange(K, dtype=torch.float32, device=self.a.device)
        t = lags.view(K, 1, 1)                      # (K, in, out)
        Kmat = self.wavelet(t)                       # (K, in, out)
        s = self._scale()                            # (in, out)
        Kmat = Kmat / torch.sqrt(s) * self.w_wav
        return Kmat.permute(2, 1, 0).contiguous()    # (out, in, K)

    def forward(self, x: torch.Tensor, idx: torch.Tensor | None = None) -> torch.Tensor:
        if x.dim() == 2:
            return self.forward_static(x)
        B, L, I = x.shape
        u = x * F.silu(self.W_g(x))                  # input gate
        w = self.conv_weight()                       # (O, I, K)
        ut = F.pad(u.transpose(1, 2), (self.kernel_len - 1, 0))
        y = F.conv1d(ut, w).transpose(1, 2)          # (B, L, O)
        return y * F.silu(self.W_z(x)) + x @ self.w_base


class WaveletStateKANLMV9(WaveletStateKANLM):
    """V9 LM: V1 LM wrapper with wavelet-conv layers (pre-norm residuals)."""

    def __init__(self, vocab_size: int = 256, d_model: int = 32, n_layers: int = 3,
                 n_states: int = 6, kernel_len: int = 32):
        super().__init__(vocab_size, d_model, n_layers, n_states)
        self.layers = nn.ModuleList(
            WaveletConvKANLayer(d_model, d_model, n_states, kernel_len)
            for _ in range(n_layers)
        )
        self.prenorms = nn.ModuleList(nn.LayerNorm(d_model) for _ in range(n_layers))

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        x = self.tok_emb(idx)
        for norm, layer in zip(self.prenorms, self.layers):
            x = x + layer(norm(x))
        return self.head(self.norm(x))
