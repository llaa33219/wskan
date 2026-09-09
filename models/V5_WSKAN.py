"""V5 Wavelet-State-KAN: short convolution + geometric (octave) wavelet ladder.

Two evidence-driven deltas over V4, each behind its own flag for ablation:

1. Short causal depthwise conv (kernel 4) on the layer input before the
   selective scan. Mamba-2 has this and WSKAN never did; the 100k-scale runs
   showed wavelet modes pinned into near-delta kernels, i.e. forced to act as
   a poor man's conv. The conv takes over local structure so the wavelet
   modes can specialize. Params: d*(4+1) per layer - negligible.
   (flag: use_conv)

2. Geometric (octave-spirit) omega ladder: omega_k = pi * r^(k-1) with
   r = N^(1/(N-1))... concretely geometric spacing across the same range as
   the S4D-Lin linear ladder (pi .. pi*N), instead of pi*k. Wavelet
   multiresolution is octave-structured, not linear-structured; this tests
   whether the *ladder shape* matters, complementing the earlier omega=0
   ablation (which tested oscillation itself). (flag: omega_init)

Everything else is inherited from V4 (freed sigma, per-mode dt, ZOH write).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.V1_LM import WaveletStateKANLM
from models.V4_WSKAN import FreeWaveletStateKANLayer


class ConvWaveletStateKANLayer(FreeWaveletStateKANLayer):
    """V4 layer + causal depthwise short conv + geometric omega ladder option."""

    def __init__(self, in_dim: int, out_dim: int, n_states: int = 6, chunk_size: int = 16,
                 eps: float = 1e-2, oscillatory: bool = True, grad_checkpoint: bool = False,
                 compile_chunk: bool = False, dt_rank: int = 8,
                 use_conv: bool = True, omega_init: str = "geometric"):
        super().__init__(in_dim, out_dim, n_states, chunk_size, eps, oscillatory,
                         grad_checkpoint, compile_chunk, dt_rank)
        self.use_conv = use_conv
        if use_conv:
            self.short_conv = nn.Conv1d(in_dim, in_dim, kernel_size=4, groups=in_dim)
        if oscillatory and omega_init == "geometric":
            r = n_states ** (1.0 / (n_states - 1))  # pi .. pi*n_states, geometric
            with torch.no_grad():
                self.omega.copy_(
                    math.pi * torch.tensor([r ** k for k in range(n_states)]).repeat(in_dim, 1)
                )

    def _pre(self, x: torch.Tensor) -> torch.Tensor:
        if not self.use_conv:
            return x
        xt = F.pad(x.transpose(1, 2), (3, 0))  # causal left pad
        return F.silu(self.short_conv(xt).transpose(1, 2))

    def forward_recurrent(self, x: torch.Tensor) -> torch.Tensor:
        return super().forward_recurrent(self._pre(x))

    def forward_recurrent_loop(self, x: torch.Tensor) -> torch.Tensor:
        return super().forward_recurrent_loop(self._pre(x))


class WaveletStateKANLMV5(WaveletStateKANLM):
    """V4 LM with V5 conv/ladder layers (pre-norm residuals kept)."""

    def __init__(self, vocab_size: int = 256, d_model: int = 32, n_layers: int = 3,
                 n_states: int = 6, oscillatory: bool = True, chunk_size: int = 16,
                 grad_checkpoint: bool = False, compile_chunk: bool = False,
                 use_conv: bool = True, omega_init: str = "geometric"):
        super().__init__(vocab_size, d_model, n_layers, n_states)
        self.layers = nn.ModuleList(
            ConvWaveletStateKANLayer(d_model, d_model, n_states, chunk_size=chunk_size,
                                     oscillatory=oscillatory, grad_checkpoint=grad_checkpoint,
                                     compile_chunk=compile_chunk, use_conv=use_conv,
                                     omega_init=omega_init)
            for _ in range(n_layers)
        )
        self.prenorms = nn.ModuleList(nn.LayerNorm(d_model) for _ in range(n_layers))

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        x = self.tok_emb(idx)
        for norm, layer in zip(self.prenorms, self.layers):
            x = x + layer(norm(x))
        return self.head(self.norm(x))
