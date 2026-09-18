"""V10 Wavelet-State-KAN: segmented (word-level) wavelet SSM.

Approximation of the V6/V8 selective scan driven by the measured structure
of the clock: Delta is effectively binary (boundary vs letter). V10 makes
that exact:

  - segment the byte stream at boundary bytes (space/newline) into words,
    packed into a (n_words, W) grid (long words split into W-sized chunks)
  - intra-word: LTI - every mode's wavelet kernel applied as a depthwise
    causal conv (cos and sin branches), separable: modes shared across
    channels, edges mix via g
  - inter-word carry: state entering word w is
        s_{w+1} = a_b * A_l^{W-1} * s_w + a_b * h_end_w
    (boundary decay a_b applied once per word crossing; the letter
    timescale A_l^j decays the carry inside the word)
  - readout per position from the packed grid; un-pack back

Approximations vs V8 (all measurable): continuous Delta -> per-channel
class means (dt_letter / dt_boundary); intra-word LTI. The boundary decay
(the clock's causal core) is preserved exactly.

Interpretability: computation is explicitly word-level - the clock's
invention becomes the architecture.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.V1_LM import WaveletStateKANLM
from models.V1_WSKAN import WaveletStateKANLayer
from models.V8_WSKAN import _scan_fwd

_LOG_MIN, _LOG_MAX = -4.0, 2.0
_WMAX = 16


class SegmentedWaveletLayer(WaveletStateKANLayer):
    """Word-segmented wavelet layer: intra-word conv + inter-word carry scan."""

    def __init__(self, in_dim: int, out_dim: int, n_states: int = 6, eps: float = 1e-2):
        super().__init__(in_dim, out_dim, n_states, eps)
        # V10 is per-channel: collapse the inherited per-edge (i,o,N) modes to (i,N)
        del self.log_sigma
        del self.omega
        self.log_sigma = nn.Parameter(torch.full((in_dim, n_states), math.log(0.5)))
        self.omega = nn.Parameter(
            math.pi * torch.arange(1, n_states + 1, dtype=torch.float32).repeat(in_dim, 1))
        self.log_dt_letter = nn.Parameter(torch.full((in_dim,), math.log(0.1)))
        self.log_dt_boundary = nn.Parameter(torch.full((in_dim,), math.log(0.3)))
        self.W_g = nn.Linear(in_dim, in_dim, bias=False)
        self.W_z = nn.Linear(in_dim, out_dim, bias=False)
        nn.init.normal_(self.W_g.weight, std=in_dim**-0.5)
        nn.init.normal_(self.W_z.weight, std=in_dim**-0.5)

    def _seg_index(self, idx: torch.Tensor):
        B, L = idx.shape
        nsub = 80                              # static cap: words per block (excess merges into the last)
        maxchunks = 2                          # words longer than 2*W are truncated (rare; documented)
        is_bnd = (idx == 32) | (idx == 10)
        wid = is_bnd.cumsum(-1).clamp(max=nsub - 1)
        pos = torch.arange(L, device=idx.device).unsqueeze(0).expand(B, L)
        big = torch.full((B, nsub + 1), L + 1, dtype=torch.long, device=idx.device)
        wstart = torch.scatter_reduce(big, 1, wid + 1, pos, reduce="amin")
        wstart = wstart[:, 1:]
        pos_in_word = pos - wstart.gather(1, wid)
        sub = (wid * maxchunks + (pos_in_word // _WMAX)).clamp(max=nsub * maxchunks - 1)
        pos2 = pos_in_word % _WMAX
        return sub, pos2, nsub * maxchunks

    def forward(self, x: torch.Tensor, idx: torch.Tensor | None = None) -> torch.Tensor:
        if x.dim() == 2:
            return self.forward_static(x)
        B, L, I = x.shape
        N = self.n_states
        sigma = torch.exp(torch.clamp(self.log_sigma, _LOG_MIN, _LOG_MAX))
        dt_l = torch.exp(self.log_dt_letter)
        dt_b = torch.exp(self.log_dt_boundary)
        lam = torch.complex(-sigma, self.omega)      # (i, N)
        lam_l = lam * dt_l.unsqueeze(-1)
        a_b = torch.exp(lam * dt_b.unsqueeze(-1))    # (i, N)

        u = x * F.silu(self.W_g(x))

        sub, pos2, nsub = self._seg_index(idx)
        W = _WMAX
        uu = torch.zeros(B, nsub, W, I, device=x.device)
        lin = (sub * W + pos2)
        uu.view(B, nsub * W, I).scatter_(1, lin.unsqueeze(-1).expand(-1, -1, I), u)

        # intra-word: depthwise conv with the 2N mode kernels
        lags = torch.arange(W, dtype=torch.float32, device=x.device)
        env = torch.exp((lam_l.real).unsqueeze(-1) * lags)          # (i, N, W)
        phase = (lam_l.imag).unsqueeze(-1) * lags
        wt = torch.cat([env * torch.cos(phase), env * torch.sin(phase)], dim=1)  # (i, 2N, W)
        kw = wt.reshape(I * 2 * N, 1, W)                            # per-channel kernels
        uw = uu.permute(0, 1, 3, 2).reshape(B * nsub, I, W)         # (B*nsub, I, W)
        conv_out = F.conv1d(F.pad(uw, (W - 1, 0)), kw, groups=I)    # (B*nsub, I*2N, W)
        conv_out = conv_out.reshape(B, nsub, I, 2 * N, W)
        h_local = torch.complex(conv_out[..., :N, :], conv_out[..., N:, :])  # (B, I, nsub, N, W)
        h_end = h_local[..., W - 1]                                 # (B, I, nsub, N)

        # inter-word carry scan (per channel, per mode)
        A_end = torch.exp(lam_l * (W - 1))                          # (i, N)
        a_seg = a_b * A_end
        u_seg = (a_b.unsqueeze(0).unsqueeze(0) * h_end).permute(0, 2, 3, 1)  # (B, I, N, nsub)
        cre, cim = _scan_fwd(
            a_seg.real.unsqueeze(0).unsqueeze(-1).expand(B, I, N, nsub),
            a_seg.imag.unsqueeze(0).unsqueeze(-1).expand(B, I, N, nsub),
            u_seg.real, u_seg.imag,
        )
        s = torch.complex(cre, cim).permute(0, 3, 1, 2)              # (B, nsub, I, N)
        s_in = torch.cat([torch.zeros(B, 1, I, N, dtype=torch.complex64, device=x.device),
                          s[:, :-1]], dim=1)                      # state entering word w

        # carry decays into the word by the letter prefix A_l^j
        Al = torch.exp(lam_l.unsqueeze(-1) * lags)                   # (i, N, W)
        h_full = h_local + s_in.unsqueeze(-1) * Al.unsqueeze(0).unsqueeze(0)

        # readout + unpack
        yw = torch.einsum("bwikj,iok->bwjo", h_full.real, self.a) - \
             torch.einsum("bwikj,iok->bwjo", h_full.imag, self.b)
        yw = yw.reshape(B, nsub * W, self.out_dim)
        y = torch.gather(yw, 1, lin.unsqueeze(-1).expand(-1, -1, self.out_dim))
        y = y * F.silu(self.W_z(x)) + x @ self.w_base
        return y


class WaveletStateKANLMV10(WaveletStateKANLM):
    def __init__(self, vocab_size: int = 256, d_model: int = 32, n_layers: int = 3,
                 n_states: int = 6):
        super().__init__(vocab_size, d_model, n_layers, n_states)
        self.layers = nn.ModuleList(
            SegmentedWaveletLayer(d_model, d_model, n_states) for _ in range(n_layers)
        )
        self.prenorms = nn.ModuleList(nn.LayerNorm(d_model) for _ in range(n_layers))

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        x = self.tok_emb(idx)
        for norm, layer in zip(self.prenorms, self.layers):
            x = x + layer(norm(x), idx)
        return self.head(self.norm(x))
