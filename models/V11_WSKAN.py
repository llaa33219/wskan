"""V11 Wavelet-State-KAN: fused-kernel scan (V8 math, single-kernel speed).

V11 = V8 with the scan computed by the fused Triton kernel
(models/V11_kernel.py): one associative-scan kernel per layer, coalesced
tile loads, flip+conjugate trick for the backward scan. Same parameters,
checkpoint-compatible with wskan7bc.
"""

from __future__ import annotations

import torch

from models.V11_kernel import v11_scan_bwd, v11_scan_fwd
from models.V8_WSKAN import FastWaveletStateKANLayer, WaveletStateKANLMV8


class _FusedScan(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a_re, a_im, u_re, u_im):
        # inputs arrive in kernel layout (B, I, L, N)
        a_re = a_re.contiguous(); a_im = a_im.contiguous()
        u_re = u_re.contiguous(); u_im = u_im.contiguous()
        h_re, h_im = v11_scan_fwd(a_re, a_im, u_re, u_im)
        ctx.save_for_backward(a_re, a_im, h_re, h_im)
        return h_re.permute(0, 1, 3, 2), h_im.permute(0, 1, 3, 2)  # out: (B, I, N, L)

    @staticmethod
    def backward(ctx, G_re, G_im):
        a_re, a_im, h_re, h_im = ctx.saved_tensors
        # G arrives in the forward-output layout (B, I, N, L); kernel wants (B, I, L, N)
        Gr = G_re.permute(0, 1, 3, 2).contiguous()
        Gi = G_im.permute(0, 1, 3, 2).contiguous()
        da_re, da_im, du_re, du_im = v11_scan_bwd(a_re, a_im, h_re, h_im, Gr, Gi)
        return da_re, da_im, du_re, du_im  # kernel layout = input layout (B, I, L, N)


class FusedWaveletStateKANLayer(FastWaveletStateKANLayer):
    def __init__(self, *args, bf16_scan: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        self._bf16_scan = bf16_scan

    def forward_recurrent(self, x: torch.Tensor, idx: torch.Tensor | None = None) -> torch.Tensor:
        Bsz, L, I, N = x.shape[0], x.shape[1], self.in_dim, self.n_states
        self._compose_g()
        sigma = torch.exp(torch.clamp(self.log_sigma, self.log_sigma_min, self.log_sigma_max))
        rho = self._rho()
        sig_eff = sigma * rho
        om_eff = self.omega * rho
        dt = self._compute_dt(x)
        Bn = self._compute_B(x, idx)
        Cn = self._compute_C(x, idx)
        u = Bn * x.unsqueeze(-1) * dt.unsqueeze(-1)

        decay = torch.exp(-sig_eff * dt.unsqueeze(-1))
        phase = om_eff * dt.unsqueeze(-1)
        a_re = (decay * torch.cos(phase)).permute(0, 2, 1, 3).contiguous()  # (B, I, L, N)
        a_im = (decay * torch.sin(phase)).permute(0, 2, 1, 3).contiguous()
        u_re = u.permute(0, 2, 1, 3).contiguous()
        u_im = torch.zeros_like(u_re)

        if self._bf16_scan:
            a_re, a_im, u_re, u_im = (a_re.bfloat16(), a_im.bfloat16(),
                                      u_re.bfloat16(), u_im.bfloat16())
            h_re, h_im = _FusedScan.apply(a_re, a_im, u_re, u_im)
            h_re, h_im = h_re.float(), h_im.float()
        else:
            h_re, h_im = _FusedScan.apply(a_re, a_im, u_re, u_im)  # back to (B, I, N, L)

        read_re = Cn.permute(0, 2, 3, 1) * h_re
        read_im = Cn.permute(0, 2, 3, 1) * h_im
        y = torch.einsum("bikl,iok->bol", read_re, self.a) - torch.einsum(
            "bikl,iok->bol", read_im, self.b)
        y = y.permute(0, 2, 1)
        y = y * self._gate(x)
        return y + x @ self.w_base


class WaveletStateKANLMV11(WaveletStateKANLMV8):
    """V8 LM with the fused kernel scan."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.layers = torch.nn.ModuleList(
            FusedWaveletStateKANLayer(
                self.d_model, self.d_model, layer.n_states,
                use_feature_bc=True, bc_rank=32, wz_diag=False, g_rank=None,
                bf16_scan=kwargs.get("bf16_scan", False),
            )
            for layer in self.layers
        )
