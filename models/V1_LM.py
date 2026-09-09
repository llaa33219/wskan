"""V1 Wavelet-State-KAN language model wrapper.

Byte-level LM: token embedding -> stacked WSKAN layers in recurrent mode
(FFT convolution path) with residual connections -> tied LM head.

The "weights" of a KAN are the edge *functions*; here each edge function is
parameterized by its SSM wavelet (lambda = -sigma + i*omega, gain g = a + i*b,
scale s, translation mu) plus edge weights (w_base, w_wav). A checkpoint of
this model is therefore a saved bundle of edge functions, not a weight matrix.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.V1_WSKAN import WaveletStateKANLayer


class WaveletStateKANLM(nn.Module):
    """Byte-level language model built from V1 WSKAN recurrent layers.

    Args:
        vocab_size: byte-level vocab (256).
        d_model:    node width of every KAN layer.
        n_layers:   number of WSKAN layers.
        n_states:   SSM state dimension per edge.

    Notes:
        - Residual connections around each KAN layer and a final LayerNorm are
          used for trainability; this is a pragmatic deviation from a "pure"
          KAN and is documented as such.
        - The LM head is tied to the token embedding to stay within the
          ~100k parameter budget.
    """

    def __init__(
        self,
        vocab_size: int = 256,
        d_model: int = 32,
        n_layers: int = 3,
        n_states: int = 6,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        nn.init.normal_(self.tok_emb.weight, mean=0.0, std=0.02)  # GPT-style init
        self.layers = nn.ModuleList(
            WaveletStateKANLayer(d_model, d_model, n_states) for _ in range(n_layers)
        )
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self.head.weight = self.tok_emb.weight  # tied

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        """idx: (B, L) int64 -> logits (B, L, vocab)."""
        x = self.tok_emb(idx)
        for layer in self.layers:
            x = x + layer(x)  # recurrent mode, residual
        return self.head(self.norm(x))

    def loss(self, idx: torch.Tensor) -> torch.Tensor:
        """Next-byte cross-entropy on the sequence itself."""
        logits = self(idx[:, :-1])
        return F.cross_entropy(
            logits.reshape(-1, self.vocab_size), idx[:, 1:].reshape(-1)
        )

    @torch.no_grad()
    def generate(
        self, prompt: bytes, max_new: int = 200, temperature: float = 1.0
    ) -> bytes:
        """Greedy/sampled byte-level generation from a byte-string prompt."""
        device = self.tok_emb.weight.device
        idx = torch.tensor(list(prompt), dtype=torch.long, device=device).unsqueeze(0)
        for _ in range(max_new):
            logits = self(idx)[:, -1, :] / temperature
            probs = F.softmax(logits, dim=-1)
            nxt = torch.multinomial(probs, 1)
            idx = torch.cat([idx, nxt], dim=1)
        return bytes(idx[0].tolist())


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
