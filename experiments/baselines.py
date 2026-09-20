"""Baseline sequence models for the cross-family comparison.

Three families besides Mamba-2 and WSKAN:
  - TinyTransformerLM  (attention)
  - GatedConvLM        (dilated causal convolutions, WaveNet-style)
  - LSTMLM             (classic recurrent)

All expose the same .loss(idx) / .generate(bytes) interface as the WSKAN LMs.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class _Block(nn.Module):
    def __init__(self, d: int, heads: int):
        super().__init__()
        self.ln1 = nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.proj = nn.Linear(d, d, bias=False)
        self.ln2 = nn.LayerNorm(d)
        self.fc1 = nn.Linear(d, 4 * d)
        self.fc2 = nn.Linear(4 * d, d)
        self.heads = heads

    def forward(self, x):
        B, L, d = x.shape
        h = self.ln1(x)
        q, k, v = self.qkv(h).chunk(3, dim=-1)
        hd = d // self.heads
        q = q.view(B, L, self.heads, hd).transpose(1, 2)
        k = k.view(B, L, self.heads, hd).transpose(1, 2)
        v = v.view(B, L, self.heads, hd).transpose(1, 2)
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + self.proj(a.transpose(1, 2).reshape(B, L, d))
        h = self.ln2(x)
        return x + self.fc2(F.gelu(self.fc1(h)))


class TinyTransformerLM(nn.Module):
    def __init__(self, d_model: int = 64, n_layers: int = 2, block: int = 512):
        super().__init__()
        self.vocab_size = 256
        self.block = block
        self.tok_emb = nn.Embedding(256, d_model)
        self.pos_emb = nn.Embedding(block, d_model)
        nn.init.normal_(self.tok_emb.weight, std=0.02)
        nn.init.normal_(self.pos_emb.weight, std=0.02)
        heads = max(1, d_model // 32)
        while d_model % heads:
            heads -= 1
        self.blocks = nn.ModuleList(_Block(d_model, heads) for _ in range(n_layers))
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, 256, bias=False)
        self.head.weight = self.tok_emb.weight

    def forward(self, idx):
        L = idx.shape[1]
        pos = torch.arange(L, device=idx.device)
        x = self.tok_emb(idx) + self.pos_emb(pos)[None, :, :]
        for b in self.blocks:
            x = b(x)
        return self.head(self.norm(x))

    def loss(self, idx):
        logits = self(idx[:, :-1])
        return F.cross_entropy(logits.reshape(-1, 256), idx[:, 1:].reshape(-1))

    @torch.no_grad()
    def generate(self, prompt: bytes, max_new: int = 200, temperature: float = 1.0) -> bytes:
        device = self.tok_emb.weight.device
        idx = torch.tensor(list(prompt), dtype=torch.long, device=device).unsqueeze(0)
        idx = idx[:, -self.block:]
        for _ in range(max_new):
            logits = self(idx)[:, -1, :] / temperature
            nxt = torch.multinomial(F.softmax(logits, -1), 1)
            idx = torch.cat([idx, nxt], 1)[:, -self.block:]
        return bytes(idx[0].tolist())


class _ConvBlock(nn.Module):
    def __init__(self, d: int, dilation: int, kernel: int = 4):
        super().__init__()
        self.dilation = dilation
        self.pad = (kernel - 1) * dilation
        self.v = nn.Conv1d(d, d, kernel, groups=d, dilation=dilation)
        self.g = nn.Conv1d(d, d, kernel, groups=d, dilation=dilation)
        self.mix = nn.Linear(d, d)

    def forward(self, x):
        xt = x.transpose(1, 2)
        v = self.v(F.pad(xt, (self.pad, 0)))
        g = self.g(F.pad(xt, (self.pad, 0)))
        y = (torch.tanh(v) * torch.sigmoid(g)).transpose(1, 2)
        return x + self.mix(y)


class GatedConvLM(nn.Module):
    def __init__(self, d_model: int = 64, n_layers: int = 4):
        super().__init__()
        self.vocab_size = 256
        self.tok_emb = nn.Embedding(256, d_model)
        nn.init.normal_(self.tok_emb.weight, std=0.02)
        self.blocks = nn.ModuleList(_ConvBlock(d_model, 2**i) for i in range(n_layers))
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, 256, bias=False)
        self.head.weight = self.tok_emb.weight

    def forward(self, idx):
        x = self.tok_emb(idx)
        for b in self.blocks:
            x = b(x)
        return self.head(self.norm(x))

    def loss(self, idx):
        logits = self(idx[:, :-1])
        return F.cross_entropy(logits.reshape(-1, 256), idx[:, 1:].reshape(-1))

    @torch.no_grad()
    def generate(self, prompt: bytes, max_new: int = 200, temperature: float = 1.0) -> bytes:
        device = self.tok_emb.weight.device
        idx = torch.tensor(list(prompt), dtype=torch.long, device=device).unsqueeze(0)
        for _ in range(max_new):
            logits = self(idx)[:, -1, :] / temperature
            nxt = torch.multinomial(F.softmax(logits, -1), 1)
            idx = torch.cat([idx, nxt], 1)
        return bytes(idx[0].tolist())


class LSTMLM(nn.Module):
    def __init__(self, d_model: int = 64, n_layers: int = 2):
        super().__init__()
        self.vocab_size = 256
        self.tok_emb = nn.Embedding(256, d_model)
        nn.init.normal_(self.tok_emb.weight, std=0.02)
        self.lstm = nn.LSTM(d_model, d_model, n_layers, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, 256, bias=False)
        self.head.weight = self.tok_emb.weight

    def forward(self, idx):
        h, _ = self.lstm(self.tok_emb(idx))
        return self.head(self.norm(h))

    def loss(self, idx):
        logits = self(idx[:, :-1])
        return F.cross_entropy(logits.reshape(-1, 256), idx[:, 1:].reshape(-1))

    @torch.no_grad()
    def generate(self, prompt: bytes, max_new: int = 200, temperature: float = 1.0) -> bytes:
        device = self.tok_emb.weight.device
        idx = torch.tensor(list(prompt), dtype=torch.long, device=device).unsqueeze(0)
        state = None
        for _ in range(max_new):
            h, state = self.lstm(self.tok_emb(idx[:, -1:]), state)
            logits = self.head(self.norm(h))[:, -1, :] / temperature
            nxt = torch.multinomial(F.softmax(logits, -1), 1)
            idx = torch.cat([idx, nxt], 1)
        return bytes(idx[0].tolist())


def count_parameters(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())
