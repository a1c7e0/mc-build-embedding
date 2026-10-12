"""Tiny decoder-only GPT for plan token sequences (~0.8M params).

dim=128 / 4 layers / 4 heads, learned positional embeddings, causal mask.
CPU-trainable; the vocabulary is 39 tokens so the embedding/head are cheap.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GPTConfig:
    vocab: int = 39
    seq_len: int = 52
    dim: int = 128
    n_layer: int = 4
    n_head: int = 4
    dropout: float = 0.0


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: GPTConfig) -> None:
        super().__init__()
        self.n_head = cfg.n_head
        self.qkv = nn.Linear(cfg.dim, 3 * cfg.dim)
        self.proj = nn.Linear(cfg.dim, cfg.dim)
        self.dropout = cfg.dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        hd = C // self.n_head
        q = q.view(B, T, self.n_head, hd).transpose(1, 2)
        k = k.view(B, T, self.n_head, hd).transpose(1, 2)
        v = v.view(B, T, self.n_head, hd).transpose(1, 2)
        y = F.scaled_dot_product_attention(
            q, k, v, is_causal=True,
            dropout_p=self.dropout if self.training else 0.0)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(y)


class Block(nn.Module):
    def __init__(self, cfg: GPTConfig) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.dim)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.dim)
        self.mlp = nn.Sequential(
            nn.Linear(cfg.dim, 4 * cfg.dim), nn.GELU(),
            nn.Linear(4 * cfg.dim, cfg.dim), nn.Dropout(cfg.dropout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class GPT(nn.Module):
    def __init__(self, cfg: GPTConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab, cfg.dim)
        self.pos_emb = nn.Embedding(cfg.seq_len, cfg.dim)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layer))
        self.ln_f = nn.LayerNorm(cfg.dim)
        self.head = nn.Linear(cfg.dim, cfg.vocab, bias=False)
        self.head.weight = self.tok_emb.weight  # weight tying
        self.apply(self._init)

    @staticmethod
    def _init(m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor,
                targets: torch.Tensor | None = None):
        B, T = idx.shape
        assert T <= self.cfg.seq_len, f"seq len {T} > {self.cfg.seq_len}"
        pos = torch.arange(T, device=idx.device)
        x = self.drop(self.tok_emb(idx) + self.pos_emb(pos)[None])
        for blk in self.blocks:
            x = blk(x)
        logits = self.head(self.ln_f(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.shape[-1]),
                                   targets.reshape(-1), ignore_index=0)
        return logits, loss

    @torch.no_grad()
    def next_logits(self, idx: torch.Tensor) -> torch.Tensor:
        """Logits for the token after the current prefix (last position)."""
        logits, _ = self(idx)
        return logits[:, -1, :]

    def num_params(self) -> int:
        # count non-embedding params separately for reporting
        n = sum(p.numel() for p in self.parameters())
        return n

    def config_dict(self) -> dict:
        return asdict(self.cfg)
