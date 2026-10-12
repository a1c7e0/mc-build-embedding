"""Benchmark: GPT training speed on 4096-token sequences (64^3 latent).

Runs 5 steps of a small GPT on real 64^3 tokens to measure feasibility.
If < 10s/step, flat prior is viable for overnight; if > 20s/step,
hierarchical generation is needed.

usage: python src/prior64_bench.py [tokens_file]
"""
from __future__ import annotations

import sys
import time

import numpy as np
import torch
import torch.nn as nn

VOCAB = 1024
SEQ = 4096  # 16^3 latent


class GPT(nn.Module):
    def __init__(self, dim=256, layers=6, heads=8):
        super().__init__()
        self.tok = nn.Embedding(VOCAB, dim)
        self.pos = nn.Parameter(torch.zeros(1, SEQ, dim))
        self.blocks = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(dim, heads, dim * 4, batch_first=True,
                                       dropout=0.1),
            layers)
        self.ln = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, VOCAB)

    def forward(self, x):
        b, s = x.shape
        h = self.tok(x) + self.pos[:, :s]
        h = self.blocks(h)
        return self.head(self.ln(h))


def main() -> None:
    f = sys.argv[1] if len(sys.argv) > 1 else 'out_lib/tokens64_vq_data2.npz'
    tokens = np.load(f)['tokens']
    print(f'[bench] tokens {tokens.shape}')
    dev = 'cpu'
    m = GPT().to(dev)
    opt = torch.optim.AdamW(m.parameters(), lr=3e-4)
    print(f'[bench] params {sum(p.numel() for p in m.parameters())/1e6:.1f}M')
    print(f'[bench] seq={SEQ} dim=256 layers=6 heads=8')

    T = torch.from_numpy(tokens[:8]).to(dev)  # batch 8
    for step in range(5):
        t0 = time.time()
        inp, tgt = T[:, :-1], T[:, 1:]
        logits = m(inp)
        loss = nn.functional.cross_entropy(
            logits.reshape(-1, VOCAB), tgt.reshape(-1))
        opt.zero_grad()
        loss.backward()
        opt.step()
        dt = time.time() - t0
        print(f'[bench] step {step} loss {loss.item():.3f} {dt:.1f}s')

    avg = sum(float(l.split()[-1][:-1]) for l in
              open('/dev/null')) or 0  # dummy
    print(f'[bench] done')


if __name__ == '__main__':
    main()
