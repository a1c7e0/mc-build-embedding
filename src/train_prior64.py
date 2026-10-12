"""GPT prior over 64^3 VQ tokens (4096-token sequences).

Flat transformer trained autoregressively on the 16^3 latent grid of
the 64^3 VQ-VAE codebook. Uses causal attention over the full 4096
token sequence. Slower than 32^3 prior (512 tokens) but tests
feasibility of full-scale generation at this resolution.

env: P_DATA  tokens file (npz with 'tokens' key) [out_lib/tokens64.npz]
     P_CKPT  checkpoint path [out_lib/prior64_ckpt.pt]
     P_STEPS training steps [15000]
     P_DIM   model dim [256]
     P_LAYERS number of layers [8]
     PRETRAIN 1=resume [1]
"""
from __future__ import annotations

import math
import os
import time

import numpy as np
import torch
import torch.nn as nn

VOCAB = 1024
SEQ = int(os.environ.get('P_SEQ', '4096'))  # 16^3 latent grid


class CausalSelfAttention(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.h = heads
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        b, s, d = x.shape
        qkv = self.qkv(x).reshape(b, s, 3, self.h, d // self.h)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(d // self.h)
        mask = torch.triu(torch.ones(s, s, device=x.device), 1).bool()
        att.masked_fill_(mask, float('-inf'))
        att = att.softmax(-1)
        out = (att @ v).transpose(1, 2).reshape(b, s, d)
        return self.proj(out)


class Block(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.attn = CausalSelfAttention(dim, heads)
        self.ln2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4), nn.GELU(), nn.Linear(dim * 4, dim))

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class GPT(nn.Module):
    def __init__(self, dim=256, layers=8, heads=8):
        super().__init__()
        self.tok = nn.Embedding(VOCAB, dim)
        self.pos = nn.Parameter(torch.zeros(1, SEQ, dim))
        self.blocks = nn.ModuleList([Block(dim, heads) for _ in range(layers)])
        self.ln = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, VOCAB)

    def forward(self, x):
        b, s = x.shape
        h = self.tok(x) + self.pos[:, :s]
        for blk in self.blocks:
            h = blk(h)
        return self.head(self.ln(h))


def main() -> None:
    dev = 'cpu'
    data = np.load(os.environ.get('P_DATA', 'out_lib/tokens64.npz'))['tokens']
    ckpt_path = os.environ.get('P_CKPT', 'out_lib/prior64_ckpt.pt')
    steps = int(os.environ.get('P_STEPS', '15000'))
    dim = int(os.environ.get('P_DIM', '256'))
    layers = int(os.environ.get('P_LAYERS', '8'))
    pretrain = os.environ.get('PRETRAIN', '1') == '1'
    print(f'[prior64] tokens {data.shape} dim={dim} layers={layers} '
          f'steps={steps} pretrain={pretrain}')

    m = GPT(dim=dim, layers=layers, heads=8).to(dev)
    print(f'[prior64] params {sum(p.numel() for p in m.parameters())/1e6:.1f}M')

    opt = torch.optim.AdamW(m.parameters(), lr=3e-4)
    start_step = 0
    if pretrain and os.path.exists(ckpt_path):
        sd = torch.load(ckpt_path, map_location=dev, weights_only=True)
        m.load_state_dict(sd['model'])
        if 'step' in sd:
            opt.load_state_dict(sd['opt'])
            start_step = int(sd['step'])
        print(f'[prior64] resumed from step {start_step}')

    step = start_step
    t0 = time.time()
    while step < steps:
        idx = np.random.randint(0, len(data), size=8)
        T = torch.from_numpy(data[idx].astype(np.int64)).to(dev)
        inp, tgt = T[:, :-1], T[:, 1:]
        logits = m(inp)
        loss = nn.functional.cross_entropy(
            logits.reshape(-1, VOCAB), tgt.reshape(-1))
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 10 == 0:
            dt = time.time() - t0
            print(f'[prior64] step {step} loss {loss.item():.3f} ({dt:.0f}s)')
            t0 = time.time()
        if step % 500 == 0 and step > 0:
            torch.save({'model': m.state_dict(), 'opt': opt.state_dict(),
                        'step': step}, ckpt_path)
        step += 1

    torch.save({'model': m.state_dict(), 'opt': opt.state_dict(),
                'step': step}, ckpt_path)
    print(f'[prior64] done at step {step}')


if __name__ == '__main__':
    main()
