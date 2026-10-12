"""Autoregressive prior over VQ codes — the generative layer.

Extracts code indices for every library building (frozen VQ-VAE encoder),
trains a small GPT (causal, raster order over the 8^3 code grid), and can
sample new buildings: GPT -> code grid -> VQ decoder -> voxels.

usage: python src/train_prior.py [steps]        # extract + train
       python src/train_prior.py sample [n]     # sample n buildings
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(__file__))
from train_vq import VQVAE  # noqa: E402

TOKENS = os.environ.get("PRIOR_TOKENS", "out_lib/vq_tokens.npz")
CKPT = os.environ.get("PRIOR_CKPT", "out_lib/prior_ckpt.pt")
LOG = os.environ.get("PRIOR_LOG", "docs/prior_train.log")
SEQ = 512          # 8^3 latent codes per building
VOCAB = 1024 + 8   # codes + 7 styles + unknown style token
STY_OFF = 1024
STYLES = ["modern", "medieval", "asian", "scifi", "western", "nautical",
          "other"]


class Block(nn.Module):
    def __init__(self, dim: int, heads: int):
        super().__init__()
        self.heads = heads
        self.ln1 = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.ln2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, dim * 4), nn.GELU(),
                                 nn.Linear(dim * 4, dim))

    def forward(self, x):  # noqa: ANN001
        b, n, d = x.shape
        q, k, v = self.qkv(self.ln1(x)).chunk(3, -1)
        h = d // self.heads
        q = q.view(b, n, self.heads, h).transpose(1, 2)
        k = k.view(b, n, self.heads, h).transpose(1, 2)
        v = v.view(b, n, self.heads, h).transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).reshape(b, n, d)
        x = x + self.proj(y)
        return x + self.mlp(self.ln2(x))


class GPT(nn.Module):
    def __init__(self, dim=None, layers=None, heads=8, max_len=SEQ):
        super().__init__()
        dim = dim or int(os.environ.get("P_DIM", 192))
        layers = layers or int(os.environ.get("P_LAYERS", 4))
        self.tok = nn.Embedding(VOCAB, dim)
        self.pos = nn.Embedding(max_len, dim)
        self.back = nn.ModuleList(Block(dim, heads) for _ in range(layers))
        self.ln = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, VOCAB)

    def forward(self, idx):  # noqa: ANN001
        n = idx.shape[1]
        h = self.tok(idx) + self.pos(torch.arange(n, device=idx.device))[None]
        for blk in self.back:
            h = blk(h)
        return self.head(self.ln(h))


def get_vq(dev):  # noqa: ANN001
    m = VQVAE().to(dev)
    sd = torch.load("out_lib/vq_ckpt.pt", map_location=dev, weights_only=True)
    m.load_state_dict(sd["model"])
    m.eval()
    return m


def extract_tokens(dev):  # noqa: ANN001
    if os.path.exists(TOKENS):
        d = np.load(TOKENS)
        if "styles" in d:
            return d["tokens"], d["styles"]
        # legacy cache without styles: attach labels without re-encoding
        tokens, idx = d["tokens"], d["idx"]
        vtypes = json.load(open("out_lib/vision_types.json"))
        styles = np.array([
            STYLES.index(vtypes.get(str(i), {}).get("style", "zz"))
            if vtypes.get(str(i), {}).get("style") in STYLES else 7
            for i in idx], dtype=np.int32)
        np.savez_compressed(TOKENS, tokens=tokens, idx=idx, styles=styles)
        return tokens, styles
    print("[prior] extracting code tokens for the library", flush=True)
    data = np.load("out_lib/vq_data.npz")
    X, idx = data["X"], data["idx"]
    vtypes = json.load(open("out_lib/vision_types.json"))
    vq = get_vq(dev)
    out = []
    with torch.no_grad():
        for i in range(0, len(X), 16):
            xb = torch.from_numpy(np.moveaxis(
                np.eye(6, dtype=np.float32)[X[i:i + 16]], -1, 1).copy()
            ).to(dev)
            z = vq.enc(xb)
            zl = z.permute(0, 2, 3, 4, 1).reshape(z.shape[0], -1, z.shape[1])
            _, codes, _ = vq.vq(zl)
            out.append(codes.cpu().numpy())
    tokens = np.concatenate(out).astype(np.int32)
    styles = np.array([
        STYLES.index(vtypes.get(str(i), {}).get("style", "zz"))
        if vtypes.get(str(i), {}).get("style") in STYLES else 7
        for i in idx], dtype=np.int32)
    np.savez_compressed(TOKENS, tokens=tokens, idx=idx, styles=styles)
    print(f"[prior] tokens: {tokens.shape}", flush=True)
    return tokens, styles


def main() -> None:
    dev = "cpu"
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "5")))
    mode = sys.argv[1] if len(sys.argv) > 1 else "train"
    if mode == "sample":
        n = int(sys.argv[2]) if len(sys.argv) > 2 else 8
        want = sys.argv[3] if len(sys.argv) > 3 else "other"
        sty = STY_OFF + (STYLES.index(want) if want in STYLES else 7)
        vq = get_vq(dev)
        gpt = GPT().to(dev)
        sd = torch.load(CKPT, map_location=dev, weights_only=True)
        gpt.load_state_dict(sd["model"])
        gpt.eval()
        from render import render2
        os.makedirs("out_lib/prior_samples", exist_ok=True)
        CLASS_TO_ID = {0: 0, 1: 1, 2: 5, 3: 20, 4: 18, 5: 35}
        for k in range(n):
            cur = torch.tensor([[sty]], dtype=torch.long)
            for t in range(SEQ):
                logits = gpt(cur)[0, -1] / 0.9
                nxt = torch.multinomial(torch.softmax(logits, -1), 1)
                cur = torch.cat([cur, nxt[None]], dim=1)
            codes = cur[0, 1:SEQ + 1].reshape(1, 8, 8, 8)
            with torch.no_grad():
                cb = getattr(getattr(vq.vq, "_codebook", None),
                             "embed", None)
                if cb is None:
                    cb = vq.vq.codebook.embed
                if getattr(cb, "dim", lambda: 2)() == 3:
                    cb = cb[0]          # (heads, vocab, dim) -> (vocab, dim)
                q = cb[codes.to(dev)].permute(0, 4, 1, 2, 3)
                rec = vq.dec(q).argmax(1)[0].numpy()
            g = np.zeros(rec.shape, dtype=np.uint16)
            for c, i in CLASS_TO_ID.items():
                g[rec == c] = i
            render2(g, max_px=280).save(
                f"out_lib/prior_samples/sample_{want}_{k}.png")
        print(f"[prior] sampled {n} -> out_lib/prior_samples/", flush=True)
        return

    tokens, styles = extract_tokens(dev)
    N = len(tokens)
    gpt = GPT().to(dev)
    opt = torch.optim.AdamW(gpt.parameters(), lr=3e-4)
    start = 0
    if os.path.exists(CKPT):
        try:
            sd = torch.load(CKPT, map_location=dev, weights_only=True)
            gpt.load_state_dict(sd["model"])
            opt.load_state_dict(sd["opt"])
            start = sd["step"]
            print(f"[prior] resumed from step {start}", flush=True)
        except RuntimeError:
            print("[prior] ckpt incompatible (v2 arch), fresh start", flush=True)
    steps = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() \
        else 20000
    log = open(LOG, "a", encoding="utf-8")
    rng = np.random.default_rng(1)
    t0 = time.time()
    gpt.train()
    for step in range(start, steps):
        # class-balanced style sampling: minority styles get boosted draw
        w = 1.0 / np.maximum(np.bincount(styles, minlength=8), 1)
        p = w[styles] / w[styles].sum()
        bi = rng.choice(N, 16, replace=True, p=p)
        tok = torch.from_numpy(tokens[bi].astype(np.int64)).to(dev)
        sty = torch.from_numpy(
            (STY_OFF + styles[bi]).astype(np.int64)).to(dev)
        seq = torch.cat([sty[:, None], tok], dim=1)   # style token first
        logits = gpt(seq[:, :-1])
        loss = F.cross_entropy(logits.reshape(-1, VOCAB), seq[:, 1:].reshape(-1))
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 20 == 0:
            msg = (f"[prior] step {step} loss {loss.item():.3f} "
                   f"({time.time()-t0:.0f}s)")
            print(msg, flush=True)
            log.write(msg + "\n")
            log.flush()
        if step % 200 == 0 and step > start:
            torch.save({"model": gpt.state_dict(), "opt": opt.state_dict(),
                        "step": step + 1}, CKPT)
    torch.save({"model": gpt.state_dict(), "opt": opt.state_dict(),
                "step": steps}, CKPT)
    print(f"[prior] done, saved {CKPT}", flush=True)


if __name__ == "__main__":
    main()
