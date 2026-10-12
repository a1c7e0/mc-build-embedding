"""Encode ProcTHOR voxels with the v0.2 VQ and merge into the prior corpus.

Reads out_lib/vq_data_pro.npz (ProcTHOR houses, styles=0 'modern'), encodes
to 512-token sequences with the VQ encoder, appends to the existing
vq_tokens.npz corpus (library buildings) -> vq_tokens_all.npz.

usage: python src/make_tokens_v3.py
"""
from __future__ import annotations

import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from train_vq import VQVAE  # noqa: E402


def main() -> None:
    dev = "cpu"
    m = VQVAE().to(dev)
    sd = torch.load("out_lib/vq_ckpt2.pt", map_location=dev, weights_only=True)
    m.load_state_dict(sd["model"])
    m.eval()

    def encode(X):  # noqa: ANN001
        out = []
        with torch.no_grad():
            for i in range(0, len(X), 8):
                xb = torch.from_numpy(np.moveaxis(
                    np.eye(6, dtype=np.float32)[X[i:i + 8]], -1, 1).copy()
                ).to(dev)
                z = m.enc(xb)
                zl = z.permute(0, 2, 3, 4, 1).reshape(z.shape[0], -1,
                                                      z.shape[1])
                _, codes, _ = m.vq(zl)
                out.append(codes.cpu().numpy())
        return np.concatenate(out).astype(np.int32)

    pro = np.load("out_lib/vq_data_pro.npz")
    pro_tok = encode(pro["X"])
    base = np.load("out_lib/vq_tokens.npz")
    tokens = np.concatenate([base["tokens"], pro_tok])
    styles = np.concatenate([base["styles"], pro["styles"]])
    np.savez_compressed("out_lib/vq_tokens_all.npz", tokens=tokens,
                        styles=styles)
    print(f"[make-tokens-v3] base {base['tokens'].shape} + procthor "
          f"{pro_tok.shape} -> {tokens.shape}", flush=True)


if __name__ == "__main__":
    main()
