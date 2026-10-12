"""Plan loading, house-group split, and token serialization.

Serialization (identical rule for training and sampling):

    [BOS] (type x0 z0 x1 z1) * n_rooms [EOS] [PAD ...]

Room order: stable sort by (type rank, program index). The rank is the
position of the type in TYPES; the program index is the room's position in
the input program list. This rule is computable from the program alone --
unlike e.g. area-descending order -- which is what makes type-conditioned
generation possible: slot order is fixed before any coordinate exists.

Coordinates: each house is normalized per axis to [0,1] using the bounding
box of its room rects, then quantized to NBINS=32 bins on a margin grid
[BIN_LO, BIN_HI] = [-0.1, 1.1]. The 10% margin means out-of-canvas
coordinates ARE representable tokens, so the out-of-bounds metric is a live
quantity rather than a constant zero.

Vocabulary (flat, 39 tokens):
    0 PAD, 1 BOS, 2 EOS, 3..6 type tokens (TYPES order), 7..38 coord bins.
Coordinate bins are shared across the four coordinate channels; the channel
is recoverable from the sequence position (pos-1) mod 5.
"""
from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass, field

import torch
from torch.utils.data import Dataset

TYPES = ["Bedroom", "Bathroom", "LivingRoom", "Kitchen"]
TYPE_RANK = {t: i for i, t in enumerate(TYPES)}

PAD, BOS, EOS = 0, 1, 2
TYPE_BASE = 3
COORD_BASE = TYPE_BASE + len(TYPES)  # 7
NBINS = 32
VOCAB = COORD_BASE + NBINS  # 39

BIN_LO, BIN_HI = -0.1, 1.1
MAX_ROOMS = 10
TOKS_PER_ROOM = 5
SEQ_LEN = 2 + TOKS_PER_ROOM * MAX_ROOMS  # 52

COORD_EPS = 1e-9


# ---------------------------------------------------------------- loading

def load_plans(path: str) -> list[dict]:
    houses = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                houses.append(json.loads(line))
    return houses


def house_fingerprint(h: dict) -> str:
    """Identity of a house: sorted (type, rounded rect) multiset.

    Used to keep duplicated houses inside a single split (no leakage).
    """
    sig = sorted(
        (r["type"], *[round(float(v), 2) for v in r["rect"]]) for r in h["rooms"]
    )
    return hashlib.md5(json.dumps(sig).encode()).hexdigest()


def make_split(houses: list[dict], train_target: int = 9000,
               seed: int = 42) -> tuple[list[int], list[int], dict]:
    """Group-aware split: every fingerprint lands entirely in one split.

    Groups are shuffled, then whole groups go to test until it reaches
    train_target's complement (10% of the houses). Exact sizes may differ
    by the size of the duplicated groups; the actual numbers are reported.
    """
    groups: dict[str, list[int]] = {}
    for i, h in enumerate(houses):
        groups.setdefault(house_fingerprint(h), []).append(i)
    keys = sorted(groups)
    rnd = random.Random(seed)
    rnd.shuffle(keys)
    n = len(houses)
    test_target = max(1, n - train_target)
    train: list[int] = []
    test: list[int] = []
    for k in keys:
        if len(test) < test_target:
            test.extend(groups[k])
        else:
            train.extend(groups[k])
    train.sort()
    test.sort()
    fp_tr = {house_fingerprint(houses[i]) for i in train}
    fp_te = {house_fingerprint(houses[i]) for i in test}
    info = {
        "n_houses": n,
        "n_train": len(train),
        "n_test": len(test),
        "n_unique_fingerprints": len(groups),
        "n_duplicated_houses": n - len(groups),
        "fingerprint_overlap": len(fp_tr & fp_te),
    }
    assert info["fingerprint_overlap"] == 0, "house fingerprint leaked across splits"
    return train, test, info


# ----------------------------------------------------------- normalization

def normalize_house(h: dict) -> list[tuple[str, float, float, float, float]]:
    """Rooms as (type, x0, z0, x1, z1) in [0,1]^2, ordered by the rule above.

    Per-axis normalization by the house bbox of room rects.
    """
    rooms = h["rooms"]
    assert len(rooms) <= MAX_ROOMS, f"house has {len(rooms)} rooms > MAX_ROOMS"
    xs = [float(v) for r in rooms for v in (r["rect"][0], r["rect"][2])]
    zs = [float(v) for r in rooms for v in (r["rect"][1], r["rect"][3])]
    x_min, x_max = min(xs), max(xs)
    z_min, z_max = min(zs), max(zs)
    x_span = max(x_max - x_min, COORD_EPS)
    z_span = max(z_max - z_min, COORD_EPS)
    out = []
    for i, r in enumerate(rooms):
        x0, z0, x1, z1 = (float(v) for v in r["rect"])
        out.append((
            r["type"],
            (x0 - x_min) / x_span, (z0 - z_min) / z_span,
            (x1 - x_min) / x_span, (z1 - z_min) / z_span,
            i,  # program index for ordering
        ))
    out.sort(key=lambda t: (TYPE_RANK[t[0]], t[5]))
    return [(t[0], t[1], t[2], t[3], t[4]) for t in out]


def plan_order(type_list: list[str]) -> list[int]:
    """Slot order for a program: permutation P s.t. slot k holds type_list[P[k]].

    Same rule as normalize_house: stable sort by (type rank, program index).
    """
    idx = list(range(len(type_list)))
    idx.sort(key=lambda i: (TYPE_RANK[type_list[i]], i))
    return idx


# ------------------------------------------------------------- quantizing

def quantize(v: float) -> int:
    b = int(round((v - BIN_LO) / (BIN_HI - BIN_LO) * (NBINS - 1)))
    return max(0, min(NBINS - 1, b))


def dequantize(b: int) -> float:
    return BIN_LO + b * (BIN_HI - BIN_LO) / (NBINS - 1)


def encode(rooms: list[tuple[str, float, float, float, float]]) -> list[int]:
    """rooms in normalized+ordered form -> token ids (with BOS/EOS, no PAD)."""
    toks = [BOS]
    for typ, x0, z0, x1, z1 in rooms:
        toks += [TYPE_BASE + TYPE_RANK[typ],
                 COORD_BASE + quantize(x0), COORD_BASE + quantize(z0),
                 COORD_BASE + quantize(x1), COORD_BASE + quantize(z1)]
    toks.append(EOS)
    return toks


def decode(tokens) -> list[tuple[str, float, float, float, float]]:
    """token ids -> [(type, x0, z0, x1, z1)] dequantized.

    Parses by sequence POSITION, never by token value: a model may emit an
    out-of-range id in a coordinate slot and it must not shift alignment.
    Such ids dequantize outside the margin grid and surface in the metrics
    (degenerate / out-of-bounds) instead of being silently dropped.
    A BOS is skipped if present; parsing stops at an EOS on a room boundary.
    """
    toks = [int(t) for t in tokens]
    if toks and toks[0] == BOS:
        toks = toks[1:]
    rooms = []
    i = 0
    while i + TOKS_PER_ROOM <= len(toks):
        if toks[i] == EOS:
            break
        t = toks[i]
        typ = TYPES[t - TYPE_BASE] if TYPE_BASE <= t < TYPE_BASE + len(TYPES) \
            else f"invalid{t}"
        rooms.append((typ,
                      dequantize(toks[i + 1] - COORD_BASE),
                      dequantize(toks[i + 2] - COORD_BASE),
                      dequantize(toks[i + 3] - COORD_BASE),
                      dequantize(toks[i + 4] - COORD_BASE)))
        i += TOKS_PER_ROOM
    return rooms


# ---------------------------------------------------------------- dataset

@dataclass
class PlanDataset(Dataset):
    """Fixed-length padded token sequences for next-token training."""

    seqs: torch.Tensor  # [N, SEQ_LEN] long
    house_ids: list[int] = field(default_factory=list)

    @classmethod
    def from_houses(cls, houses: list[dict], indices: list[int]) -> "PlanDataset":
        seqs = torch.full((len(indices), SEQ_LEN), PAD, dtype=torch.long)
        ids = []
        for row, hi in enumerate(indices):
            toks = encode(normalize_house(houses[hi]))
            seqs[row, :len(toks)] = torch.tensor(toks, dtype=torch.long)
            ids.append(hi)
        return cls(seqs=seqs, house_ids=ids)

    def __len__(self) -> int:
        return self.seqs.shape[0]

    def __getitem__(self, i: int) -> torch.Tensor:
        return self.seqs[i]


def batches(seqs: torch.Tensor, batch_size: int, seed: int = 0):
    """Yield (x, y) shifted-by-one batches over shuffled rows."""
    g = torch.Generator().manual_seed(seed)
    order = torch.randperm(seqs.shape[0], generator=g)
    for i in range(0, seqs.shape[0], batch_size):
        rows = order[i:i + batch_size]
        s = seqs[rows]
        yield s[:, :-1], s[:, 1:]
