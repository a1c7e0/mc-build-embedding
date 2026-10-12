# plan_gen: program -> plan generator

Small GPT that turns a building program (list of room types) into a
rectangle layout for those rooms. Trained on `out_lib/plans/procthor_plans.jsonl`
(10k ProcTHOR plans). Part of the skeleton-line neural stack: generated
plans can be exported through the Skeleton DSL v0 compiler
(`src/skeleton_compile.py`) into a voxel shell.

## Serialization

`[BOS] (type x0 z0 x1 z1) * n_rooms [EOS] [PAD ...]`

- room order: stable sort by (type rank, program index) -- computable from
  the program alone, which is what makes type-conditioned generation
  possible (area-descending order would require coordinates that do not
  exist yet at conditioning time)
- coordinates: per-house per-axis bbox normalization to [0,1], quantized to
  32 bins on the margin grid [-0.1, 1.1]; out-of-canvas coordinates are
  representable tokens, so the out-of-bounds metric is live
- vocabulary (39): PAD/BOS/EOS, 4 type tokens, 32 shared coord bins
  (channel recoverable from sequence position)

## Conditioning

Slot order is fixed by the ordering rule; the type token of every slot is
forced to the requested type; only the 4 coordinate tokens per room are
sampled. `--mode greedy|sample`, `--temp`, `--top-k` supported.

## Honest metrics (metrics.py)

degenerate rate (x1<=x0 or z1<=z0), intra-house room-pair overlap rate
(intersection area > 0), out-of-bounds rate (room not inside the [0,1]^2
canvas), clean-house rate. `metrics.self_test()` injects known-bad rooms
and asserts each metric fires on exactly those. Next-token accuracy is
reported for reference only.

## Split

House-fingerprint grouped (sorted type+rect multiset, md5): duplicated
houses stay within one split; `make_split` asserts zero fingerprint overlap.

## Smoke

```
docker run --rm --cpus=1 --memory=1200m -v $PWD:/work -w /work mcb-embed:torch \
  python src/plan_gen/smoke.py --steps 300 --batch 32 --samples 48
```

CLIs: `train.py` (training), `sample.py` (conditional generation + optional
`--dsl` export), `smoke.py` (whole chain, writes `out/plan_gen_smoke/`).
