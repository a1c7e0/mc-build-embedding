# Overnight Work Summary (2026-10-11 21:00 → 2026-10-12)

## Major Accomplishments

### 1. WSL Auto-Shutdown Problem — ROOT CAUSE FOUND & FIXED
- **Root cause**: WSL2 shuts down when no `wsl.exe` client is attached (not just "no
  Linux processes"). All remote `nohup tail`/`screen` attempts failed because they
  were nested SSH sessions that die on disconnect.
- **Fix**: Native Windows scheduled task `WSLHold` running
  `wsl.exe -d Debian -u root -- tail -f /dev/null` (single non-repeating launch).
- **Critical lesson**: A repeating scheduled task with a `:loop`/`goto loop` batch
  (the old `WSLKeepAlive`) CAUSES the churn — Task Scheduler kills the previous
  wsl.exe on each trigger, restarting WSL instead of protecting it. Disabled it.
- **Updated `wsl-remote-ops` skill** with correct guidance + diagnostic steps.
- System sleep was ruled out via kernel-power event log (last sleep event was Sept 21).

### 2. Plan Generator (program→plan) — COMPLETE & FIXED
- **Training done**: 10,000 steps on mcbox (604 seconds, 0.06s/step).
  Test next-token accuracy: 73.1% (type 89.4%, coord 67.3%).
- **Aggregate quality (20 samples, temp=1.0)**:
  - degenerate_rate: 0.0, oob_rate: 0.0 (perfect)
  - overlap_rate: 0.20 (vs GT 0.049)
  - clean_house_rate: 0.40 (vs GT 0.8125)
- **DSL export fixed** (3 bugs found and fixed):
  1. Missing `skeleton_compile.py` on mcbox (tar didn't include it)
  2. `render.py` outdated on mcbox (missing `cutaway` param)
  3. Coordinate quantization bug: `floor`/`ceil` mismatch broke shared-edge
     detection → interior walls never generated. Fixed with consistent `round()`.
  4. Added `_derive_walls()`: interior partition walls from shared room-rect
     edges + door openings + exterior window openings.
- **Visual QA confirmed**: interior walls now present, doors/windows visible.

### 3. 64³ VQ Resolution Verdict — RESOLUTION HYPOTHESIS WINS
| Metric | 32³ | 64³ | Change |
|---|---|---|---|
| occupied IoU | 0.836 | **0.895** | +5.9pp |
| non-air acc | 0.803 | **0.865** | +6.2pp |
| wood IoU | 0.233 | **0.370** | +13.7pp |
| glass IoU | 0.101 | **0.213** | **doubled** |
| organic IoU | 0.097 | **0.169** | +7.2pp |
| colourful IoU | 0.083 | **0.188** | **doubled** |
- Rare materials roughly doubled — confirms "细节低于分辨率" was the bottleneck.
- 64³ codebook atlas running (will confirm part-level semantics at higher res).

### 4. Style Data — 800 Variants Ready for Training
- 4 styles × 200 variants each (chinese_temple, japanese_house, gothic_church,
  nautical_dock) — all compiled successfully.
- Converted to vq_data format: `vq_data_stylegen.npz` (800, 32, 32, 32) with
  exact style labels (medieval 200 / asian 400 / nautical 200).

### 5. Disk Cleanup
- cr60: freed ~800MB (MiMo batch inputs 530MB + vision_tmp 263MB + old tars)
- seewo A: freed 89MB (old prior checkpoints v3, v2)
- seewo B: minimal (space was already fine)
- mcbox: minimal (one docker image)

### 6. Other
- seewo A/B: EasiNote killed, both raised to 10 cores.
- vq64 on seewo B: completed training (but step counter inflated through
  restart-loop; weights saved correctly). Container stopped.
- Prior v4 on seewo A: training in progress (step 7180/20000, loss 0.054,
  ~18h remaining).

## Pending / Next Steps
1. **64³ codebook atlas** (running in background) — visual confirmation of
   part-level semantics at higher resolution.
2. **Prior v4 completion** (~18h) — then sampling + visual QA comparison.
3. **Style-conditioned training** with the 800 stylegen variants.
4. **Attribute tags → condition tokens** (from ProcTHOR metadata).
5. **Residual plan quality gap**: overlap_rate 20% vs GT 4.9% — needs improvement
   (maybe higher training steps, or overlap penalty in loss).
