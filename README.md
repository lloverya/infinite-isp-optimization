# Infinite-ISP Optimizations

Performance and color-accuracy optimizations on top of the open-source
[Infinite-ISP](https://github.com/10xEngineersTech/Infinite-ISP_ReferenceModel)
reference ISP pipeline (Python), Apache-2.0. Two independent directions:

| # | Direction | What changed | Result |
|---|---|---|---|
| **A** | Denoise **speed** | 2 module files rewritten (redundant computation removed) | **2.2×** end-to-end, **bit-identical** output; up to **4.7×** visually lossless |
| **B** | **Camera-level** color accuracy (CCM) | Configuration only — no module code touched | ColorChecker ΔE **63 → 17**; real-scene clipping **~33% → <2.2%** |

The two directions are independent: **A is lossless** (bit-identical, verifiable),
**B deliberately changes color** (this is why A/B PSNR is finite rather than ∞).

> This repository is a **derivative work**. See [ORIGINAL.md](ORIGINAL.md) for the
> exact baseline, the list of modified files, and the attribution/licensing notes.
> Upstream copyright: `Infinite-ISP, Copyright 2024, 10xEngineers` (see [NOTICE](NOTICE)).

> **Disclaimer / 声明** — This project is the author's learning work on ISP
> (image signal processing). If there are any errors, or any infringement of
> existing work, it is unintentional — please point them out and it will be
> corrected. 本项目为作者为学习ISP（图像信号处理）的产物，如有错误或侵权，敬请指正！

## Acknowledgments

The optimizations in this repository were developed with the assistance of
**Claude (Anthropic)** — the profiling, the bit-identical denoise rewrite, the
joint CCM calibration, and the end-to-end verification were carried out
interactively with [Claude Code](https://claude.com/claude-code). The author
directed the project and made all design decisions.

---

## A. Denoise acceleration — 2.2× with bit-identical output

### Problem

Profiling a 2592×1536 RGGB RAW shows the two denoise modules dominate:

| Module | Time | Share | Algorithm |
|---|---|---|---|
| **NR2D** (2D denoise) | 11.8 s | 60.6% | NLM |
| **BNR** (Bayer denoise) | 5.8 s | 29.7% | Joint bilateral filter |
| Demosaic + rest | ~1.9 s | ~10% | AHD + … |

They contain pure redundancy:

* `non_local_means.py`: the mean filter over each patch was 25 nested
  full-image `int32` accumulations per search-window offset (≈ 81×25 = 2025
  full-image scans per frame).
* `joint_bf.py`: `np.exp(...)` of the range term was computed **twice per offset**
  with the identical value.

### Fix (modified files in `patches/`)

| File | Change | Module speed-up |
|---|---|---|
| `modules/noise_reduction_2d/non_local_means.py` | 25-loop mean filter → one `cv2.boxFilter(normalize=False, borderType=BORDER_REFLECT_101)` + `S//25` | 3.05× |
| `modules/bayer_noise_reduction/joint_bf.py` | compute `np.exp()` once, reuse `s_kern*w` in both the normalizer and the weighted sum | 1.6× |

**Why bit-identical:** both edits remove redundant computation without changing
any arithmetic semantics — `boxFilter(normalize=False)` returns the exact integer
window sum (window sum ≤ 1.6e6 ≪ 2^53, exactly representable in float64), and
`S//25 == trunc(S/25.0)` for non-negative integers. Verified end-to-end:
`scripts/verify_patch_end2end.py` reports **`bit-identical: True`** (max diff = 0
over the full 1536×2592×3 output).

> Trap recorded: cv2 `BORDER_REFLECT_101` == `np.pad 'reflect'`, while
> `BORDER_REFLECT` is `'symmetric'`. The wrong border gives max diff 8141.

### Results (whole pipeline, same 3 RAW images)

| Config | Pipeline time | Speed-up | Quality vs original |
|---|---|---|---|
| Original | ≈ 19.5 s | 1.0× | baseline |
| **Patch (w9/f9)** | ≈ 8.9 s | **2.2×** | **bit-identical (PSNR = ∞)** |
| **Aggressive (w5/f5)** | ≈ 4.1 s | **4.7×** | PSNR 49.5 dB, max diff 7/255 |

PSNR ≥ 49 dB ⇔ pixel RMS < 0.9/255, below the visible threshold.

---

## B. Camera-level CCM color calibration — ΔE 63 → 17

### Problem

The out-of-box CCM is not calibrated for this sensor: rendering a ColorChecker
RAW gives **ΔE ≈ 63** and visibly wrong saturated colors (red `[244,6,67]` renders
magenta). The root cause is structural:

> The pipeline applies CSE (color saturation enhancement) **after gamma**, scaling
> YUV chroma by `saturation_gain: 1.5`. The CCM chroma gain + a second ×1.5
> saturation multiply stack and push high-saturation patches out of the sRGB
> gamut — so a new CCM alone cannot fix it; **CCM and saturation_gain must be
> calibrated together**.

Real scenes confirm it is a global effect, not a test-card artifact: 6–35% of
pixels are clipped to a pure color on all 5 built-in scenes.

### Method

1. **Joint linear-domain calibration** (`scripts/joint_ccm_calibrate.py`) — fit
   one camera-level 3×3 CCM on 72 non-saturated patches of 3 ColorChecker RAWs
   (different ISO/lighting), with leave-one-out cross-validation (2 fit /
   1 predict): each fold drops ΔE ~60 → 18–20, so it generalizes, not overfits.
2. **Key insight — the linear-domain fit misses the pipeline's nonlinearity.**
   The real chain is
   `CCM → gamma LUT (4096, non-sRGB) → CSC (BT.601, CSE ×1.5) → RGBC → 8bit`.
3. **End-to-end optimization** (`scripts/optimize_ccm_end2end.py`) — build a
   forward surrogate of that whole chain (validation error ≈ 3/255), then fit
   the 3×3 CCM against the **actual rendered output vs X-Rite sRGB** with
   `least_squares`. A saturation-gain sweep shows 1.5 is the accuracy
   bottleneck; **sat 1.0 + neutral-patch weighting is optimal**.
4. **Final**: `M_e2e` + `saturation_gain: 1.0` (matrix in
   [`config/ccm_end2end_g100_n3_raw.yml`](config/ccm_end2end_g100_n3_raw.yml)).

### Results

ColorChecker rendered-domain ΔE (24 patches, same anchors):

| Chart image | Original | M_joint (linear) | **M_e2e (sat 1.0)** |
|---|---|---|---|
| ColorChecker | 63.3 | 37.1 | **17.3** |
| ISO2500 | 57.8 | 39.7 | **16.4** |
| 100DPs | 62.5 | 37.3 | **16.8** |

Saturated patches all fixed: red `[244,6,67]` magenta → `[190,46,77]`; orange,
blue, magenta, mid-red hues all corrected; gray patches ΔE ≈ 3.

Real-scene generalization (5 built-in scenes, % pixels clipped to pure color):

| Scene | Original | M_joint | **M_e2e (sat 1.0)** |
|---|---|---|---|
| Indoor1 | 32.9% | — | **0.5%** |
| Outdoor1 | 34.9% | 11.0% | **2.0%** |
| Outdoor2 | 17.9% | — | **1.5%** |
| Outdoor3 | 6.0% | — | **0.9%** |
| Outdoor4 | 15.2% | — | **2.2%** |

Why M_e2e wins: M_joint (pure linear fit) looks great indoors but blows out on
outdoor pure-color scenes (11% clipping, purple sky) — the same gamut-overflow
mechanism as the chart. M_e2e models the full chain, so it is stable on all 5
scenes (clipping ≤ 2.2%, black-truncated pixels 0.03% vs 2.15%).

---

## Repository layout

```
├── README.md                this file
├── ORIGINAL.md              baseline & modification declaration
├── LICENSE                  Apache-2.0 (upstream, retained)
├── NOTICE                   upstream attribution (retained)
├── requirements.txt         upstream deps + opencv-python (new)
├── patches/                 ← the only modified upstream files
│   ├── noise_reduction_2d/  non_local_means.py (+ .bak_orig, .patch)
│   └── bayer_noise_reduction/ joint_bf.py (+ .bak_orig, .patch)
├── config/
│   └── ccm_end2end_g100_n3_raw.yml   ← final CCM matrix (M_e2e, sat 1.0)
├── scripts/
│   ├── make_ab_render3a_pca.py       A/B comparison generator (results/ab_compare)
│   ├── make_cse_cross_test.py        CCM × CSE ablation generator (results/ccm_x_cse_ablation)
│   ├── joint_ccm_calibrate.py        joint linear-domain CCM fit + LOO cross-validation
│   ├── optimize_ccm_end2end.py       end-to-end CCM optimization (forward surrogate)
│   └── verify_patch_end2end.py       bit-identical proof of the denoise patch
└── results/                 rendered comparisons (full resolution)
    ├── ab_compare/          original-vs-optimized A/B, all 6 built-in scenes
    └── ccm_x_cse_ablation/  CCM × CSE-saturation 2×2 ablation matrices
```

## Reproduce

```bash
# 1) apply the denoise acceleration (bit-identical):
#    patches/noise_reduction_2d/non_local_means.py       -> modules/noise_reduction_2d/
#    patches/bayer_noise_reduction/joint_bf.py           -> modules/bayer_noise_reduction/
pip install -r requirements.txt          # includes opencv-python (new dependency)

# 2) apply the CCM calibration (configuration-only):
#    write color_correction_matrix from config/ccm_end2end_g100_n3_raw.yml
#    into configs.yml, and set color_saturation_enhancement.saturation_gain: 1.0

# 3) reproduce the results/ comparisons (adjust the two project paths in each script):
python scripts/make_ab_render3a_pca.py   # runs both pipelines, renders A/B + PSNR
python scripts/make_cse_cross_test.py    # CCM × CSE 2×2 ablation
```

## A/B comparison (original vs optimized)

Produced by running each pipeline independently on the same RAW (left: original
ISP, right: optimized ISP; top-center PSNR of the optimized output vs the
original). The composites are rendered at 2568 px wide with labels sized so they
stay readable when GitHub scales them to the container width.

**ColorChecker** — the CCM recalibration fixes the oversaturated/magenta reds:

![ColorChecker A/B](https://cdn.jsdelivr.net/gh/lloverya/infinite-isp-optimization@main/results/ab_compare/ColorChecker.jpg)

**Indoor1**:

![Indoor1 A/B](https://cdn.jsdelivr.net/gh/lloverya/infinite-isp-optimization@main/results/ab_compare/Indoor1.jpg)

**CCM × CSE-saturation ablation** — the 2×2 matrix showing that the calibrated
CCM alone beats the original CCM at *any* saturation setting:

![CCM x CSE ablation](https://cdn.jsdelivr.net/gh/lloverya/infinite-isp-optimization@main/results/ccm_x_cse_ablation/ColorChecker.jpg)
