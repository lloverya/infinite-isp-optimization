# Original project & modification declaration

This repository is a **derivative work** of the open-source
[Infinite-ISP reference model](https://github.com/10xEngineersTech/Infinite-ISP_ReferenceModel)
(Python), released under the **Apache License 2.0** by 10xEngineers
(Copyright 2024). The upstream license ([LICENSE](LICENSE)) and attribution
([NOTICE](NOTICE)) are retained unchanged, as required by Apache-2.0 §4.

## Baseline

- Upstream commit/version: **Python reference-model implementation** of the
  Infinite-ISP pipeline (`infinite_isp.py` / `isp_pipeline.py`),
  Apache-2.0, Copyright 2024, 10xEngineers.
- Baseline pipeline order (unchanged): `RAW → Crop → DPC → BLC → OECF → DG →
  LSC → BNR → AWB → WB → Demosaic → CCM → Gamma → AE → CSC → LDCI → Sharpen →
  NR2D → RGBC → Scale → YUV`.
- The pipeline is driven by a single `configs.yml`; test images are the
  repository's built-in 2592×1536 12-bit RGGB RAWs.

## What was changed

Every file in this repo is byte-for-byte the upstream file **except** the
following. Nothing else was modified.

| File | Status | Change |
|---|---|---|
| `modules/noise_reduction_2d/non_local_means.py` | **modified** | `apply_mean_filter` rewritten with `cv2.boxFilter` — see `patches/noise_reduction_2d/non_local_means.py.patch`. **Bit-identical** output. Also added a defensive `uint16→uint8` normalization branch (not hit by the standard pipeline, documented in-code). |
| `modules/bayer_noise_reduction/joint_bf.py` | **modified** | JBF `np.exp()` range weight computed once and reused — see `patches/bayer_noise_reduction/joint_bf.py.patch`. **Bit-identical** output. |
| `config/configs.yml` | **values supplied in this repo** (`config/ccm_end2end_g100_n3_raw.yml`) | The CCM direction is **configuration-only**: `color_correction_matrix` is set from `config/ccm_end2end_g100_n3_raw.yml`, and `color_saturation_enhancement.saturation_gain` is set to `1.0`. No module code is touched. |

Everything else — `config/`, `scripts/`, `results/`, the docs — is **new work
added on top of the upstream project** and is not part of the upstream source.

## New dependency

The upstream project ships **no OpenCV**. The denoise acceleration introduces
`cv2.boxFilter`, so `opencv-python` is a new runtime dependency (see
[requirements.txt](requirements.txt)). The acceleration is bit-identical but the
dependency is real — install `opencv-python` to run the patched modules.

## Verification

- **Denoise patch, bit-identical end-to-end:** the modified pipeline output
  (`1536×2592×3` uint8) equals the original pipeline output byte-for-byte
  (`verify_patch_end2end.py` → `bit-identical: True`, max diff = 0).
- **CCM, quantified:** rendered-domain ColorChecker ΔE 63 → 17 (all 3 charts);
  real-scene clipping ~33% → ≤2.2% (all 5 scenes).

## License of this work

The modifications and new files are contributed under the same **Apache License
2.0** as the upstream project. If you redistribute, retain this notice, keep the
upstream `LICENSE` and `NOTICE`, and state which files were modified (this file
does so).
