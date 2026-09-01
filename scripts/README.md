# Scripts

## Reproduce the `results/` comparisons

Both scripts inject the RAW `filename` / `RAW_DATA` into each project's
`configs.yml` and `isp_pipeline.py`, run `isp_pipeline.py`, parse the elapsed
time, then composite the comparison. **Adjust the two project root paths at the
top of each script** (`ROOT_E` = original project, `ROOT_D` = optimized project
with the patches + CCM config applied) before running.

| Script | Produces |
|---|---|
| `make_ab_render3a_pca.py` | `results/ab_compare/*` — A/B comparison for the 6 built-in scenes (ColorChecker, Indoor1, Outdoor1–4) |
| `make_cse_cross_test.py` | `results/ccm_x_cse_ablation/*` — 2×2 ablation: CCM (original/optimized) × CSE saturation (1.5/1.0), each panel tagged with rendering-domain ΔE (locates the ColorChecker grid on the top-left panel, samples all 24 patches on the same anchor) |

## Method / verification scripts

| Script | Purpose |
|---|---|
| `joint_ccm_calibrate.py` | Fit one camera-level 3×3 CCM on 3 ColorChecker RAWs (linear domain) + leave-one-out cross-validation |
| `optimize_ccm_end2end.py` | End-to-end CCM optimization: forward surrogate of `CCM→gamma→CSE→CSC→RGBC`, fitted against X-Rite sRGB; `--gain/--weight/--target_sat` sweepable |
| `verify_patch_end2end.py` | Run the patched vs original pipeline on the same RAW and assert byte-identical output (`bit-identical: True`, max diff = 0) |
