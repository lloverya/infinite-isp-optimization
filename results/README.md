# Results — rendered comparisons

Rendered comparisons (JPEG, full resolution ~5192 px wide), produced by running
each pipeline independently on the same RAW (2592×1536, 12-bit RGGB). Panels are
kept 1:1 with small labels (the original render style).

| Folder | Contents | What to look for |
|---|---|---|
| `ab_compare/` | `ColorChecker/Indoor1/Outdoor1-4` — original-vs-optimized side-by-side | Top-center PSNR; the optimized output has correct colors (no magenta reds, no clipped pure-color regions) |
| `ccm_x_cse_ablation/` | 2×2 ablation matrices: CCM (original/optimized) × CSE saturation (1.5/1.0) | On the well-lit ColorChecker scene the optimized CCM has lower ΔE at every saturation (57→25 at sat 1.0) and oversaturates less; on Indoor1 the card is dimly lit so ΔE is lighting-dominated |

## A/B configuration

`AWB = PCA`, `render_3a = true` (the default 3A loop), built-in test images.
Badges: left panel `Original ISP <time> s`, right panel `Optimized ISP <time> s`,
center PSNR = optimized vs original output.
