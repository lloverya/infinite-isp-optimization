# Patches

Two module files of the upstream Infinite-ISP Python pipeline, modified for the
denoise acceleration. Both edits are **bit-identical** (verified end-to-end,
max diff = 0).

| File | Original | Modified | Patch |
|---|---|---|---|
| `modules/noise_reduction_2d/non_local_means.py` | `.bak_orig` | `non_local_means.py` | `non_local_means.py.patch` |
| `modules/bayer_noise_reduction/joint_bf.py` | `.bak_orig` | `joint_bf.py` | `joint_bf.py.patch` |

## Apply

```bash
cd <your Infinite-ISP checkout>
# option 1: patch
git apply <repo>/patches/noise_reduction_2d/non_local_means.py.patch
git apply <repo>/patches/bayer_noise_reduction/joint_bf.py.patch

# option 2: overwrite the modules directly
cp <repo>/patches/noise_reduction_2d/non_local_means.py  modules/noise_reduction_2d/
cp <repo>/patches/bayer_noise_reduction/joint_bf.py      modules/bayer_noise_reduction/

pip install opencv-python   # new dependency (the upstream project ships no cv2)
```

## Revert

```bash
cp modules/noise_reduction_2d/non_local_means.py.bak_orig modules/noise_reduction_2d/non_local_means.py
cp modules/bayer_noise_reduction/joint_bf.py.bak_orig     modules/bayer_noise_reduction/joint_bf.py
```
