# -*- coding: utf-8 -*-
"""
相机通用 CCM 联合标定 (方向 1)
================================
用 3 张内置 ColorChecker RAW 联合拟合一个「相机级」3x3 CCM (对整机、跨光照/ISO 有效):

  1. 逐图跑模块链到 demosaic+WB, 取线性域观察值 (修 calibrate_ccm_raw 的 filename bug)
  2. 在原版 sRGB 输出上用模板扫描定位 4x6 网格 (find_grid), 失败退 detect_grid
  3. 线性域采样 24 块, 排除饱和块 (obs.max() > 0.98)
  4. 3 图有效块堆叠 -> lstsq 拟合 ref_linear = M_joint @ obs_linear, 存 ccm_joint_raw.yml
  5. LOO 交叉验证 (2 拟合 / 1 验证): 预测留出图 -> sRGB ΔE (均值/中位/最大)
  6. 渲染 {label}_joint.png, 与 orig / 现有单图 CCM 同锚点采样 -> 渲染域 ΔE 汇总表
  7. 生成 orig vs joint 并排对比图

用法: python joint_ccm_calibrate.py
"""
import os
import sys
import time
import glob
import numpy as np
import cv2
import yaml

ISP_DIR = "E:/Infinite-ISP-main/Infinite-ISP-main"
sys.path.append(ISP_DIR)
ROOT = "F:/Claude code/ISP_learning/infinite-ISP-development/experiments/isp_dncnn_comparison"
sys.path.insert(0, os.path.join(ROOT, "experiments"))

from infinite_isp import InfiniteISP
from detect_colorchecker import X_RITE_SRGB, sample_patch, imread_u

OUT_DIR = os.path.join(ROOT, "visual_compare", "outputs")
CONFIGS_DIR = os.path.join(ROOT, "visual_compare", "configs")
COMPARE_DIR = os.path.join(ROOT, "visual_compare", "comparisons")
for d in (OUT_DIR, CONFIGS_DIR, COMPARE_DIR):
    os.makedirs(d, exist_ok=True)
RAW_DIR = os.path.join(ISP_DIR, "in_frames", "normal")
CONFIG_PATH = os.path.join(ISP_DIR, "config", "configs.yml")
REF = np.array(X_RITE_SRGB, dtype=float)
DXDY = (218, 207)          # 模板扫描网格间隔 (validate_ccm_raw 标定)
SAT_MAX = 0.98             # 线性域饱和阈值 (DNG 路径同款)

RAW_LABELS = [
    ("ColorChecker",           "ColorChecker_2592x1536_12bits_RGGB.raw"),
    ("ColorCheckerRAW_ISO2500", "ColorCheckerRAW_ISO2500_2592x1536_12bit_RGGB.raw"),
    ("ColorCheckerRaw_100DPs",  "ColorCheckerRaw_100DPs_ISO100_2592x1536_12bits_RGGB.raw"),
]

# ---------- 色彩数学 (calibrate_ccm_raw 同款) ----------
def srgb2linear(v):
    v = np.asarray(v, dtype=np.float64) / 255.0
    return np.where(v <= 0.04045, v / 12.92, ((v + 0.055) / 1.055) ** 2.4)

def lin2srgb(v):
    v = np.clip(v, 0, 1)
    out = np.where(v <= 0.0031308, v * 12.92, 1.055 * (v ** (1 / 2.4)) - 0.055)
    return out * 255

def fit_ccm(obs_linear, ref_linear):
    """lstsq 拟合 ref = M @ obs (每列独立解)"""
    M, *_ = np.linalg.lstsq(obs_linear, ref_linear, rcond=None)
    return M.T

# ---------- 网格定位 (validate_ccm_raw 同款) ----------
def grid_de(img, ax, ay, dx, dy, half=28):
    de = np.zeros(24)
    for r in range(4):
        for c in range(6):
            cx, cy = ax + c * dx, ay + r * dy
            y1, y2 = max(0, int(cy) - half), min(img.shape[0], int(cy) + half)
            x1, x2 = max(0, int(cx) - half), min(img.shape[1], int(cx) + half)
            if y2 <= y1 or x2 <= x1:
                return 1e9, None
            obs = img[y1:y2, x1:x2].mean(axis=(0, 1))
            de[r * 6 + c] = np.linalg.norm(obs - REF[r * 6 + c])
    return de.mean(), de

def find_grid(img, dx=DXDY[0], dy=DXDY[1]):
    best = (1e9, None, None)
    for ay in range(150, 1200, 30):
        for ax in range(200, 1900, 30):
            score, de = grid_de(img, ax, ay, dx, dy)
            if score < best[0]:
                best = (score, (ax, ay), de)
    return best

# ---------- 线性观察值 ----------
def get_linear_img(config_path):
    """模块链 Crop->...->demosaic+WB, 返回 (linear_float01, bit_depth)。
    config_path 必须已含正确的 platform.filename —— InfiniteISP 的 raw_file 在
    __init__/load_config 时捕获, 事后改 platform['filename'] 不会生效
    (这是 calibrate_ccm_raw 的坑: 它没设 filename, 三张图会全加载 config 默认图)。
    """
    isp = InfiniteISP(data_path=RAW_DIR, config_path=config_path)
    isp.load_raw()
    bit_depth = isp.sensor_info["bit_depth"]
    from modules.crop.crop import Crop
    from modules.dead_pixel_correction.dead_pixel_correction import DeadPixelCorrection as DPC
    from modules.black_level_correction.black_level_correction import BlackLevelCorrection as BLC
    from modules.oecf.oecf import OECF
    from modules.digital_gain.digital_gain import DigitalGain as DG
    from modules.lens_shading_correction.lens_shading_correction import LensShadingCorrection as LSC
    from modules.bayer_noise_reduction.bayer_noise_reduction import BayerNoiseReduction as BNR
    from modules.auto_white_balance.auto_white_balance import AutoWhiteBalance as AWB
    from modules.white_balance.white_balance import WhiteBalance as WB
    from modules.demosaic.demosaic import Demosaic

    crop = Crop(isp.raw, isp.platform, isp.sensor_info, isp.parm_cro)
    cropped = crop.execute()
    dpc = DPC(cropped, isp.sensor_info, isp.parm_dpc, isp.platform)
    dpc_raw = dpc.execute()
    blc = BLC(dpc_raw, isp.platform, isp.sensor_info, isp.parm_blc)
    blc_raw = blc.execute()
    oecf = OECF(blc_raw, isp.platform, isp.sensor_info, isp.parm_oec)
    oecf_raw = oecf.execute()
    dga = DG(oecf_raw, isp.platform, isp.sensor_info, isp.parm_dga)
    dga_raw, _ = dga.execute()
    lsc = LSC(dga_raw, isp.platform, isp.sensor_info, isp.parm_lsc)
    lsc_raw = lsc.execute()
    bnr = BNR(lsc_raw, isp.sensor_info, isp.parm_bnr, isp.platform)
    bnr_raw = bnr.execute()
    awb = AWB(bnr_raw, isp.sensor_info, isp.parm_awb)
    awb.execute()
    wbc = WB(bnr_raw, isp.platform, isp.sensor_info, isp.parm_wbc)
    wb_raw = wbc.execute()
    cfa = Demosaic(wb_raw, isp.platform, isp.sensor_info, isp.parm_dem)
    demos = cfa.execute()
    return demos.astype(np.float32) / (2 ** bit_depth - 1), bit_depth

# ---------- 渲染 (run_visual_comparison.run_standard 本地副本, 不拉 DnCNN/torch) ----------
def build_config(name, raw_filename, ccm_yml=None, saturation_gain=None):
    with open(CONFIG_PATH, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["platform"]["filename"] = raw_filename
    if ccm_yml:
        with open(ccm_yml, encoding="utf-8") as f:
            calib = yaml.safe_load(f)["color_correction_matrix"]
        cfg["color_correction_matrix"].update(calib)
    if saturation_gain is not None:
        # CSE 色度增强系数 (默认 1.5): 调它改变全局饱和
        cfg["color_saturation_enhancement"]["saturation_gain"] = saturation_gain
    for k, v in cfg.items():
        if isinstance(v, dict) and "is_save" in v:
            v["is_save"] = False
    p = os.path.join(CONFIGS_DIR, name + ".yml")
    with open(p, "w", encoding="utf-8") as f:
        yaml.dump(cfg, f, allow_unicode=True)
    return p

def run_standard(config_path, data_path):
    isp = InfiniteISP(data_path=data_path, config_path=config_path)
    isp.load_raw()
    cwd = os.getcwd()
    os.chdir(ISP_DIR)
    try:
        t0 = time.time()
        isp.run_pipeline(visualize_output=True)
        dt = time.time() - t0
        files = glob.glob(os.path.join("out_frames", f"{isp.out_file}_*.png"))
        if not files:
            return None, dt
        latest = max(files, key=os.path.getctime)
        data = np.fromfile(latest, dtype=np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB), dt
    finally:
        os.chdir(cwd)

# ---------- 汇总工具 ----------
def de_stats(de):
    return de.mean(), np.median(de), de.max()

def print_matrix(title, M):
    print(f"  {title}")
    for i, name in enumerate(("corrected_red", "corrected_green", "corrected_blue")):
        print(f"    {name:16s}: {[round(M[i, j], 4) for j in range(3)]}")
    print(f"    行和: {[round(M[i].sum(), 3) for i in range(3)]}")

# ================= main =================
def main():
    print("=" * 72)
    print("相机通用 CCM 联合标定: 3 张内置 ColorChecker RAW 联合拟合 1 个相机级 CCM")
    print("=" * 72)

    # 0. 确保 orig render 存在 (无 CCM 版本)
    for label, fname in RAW_LABELS:
        p = os.path.join(OUT_DIR, f"{label}_orig.png")
        if not os.path.exists(p):
            print(f"\n[渲染] {label}_orig (缺失, 现渲染)...")
            cfg = build_config(f"{label}_orig", fname)
            img, dt = run_standard(cfg, RAW_DIR)
            if img is None:
                print(f"  ERROR: {label}_orig 渲染失败"); return
            cv2.imwrite(p, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            print(f"  done {dt:.1f}s")

    # 1. 逐图: 线性观察值 + 网格定位 + 采样 + 饱和掩码
    print("\n[1] 逐图提取线性域观察值 (demosaic+WB 后)")
    ref_linear = np.array([srgb2linear(v) for v in X_RITE_SRGB])
    data = []
    for label, fname in RAW_LABELS:
        t0 = time.time()
        cfg = build_config(f"linear_{label}", fname)   # 配置文件里写对 filename
        linear, bit = get_linear_img(cfg)
        print(f"  {label}: 线性图 {linear.shape} bit={bit}  ({time.time()-t0:.1f}s)")

        orig = imread_u(os.path.join(OUT_DIR, f"{label}_orig.png"))
        if orig is None:
            print(f"  ERROR: {label}_orig.png 读取失败"); return
        score_o, anchor, de_o = find_grid(orig)
        if anchor is None:
            print(f"  ERROR: {label} 模板网格定位失败 (尝试 detect_grid 退回)"); return
        ax, ay = anchor
        obs = np.array([sample_patch(linear, ax + c * DXDY[0], ay + r * DXDY[1])
                        for r in range(4) for c in range(6)])
        valid = obs.max(axis=1) <= SAT_MAX
        print(f"     锚点=({ax},{ay})  orig渲染ΔE={score_o:5.1f}  饱和排除 {24 - valid.sum()}/24 块")
        data.append(dict(label=label, fname=fname, obs=obs, valid=valid,
                         anchor=anchor, de_orig_render=score_o))

    # 2. 联合拟合
    print("\n[2] 3 图堆叠拟合 M_joint")
    A, B = [], []
    for d in data:
        v = d["valid"]
        A.append(d["obs"][v]); B.append(ref_linear[v])
    A, B = np.vstack(A), np.vstack(B)
    print(f"    有效块数: {A.shape[0]} (3 图)")
    M_joint = fit_ccm(A, B)
    print_matrix("M_joint (ref = M @ obs, 线性域):", M_joint)

    joint_yml = os.path.join(CONFIGS_DIR, "ccm_joint_raw.yml")
    calib = {"color_correction_matrix": {
        "corrected_red": M_joint[0].tolist(),
        "corrected_green": M_joint[1].tolist(),
        "corrected_blue": M_joint[2].tolist(),
    }}
    with open(joint_yml, "w", encoding="utf-8") as f:
        yaml.dump(calib, f, allow_unicode=True)
    print(f"    已保存: {joint_yml}")

    # 3. LOO 交叉验证 (线性域, 每折 M_k 由另 2 张拟合)
    print("\n[3] LOO 交叉验证 (2 拟合 / 1 验证, 线性域观察值直接测)")
    print(f"  {'留出图':<24} {'无CCM ΔE':>9} {'LOO预测 ΔE':>10} {'中位':>6} {'最大':>6} {'改善':>7}")
    loo_rows = []
    for i, d in enumerate(data):
        A_tr, B_tr = [], []
        for j, dj in enumerate(data):
            if j == i:
                continue
            v = dj["valid"]
            A_tr.append(dj["obs"][v]); B_tr.append(ref_linear[v])
        M_k = fit_ccm(np.vstack(A_tr), np.vstack(B_tr))
        obs = d["obs"]
        v = d["valid"]
        no_ccm_srgb = lin2srgb(obs)
        pred_srgb = lin2srgb(np.clip(obs @ M_k.T, 0, 1))
        de0 = np.linalg.norm(no_ccm_srgb[v] - REF[v], axis=1)
        de_k = np.linalg.norm(pred_srgb[v] - REF[v], axis=1)
        m, md, mx = de_stats(de_k)
        impr = (1 - de_k.mean() / de0.mean()) * 100
        loo_rows.append((d["label"], de_k.mean()))
        print(f"  {d['label']:<24} {de0.mean():>7.1f} {m:>10.1f} {md:>6.1f} {mx:>6.1f} {impr:>6.1f}%")

    # 4. M_joint 每图线性域 ΔE
    print("\n[4] M_joint 在每图的线性域 ΔE (预测全部 24 块)")
    print(f"  {'图像':<24} {'ΔE(valid)':>10} {'ΔE(全部24)':>11}")
    for d in data:
        obs = d["obs"]
        v = d["valid"]
        pred = lin2srgb(np.clip(obs @ M_joint.T, 0, 1))
        de_v = np.linalg.norm(pred[v] - REF[v], axis=1)
        de_a = np.linalg.norm(pred - REF, axis=1)
        print(f"  {d['label']:<24} {de_v.mean():>10.1f} {de_a.mean():>11.1f}")

    # 5. 渲染 joint CCM
    print("\n[5] 渲染 {label}_joint.png (应用 M_joint)")
    for label, fname in RAW_LABELS:
        p = os.path.join(OUT_DIR, f"{label}_joint.png")
        cfg = build_config(f"{label}_joint", fname, joint_yml)
        img, dt = run_standard(cfg, RAW_DIR)
        if img is None:
            print(f"  ERROR: {label}_joint 渲染失败"); return
        cv2.imwrite(p, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        print(f"  {label:<24} {p}  ({dt:.1f}s)")

    # 6. 渲染域 ΔE 汇总 (同锚点, 模板采样, 全部 24 块 -> 与 README 口径一致)
    print("\n[6] 渲染域 ΔE 汇总 (同锚点, 24 块)")
    print(f"  {'图像':<24} {'orig':>7} {'现有单图CCM':>10} {'M_joint':>8} {'orig→joint 改善':>15}")
    joint_rows = []
    for d in data:
        label = d["label"]
        orig = imread_u(os.path.join(OUT_DIR, f"{label}_orig.png"))
        col = imread_u(os.path.join(OUT_DIR, f"{label}_color.png"))
        joint = imread_u(os.path.join(OUT_DIR, f"{label}_joint.png"))
        ax, ay = d["anchor"]
        de_o, _ = grid_de(orig, ax, ay, *DXDY)
        de_j, _ = grid_de(joint, ax, ay, *DXDY)
        de_c = grid_de(col, ax, ay, *DXDY)[0] if col is not None else None
        impr = (1 - de_j / de_o) * 100
        joint_rows.append((label, de_o, de_j, impr))
        de_c_s = f"{de_c:5.1f}" if de_c is not None else "   n/a"
        print(f"  {label:<24} {de_o:>5.1f} {de_c_s:>10} {de_j:>6.1f}   {impr:>6.1f}%")

    # 7. 并排对比图 orig vs joint
    print("\n[7] 生成 orig vs joint 对比图")
    font = cv2.FONT_HERSHEY_SIMPLEX
    for label, *_ in RAW_LABELS:
        orig = imread_u(os.path.join(OUT_DIR, f"{label}_orig.png"))
        joint = imread_u(os.path.join(OUT_DIR, f"{label}_joint.png"))
        if orig is None or joint is None:
            continue
        h = max(orig.shape[0], joint.shape[0])
        def padv(img):
            if img.shape[0] != h:
                img2 = np.zeros((h, img.shape[1], 3), dtype=np.uint8)
                img2[:img.shape[0]] = img
                return img2
            return img
        panel = np.hstack([padv(orig), padv(joint)])
        bar_h = 50
        p2 = np.zeros((panel.shape[0] + bar_h, panel.shape[1], 3), dtype=np.uint8)
        p2[bar_h:] = panel
        for i, t in enumerate(("Original", "M_joint (相机通用 CCM)")):
            (tw, th), _ = cv2.getTextSize(t, font, 1.0, 2)
            cx = i * orig.shape[1] + (orig.shape[1] - tw) // 2
            cv2.putText(p2, t, (cx, 32), font, 1.0, (255, 255, 255), 2)
        save = os.path.join(COMPARE_DIR, f"compare_{label}_joint.png")
        cv2.imwrite(save, p2)
        print(f"  {label:<24} {save}")

    # 8. 断言
    print("\n[8] 验证断言")
    ok = True
    for label, de_o, de_j, impr in joint_rows:
        good = de_j < de_o
        ok &= good
        print(f"  {label}: M_joint 渲染ΔE {de_j:.1f} < orig {de_o:.1f}  ->  {'PASS' if good else 'FAIL'}")
    for label, de_loo in loo_rows:
        d = next(x for x in data if x["label"] == label)
        v = d["valid"]
        no_ccm = np.linalg.norm(lin2srgb(d["obs"])[v] - REF[v], axis=1)
        good = de_loo < no_ccm.mean()
        ok &= good
        print(f"  LOO[{label}]: {de_loo:.1f} < 无CCM {no_ccm.mean():.1f}  ->  {'PASS' if good else 'FAIL'}")
    # obs 空间是「WB 后但未按白点归一」的线性 RGB，AE 曝光到中灰 => 中性灰线性值≈0.45-0.55。
    # 参考白点线性≈0.9，故行和≈0.9/0.5≈2（与现有单图 CCM 的 1.8-1.9 同属本空间）。
    # 物理 sanity: 行和全为正、且彼此一致（意味着对中性色映射稳定）。
    rows = np.array([M_joint[i].sum() for i in range(3)])
    good = bool(np.all(rows > 0) and (rows.max() / rows.min()) < 1.15)
    ok &= good
    print(f"  M_joint 行和 {np.round(rows, 3)} (对应观测中性电平≈{1.0/np.mean(rows):.2f}, "
          f"与现有单图 CCM 1.83/1.75/1.90 同属本空间)  ->  {'PASS' if good else 'FAIL'}")
    print(f"\n  全部断言: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1

if __name__ == "__main__":
    sys.exit(main())
