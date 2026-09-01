# -*- coding: utf-8 -*-
"""
端到端 CCM 优化: 以 pipeline 实际渲染输出为目标拟合 CCM
================================================================
线性域拟合(calibrate_ccm_raw / joint_ccm_calibrate)假设 pipeline 输出
== 标准 sRGB gamma(CCM(obs)), 但本 pipeline 实际是:

    CCM(12bit线性) -> gamma LUT -> CSC(BT.601, CSE 把 U/V 乘 1.5) -> RGBC(还原) -> 8bit RGB

CSE 的 1.5x 色度增强会让线性域拟合的结果在最终输出上过饱和
(例如红色块实际渲染 [244,6,67] vs 参考 [175,54,60])。

本脚本构建一个与真实渲染误差 ~3/255 的前向代理(已验证),
直接以「前向输出 vs X-Rite sRGB 的欧氏距离」为目标最小二乘优化 3x3 CCM。

用法: python optimize_ccm_end2end.py
"""
import os
import sys
import time
import glob
import numpy as np
import cv2
import yaml
from scipy.optimize import least_squares

import joint_ccm_calibrate as J  # get_linear_img / build_config / run_standard / find_grid / ...

ISP_DIR = J.ISP_DIR
CONFIG_PATH = os.path.join(ISP_DIR, "config", "configs.yml")
REF = J.REF  # X-Rite sRGB (0-255)

# ---------- 前向代理 ----------
_cfg = None
_lut = None
M_FWD = np.array([[77, 150, 29], [-43, -85, 128], [128, -107, -21]], dtype=np.float64)  # /256
M_INV = np.array([[64, 0, 90], [64, -22, -46], [64, 113, 0]], dtype=np.float64)          # >>6
CSE_GAIN = 1.5

def _load():
    global _cfg, _lut
    with open(CONFIG_PATH, encoding="utf-8") as f:
        _cfg = yaml.safe_load(f)
    _lut = np.array(_cfg["gamma_correction"]["gamma_lut_12"], dtype=np.float64)

def csc_exact(g12, gain=CSE_GAIN):
    """12bit gamma RGB -> 8bit YUV (含 CSE gain 倍, conv_std=2 偏移)"""
    yuv = g12 @ M_FWD.T / 256.0
    yuv = np.where(yuv >= 0, np.floor(yuv + 0.5), np.ceil(yuv - 0.5))
    yuv[..., 1] *= gain
    yuv[..., 2] *= gain
    yuv[..., 0] += 0
    yuv[..., 1] += 2048
    yuv[..., 2] += 2048
    yuv = np.clip(yuv, 0, 4095)
    yuv8 = np.where(yuv >= 0, np.floor(yuv / 16 + 0.5), np.ceil(yuv / 16 - 0.5))
    return np.clip(yuv8, 0, 255)

def rgbc_exact(yuv8):
    """8bit YUV -> 8bit RGB"""
    rgb = (yuv8 - np.array([0.0, 128, 128])) @ M_INV.T / 64.0
    return np.clip(np.floor(rgb + 0.5), 0, 255)

def csc_smooth(g12, gain=CSE_GAIN):
    """12bit gamma RGB -> 8bit YUV (浮点, 无取整, 给优化器平滑梯度)"""
    yuv = g12 @ M_FWD.T / 256.0
    yuv[..., 1] *= gain
    yuv[..., 2] *= gain
    yuv[..., 0] += 0
    yuv[..., 1] += 2048
    yuv[..., 2] += 2048
    return np.clip(np.clip(yuv, 0, 4095) / 16.0, 0, 255)

def rgbc_smooth(yuv8):
    """8bit YUV -> 8bit RGB (浮点, 无取整)"""
    return np.clip((yuv8 - np.array([0.0, 128, 128])) @ M_INV.T / 64.0, 0, 255)

def forward_exact(M, obs01, gain=CSE_GAIN):
    """完整前向(整数 LUT+取整, 与真实渲染逐位接近)"""
    lin12 = np.clip(obs01 @ M.T * 4095, 0, 4095)
    g12 = _lut[np.round(lin12).astype(int)]
    return rgbc_exact(csc_exact(g12, gain))

def forward_smooth(M, obs01, gain=CSE_GAIN):
    """完整前向(插值 LUT + 浮点, 给优化器平滑梯度)"""
    lin12 = np.clip(obs01 @ M.T * 4095, 0, 4095)
    g12 = np.interp(lin12, np.arange(4096), _lut)
    return rgbc_smooth(csc_smooth(g12, gain))

# ---------- 目标函数 ----------
def build_target(k=1.0):
    """目标参考: k=1.0 用 X-Rite 本色; k>1 更鲜艳 (绕 128 缩放色度)。
    缩放后白/灰保持中性, 饱和色往外推 -> 等价于更高饱和度的标定目标。"""
    return np.clip((REF - 128.0) * k + 128.0, 0, 255)

def build_residual(data, gain=CSE_GAIN, weight=None, target_sat=1.0):
    """返回函数 r(M_flat) -> 所有有效块 (前向-目标) 展平
    weight: None=等权; 'neutral3'=中性块×3(白/灰块更受重视); callable=per-patch 权重
    target_sat: 目标饱和度缩放 (见 build_target)"""
    T = build_target(target_sat)
    def r(M_flat):
        M = M_flat.reshape(3, 3)
        errs = []
        for d in data:
            v = d["valid"]
            e = forward_smooth(M, d["obs"][v], gain) - T[v]
            if weight is None:
                pass
            elif weight == "neutral3":
                # 中性块索引: 白18 灰8:19 灰65:20 灰5:21 灰35:22 黑23
                w = np.ones(24); w[18:24] *= 3.0
                e = e * w[v, None]
            elif callable(weight):
                e = e * weight(v, d)
            errs.append(e)
        return np.concatenate(errs).ravel()
    return r

# ---------- main ----------
def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--gain", type=float, default=1.5, help="CSE saturation_gain (默认 1.5)")
    ap.add_argument("--weight", choices=["equal", "neutral3"], default="equal",
                    help="中性块权重: equal=等权 / neutral3=中性块×3")
    ap.add_argument("--target_sat", type=float, default=1.0,
                    help="目标饱和度缩放 (绕128, >1 更鲜艳, 默认 1.0 = X-Rite 本色)")
    args = ap.parse_args(argv)
    gain = args.gain
    weight = None if args.weight == "equal" else args.weight
    ts = args.target_sat

    _load()
    ts_sfx = f"_ts{int(ts * 100)}" if abs(ts - 1.0) > 1e-6 else ""
    tag = "" if (gain == 1.5 and weight is None and ts_sfx == "") else \
        f"_g{int(gain * 100)}" + ("" if weight is None else "_n3") + ts_sfx
    print("=" * 72)
    print(f"端到端 CCM 优化: 以实际渲染输出为目标 (补偿 gamma LUT + CSE {gain}x)"
          f"{' 中性×3' if weight == 'neutral3' else ''}"
          f"{f' 目标饱和度×{ts}' if ts_sfx else ''}")
    print("=" * 72)

    # 1. 逐图线性 obs + 锚点 (复用 joint_ccm_calibrate)
    data = []
    for label, fname in J.RAW_LABELS:
        t0 = time.time()
        cfg = J.build_config(f"linear_{label}", fname)
        linear, bit = J.get_linear_img(cfg)
        orig = J.imread_u(os.path.join(J.OUT_DIR, f"{label}_orig.png"))
        score, anchor, de = J.find_grid(orig)
        ax, ay = anchor
        obs = np.array([J.sample_patch(linear, ax + c * J.DXDY[0], ay + r * J.DXDY[1])
                        for r in range(4) for c in range(6)])
        valid = obs.max(axis=1) <= J.SAT_MAX
        print(f"  {label}: 锚点=({ax},{ay}) 饱和排除 {24 - valid.sum()}/24 ({time.time()-t0:.1f}s)")
        data.append(dict(label=label, fname=fname, obs=obs, valid=valid, anchor=anchor))

    # 2. 验证代理对当前 M_joint 的忠实度 (在当前 gain 语义下)
    joint_yml = os.path.join(J.CONFIGS_DIR, "ccm_joint_raw.yml")
    with open(joint_yml, encoding="utf-8") as f:
        ccm = yaml.safe_load(f)["color_correction_matrix"]
    M0 = np.array([ccm["corrected_red"], ccm["corrected_green"], ccm["corrected_blue"]], dtype=float)
    errs = []
    for d in data:
        v = d["valid"]
        errs.append(np.abs(forward_exact(M0, d["obs"][v], gain) - REF[v]).max())
    print(f"\n[验证] 前向代理 vs X-Rite 参考: 起始 M_joint 各图 maxAbs 误差 = "
          f"{[f'{e:.1f}' for e in errs]} (越小代理越贴近渲染语义)")

    # 3. 优化
    print("\n[优化] least_squares (LM), 初始 = M_joint")
    from scipy.optimize import minimize
    r = build_residual(data, gain=gain, weight=weight, target_sat=ts)
    n0 = np.linalg.norm(r(M0.ravel()))
    res = least_squares(r, M0.ravel(), method="lm", max_nfev=3000, ftol=1e-12, xtol=1e-12, gtol=1e-12)
    n1 = np.linalg.norm(res.fun)
    if res.nfev < 5 or n1 >= n0 * 0.999:
        # LM 卡死(离散/平坦) -> 退到 Powell(无梯度) 直接最小化均方ΔE
        print(f"  LM 未收敛 (nfev={res.nfev}, {n0:.1f}->{n1:.1f}), 改用 Powell")
        def obj(m):
            return np.mean(r(m) ** 2)
        pr = minimize(obj, M0.ravel(), method="Powell", options={"maxiter": 3000, "ftol": 1e-14})
        M_opt = pr.x.reshape(3, 3)
        n1 = np.linalg.norm(r(M_opt.ravel()))
        print(f"  Powell 迭代 {pr.nit} 次, 残差范数 {n0:.1f} -> {n1:.1f}")
    else:
        M_opt = res.x.reshape(3, 3)
    print(f"  最终残差范数 = {n1:.1f} (起始 = {n0:.1f})")
    print("  M_opt:")
    for i, name in enumerate(("corrected_red", "corrected_green", "corrected_blue")):
        print(f"    {name:16s}: {[round(M_opt[i, j], 4) for j in range(3)]}")
    print(f"    行和: {[round(M_opt[i].sum(), 3) for i in range(3)]}")

    # 4. 保存 (非默认配置加后缀)
    out_yml = os.path.join(J.CONFIGS_DIR, f"ccm_end2end{tag}_raw.yml")
    calib = {"color_correction_matrix": {
        "corrected_red": M_opt[0].tolist(),
        "corrected_green": M_opt[1].tolist(),
        "corrected_blue": M_opt[2].tolist(),
    }}
    with open(out_yml, "w", encoding="utf-8") as f:
        yaml.dump(calib, f, allow_unicode=True)
    print(f"  已保存: {out_yml}")

    # 5. 渲染 + 测 ΔE (渲染配置同时注入 saturation_gain)
    sfx = tag.replace("_n3", "")          # 图文件后缀只带 gain (中性权重不改变渲染)
    print(f"\n[渲染] {{label}}_e2e{sfx}.png")
    for label, fname in J.RAW_LABELS:
        cfg = J.build_config(f"{label}_e2e{sfx}", fname, out_yml, saturation_gain=gain)
        img, dt = J.run_standard(cfg, J.RAW_DIR)
        if img is not None:
            cv2.imwrite(os.path.join(J.OUT_DIR, f"{label}_e2e{sfx}.png"),
                        cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            print(f"  {label:<24} done ({dt:.1f}s)")

    print("\n[对比] 渲染域 ΔE (同锚点)")
    print(f"  {'图像':<24} {'orig':>7} {'线性域M_joint':>12} {'端到端M_e2e':>10} {'orig→e2e':>8}")
    for d in data:
        label = d["label"]
        ax, ay = d["anchor"]
        de_o, _ = J.grid_de(J.imread_u(os.path.join(J.OUT_DIR, f"{label}_orig.png")), ax, ay, *J.DXDY)
        de_j, _ = J.grid_de(J.imread_u(os.path.join(J.OUT_DIR, f"{label}_joint.png")), ax, ay, *J.DXDY)
        de_e, _ = J.grid_de(J.imread_u(os.path.join(J.OUT_DIR, f"{label}_e2e{sfx}.png")), ax, ay, *J.DXDY)
        print(f"  {label:<24} {de_o:>5.1f} {de_j:>12.1f} {de_e:>10.1f} {(1 - de_e / de_o) * 100:>7.1f}%")

    # 6. 端到端 M 在每图逐块 ΔE (只列最差块, 对照线性域 M_joint)
    print("\n[逐块] 每图最差 5 块 (端到端 M_e2e vs 线性域 M_joint 的 ΔE)")
    names = ['深肤','浅肤','蓝天','绿叶','蓝花','蓝绿','橙','紫蓝','中红','紫','黄绿','橙黄',
             '蓝','绿','红','黄','品红','青','白','灰8','灰65','灰5','灰35','黑']
    for d in data:
        label = d["label"]
        ax, ay = d["anchor"]
        img_e = J.imread_u(os.path.join(J.OUT_DIR, f"{label}_e2e{sfx}.png"))
        img_j = J.imread_u(os.path.join(J.OUT_DIR, f"{label}_joint.png"))
        de_e = np.array([np.linalg.norm(J.sample_patch(img_e, ax + c * J.DXDY[0], ay + r * J.DXDY[1]) - REF[i])
                         for i, (r, c) in enumerate([(r, c) for r in range(4) for c in range(6)])])
        de_j = np.array([np.linalg.norm(J.sample_patch(img_j, ax + c * J.DXDY[0], ay + r * J.DXDY[1]) - REF[i])
                         for i, (r, c) in enumerate([(r, c) for r in range(4) for c in range(6)])])
        order = np.argsort(-de_e)[:5]
        print(f"  {label}:")
        for i in order:
            print(f"    #{i+1:02d} {names[i]:<3} ΔE e2e={de_e[i]:5.1f}  (线性域M_joint={de_j[i]:5.1f})")

    return 0

if __name__ == "__main__":
    sys.exit(main())
