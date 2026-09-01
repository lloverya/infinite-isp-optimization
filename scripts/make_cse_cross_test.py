# -*- coding: utf-8 -*-
"""
CSE 饱和度 × CCM 交叉消融测试。

对每张图跑两个新组合：
  - E(原项目, 原CCM) + saturation_gain=1.0   （原CCM 去掉 1.5× 提饱和后的表现）
  - D(优化项目, 优化CCM) + saturation_gain=1.5（优化CCM 回到默认高饱和的表现）
并复用之前 A/B 已渲染的基线：
  - E@1.5（原CCM 默认饱和）、D@1.0（优化CCM 定稿饱和）
拼成 2×2 消融矩阵：行=CCM(原/优化)，列=CSE饱和(1.5/1.0)。
结果放 result/04_测试图像对比/CSE交叉测试/<图名>.png。
结束后恢复两项目的 configs.yml 与 isp_pipeline.py。
"""
import os, sys, glob, re, time, subprocess
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont

ROOT_E = r"E:\Infinite-ISP-main\Infinite-ISP-main"
ROOT_D = r"D:\Infinite-ISP-main"
RESULT = r"F:\Claude code\ISP_learning\result\04_测试图像对比\CSE交叉测试"
FONT = r"C:\Windows\Fonts\arialbd.ttf"
if not os.path.exists(FONT):
    FONT = r"C:\Windows\Fonts\msyhbd.ttc"

# X-Rite ColorChecker Classic 24 参考 sRGB（脚本自包含，与 detect_colorchecker 同表）
X_RITE = np.array([
    [115, 82, 68], [194, 150, 130], [98, 122, 157], [87, 108, 67],
    [133, 128, 177], [103, 189, 170], [214, 126, 44], [80, 91, 166],
    [193, 90, 99], [94, 60, 108], [157, 188, 26], [224, 163, 46],
    [56, 61, 150], [70, 148, 73], [175, 54, 60], [231, 199, 31],
    [187, 86, 149], [8, 133, 161], [243, 243, 242], [200, 200, 200],
    [160, 160, 160], [122, 122, 121], [85, 85, 85], [52, 52, 52],
], dtype=float)
DXDY = (218, 207)  # 模板扫描网格间隔 (validate_ccm_raw 标定)

def grid_de(img, ax, ay, dx=DXDY[0], dy=DXDY[1], half=28):
    """在渲染图 img 上以 (ax,ay) 为锚点采样 24 块，返回 (平均 ΔE, 逐块 ΔE)。"""
    de = np.zeros(24)
    for r in range(4):
        for c in range(6):
            cx, cy = ax + c * dx, ay + r * dy
            y1, y2 = max(0, int(cy) - half), min(img.shape[0], int(cy) + half)
            x1, x2 = max(0, int(cx) - half), min(img.shape[1], int(cx) + half)
            if y2 <= y1 or x2 <= x1:
                return 1e9, None
            obs = img[y1:y2, x1:x2].mean(axis=(0, 1))
            de[r * 6 + c] = np.linalg.norm(obs - X_RITE[r * 6 + c])
    return de.mean(), de

def find_grid(img, dx=DXDY[0], dy=DXDY[1]):
    """模板扫描定位 4x6 ColorChecker 网格，返回 (最小平均ΔE, 锚点)。"""
    best = (1e9, None)
    for ay in range(150, 1200, 30):
        for ax in range(200, 1900, 30):
            score, _ = grid_de(img, ax, ay, dx, dy)
            if score < best[0]:
                best = (score, (ax, ay))
    return best

SCENES = [
    ("ColorChecker", "ColorChecker_2592x1536_12bits_RGGB.raw", "./in_frames/normal",
     "Out_ColorChecker_2592x1536_12bits_RGGB",
     {"E": 43.67, "D": 24.63}),   # 基线：E@1.5 / D@1.0 的 A/B 实测时间
    ("Indoor1", "Indoor1_2592x1536_12bit_RGGB.raw", "./in_frames/normal/data",
     "Out_Indoor1_2592x1536_12bit_RGGB",
     {"E": 43.36, "D": 23.82}),
]

def read_text(p):
    with open(p, encoding="utf-8") as f:
        return f.read()

def write_text(p, s):
    with open(p, "w", encoding="utf-8", newline="\n") as f:
        f.write(s)

def read_bytes(p):
    with open(p, "rb") as f:
        return f.read()

def write_bytes(p, b):
    with open(p, "wb") as f:
        f.write(b)

def set_config(proj, fname, rawdata, sat):
    cfg = os.path.join(proj, "config", "configs.yml")
    s = read_text(cfg)
    s = re.sub(r'filename:\s*"[^"]*"', 'filename: "%s"' % fname, s, count=1)
    idx = s.index("color_saturation_enhancement:")
    s = s[:idx] + re.sub(r'(saturation_gain:\s*)[\d.]+', r'\g<1>%g' % sat,
                         s[idx:], count=1)
    write_text(cfg, s)
    pipe = os.path.join(proj, "isp_pipeline.py")
    s = read_text(pipe)
    s = re.sub(r'RAW_DATA\s*=\s*"[^"]*"', 'RAW_DATA = "%s"' % rawdata, s, count=1)
    write_text(pipe, s)

def run_pipeline(proj):
    before = set(glob.glob(os.path.join(proj, "out_frames", "*.png")))
    t0 = time.time()
    r = subprocess.run([sys.executable, "isp_pipeline.py"], cwd=proj,
                       capture_output=True, text=True, encoding="utf-8", errors="replace",
                       timeout=600)
    wall = time.time() - t0
    if r.returncode != 0:
        raise RuntimeError("pipeline failed in %s\n%s" % (proj, r.stderr[-1500:]))
    m = re.search(r"Pipeline Elapsed Time:\s*([\d.]+)s", r.stdout)
    parsed = float(m.group(1)) if m else wall
    after = set(glob.glob(os.path.join(proj, "out_frames", "*.png")))
    new = [p for p in after - before if p.endswith(".png")]
    if not new:
        raise RuntimeError("no new output png in %s" % proj)
    return max(new, key=os.path.getmtime), parsed

# A/B 实测的基线输出（E@1.5 / D@1.0），固定文件名避免被新跑的输出污染
BASELINE_FILES = {
    ("E", "ColorChecker"): r"Out_ColorChecker_2592x1536_12bits_RGGB_20260831_172849.png",
    ("D", "ColorChecker"): r"Out_ColorChecker_2592x1536_12bits_RGGB_20260831_172915.png",
    ("E", "Indoor1"):      r"Out_Indoor1_2592x1536_12bit_RGGB_20260831_173004.png",
    ("D", "Indoor1"):      r"Out_Indoor1_2592x1536_12bit_RGGB_20260831_173030.png",
}

def baseline_path(tag, name):
    fn = BASELINE_FILES[(tag, name)]
    p = os.path.join((ROOT_E if tag == "E" else ROOT_D), "out_frames", fn)
    if not os.path.exists(p):
        raise RuntimeError("baseline missing: %s" % p)
    return p

def draw_badge(img, x, y, text, font, pad_h=18, pad_v=12, fill=(25, 25, 25), alpha=200):
    """在 RGBA 图 img 上叠加半透明圆角徽标：白字实心、背景半透明可透出画面。"""
    dr = ImageDraw.Draw(img)
    lw = dr.textlength(text, font=font)
    bw = int(lw) + pad_h * 2
    bh = int(font.size * 1.3) + pad_v * 2
    overlay = Image.new("RGBA", (bw, bh), (0, 0, 0, 0))
    od = ImageDraw.Draw(overlay)
    od.rounded_rectangle([0, 0, bw, bh], radius=14, fill=fill + (alpha,))
    od.text((pad_h, pad_v), text, fill=(255, 255, 255, 255), font=font)
    img.alpha_composite(overlay, (int(x), int(y)))

def make_matrix(name, panels, out_path, target_w=2568):
    """panels: [(img, label), ...] 顺序 TL, TR, BL, BR。

    ΔE 先在原尺寸面板上计算（网格间距 218/207 按 2592x1536 标定），再整体等比
    缩小画布到约 target_w；文字按 GitHub 内联显示宽度(~950px)反算到约 20-30px
    有效字号。时间徽标与红色 ΔE 徽标在面板左上角**并排**（不叠放），背景半透明，
    避免遮挡图像内容。"""
    h, w = panels[0][0].shape[:2]
    # 1) 渲染域 ΔE（原尺寸，TL 面板定位网格，其余面板同锚点采样）
    tl_rgb = cv2.cvtColor(panels[0][0], cv2.COLOR_BGR2RGB)
    _, anchor = find_grid(tl_rgb)
    des = [grid_de(cv2.cvtColor(p, cv2.COLOR_BGR2RGB), anchor[0], anchor[1])[0]
           for p, _ in panels] if anchor else []
    if des:
        print("    锚点=%s  面板 ΔE = %s" % (anchor, [round(x, 1) for x in des]))
    # 2) 等比缩小面板
    gap = 8
    scale = (target_w - gap) / (2.0 * w)
    nw, nh = int(w * scale), int(h * scale)
    panels = [(cv2.resize(p, (nw, nh), interpolation=cv2.INTER_AREA), l) for p, l in panels]
    band = 170
    cw = nw * 2 + gap
    ch = band + nh * 2 + gap
    canvas = np.full((ch, cw, 3), 245, dtype=np.uint8)
    pos = [(0, band), (nw + gap, band), (0, band + nh + gap), (nw + gap, band + nh + gap)]
    for (panel, _label), (px, py) in zip(panels, pos):
        canvas[py:py + nh, px:px + nw] = panel
    # 3) 叠加标注
    img = Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)).convert("RGBA")
    dr = ImageDraw.Draw(img)
    f_title = ImageFont.truetype(FONT, 80)
    f_sub = ImageFont.truetype(FONT, 56)
    f_badge = ImageFont.truetype(FONT, 58)
    f_de = ImageFont.truetype(FONT, 52)
    title = "AWB = PCA  |  %s  |  CCM x CSE saturation" % name
    tw = dr.textlength(title, font=f_title)
    dr.text(((cw - tw) / 2, 12), title, fill=(20, 20, 20, 255), font=f_title)
    # 列头：saturation 1.5 | 1.0（行信息由每个面板的徽标承载）
    for cx, t in ((nw / 2, "CSE sat 1.5"), (nw + gap + nw / 2, "CSE sat 1.0")):
        wt = dr.textlength(t, font=f_sub)
        dr.text((cx - wt / 2, 98), t, fill=(90, 90, 90, 255), font=f_sub)
    # 时间徽标与 ΔE 徽标并排（同一行，不上下叠放）
    pad_h = 18
    for (panel, label), (px, py) in zip(panels, pos):
        draw_badge(img, px + 20, py + 16, label, f_badge, pad_h=pad_h)
    for (panel, label), (px, py), de in zip(panels, pos, des):
        twb = dr.textlength(label, font=f_badge)
        draw_badge(img, px + 20 + int(twb) + 2 * pad_h + 8, py + 16,
                   "ΔE mean %4.1f" % de, f_de, pad_h=pad_h, fill=(180, 30, 30))
    img = img.convert("RGB")
    img.save(out_path)
    print("  保存矩阵图:", out_path)

def main():
    originals = {}
    for tag, proj in (("E", ROOT_E), ("D", ROOT_D)):
        originals[tag] = (read_bytes(os.path.join(proj, "config", "configs.yml")),
                          read_bytes(os.path.join(proj, "isp_pipeline.py")))
    try:
        for name, fname, rawdata, prefix, base_t in SCENES:
            print("===== %s =====" % name)
            e_bl = baseline_path("E", name)   # E@1.5 基线
            d_bl = baseline_path("D", name)   # D@1.0 基线
            set_config(ROOT_E, fname, rawdata, 1.0)
            e_new, t_e1 = run_pipeline(ROOT_E)       # E@1.0
            set_config(ROOT_D, fname, rawdata, 1.5)
            d_new, t_d15 = run_pipeline(ROOT_D)      # D@1.5
            a = lambda p: cv2.imread(p)
            imgs = {
                "TL": (a(e_bl), "Orig CCM  sat 1.5  %5.2f s" % base_t["E"]),
                "TR": (a(e_new), "Orig CCM  sat 1.0  %5.2f s" % t_e1),
                "BL": (a(d_new), "Opt  CCM  sat 1.5  %5.2f s" % t_d15),
                "BR": (a(d_bl), "Opt  CCM  sat 1.0  %5.2f s" % base_t["D"]),
            }
            # 统一尺寸
            h0, w0 = imgs["TL"][0].shape[:2]
            for k in ("TR", "BL", "BR"):
                if imgs[k][0].shape[:2] != (h0, w0):
                    imgs[k] = (cv2.resize(imgs[k][0], (w0, h0)), imgs[k][1])
            os.makedirs(RESULT, exist_ok=True)
            out_path = os.path.join(RESULT, name + ".png")
            make_matrix(name,
                        [imgs["TL"], imgs["TR"], imgs["BL"], imgs["BR"]],
                        out_path)
    finally:
        for tag, proj in (("E", ROOT_E), ("D", ROOT_D)):
            cfg_b, pipe_b = originals[tag]
            write_bytes(os.path.join(proj, "config", "configs.yml"), cfg_b)
            write_bytes(os.path.join(proj, "isp_pipeline.py"), pipe_b)
        print("已恢复两个项目的 configs.yml 与 isp_pipeline.py")

if __name__ == "__main__":
    main()
