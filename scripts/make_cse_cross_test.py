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

def draw_badge(dr, x, y, text, font, pad_h=13, pad_v=8):
    lw = dr.textlength(text, font=font)
    dr.rounded_rectangle([x, y, x + lw + pad_h * 2, y + int(font.size * 1.3) + pad_v * 2],
                         radius=10, fill=(25, 25, 25))
    dr.text((x + pad_h, y + pad_v), text, fill=(255, 255, 255), font=font)

def make_matrix(name, panels, out_path):
    """panels: [(img, label), ...] 顺序 TL, TR, BL, BR"""
    h, w = panels[0][0].shape[:2]
    gap, band = 8, 78
    cw = w * 2 + gap
    ch = band + h * 2 + gap
    canvas = np.full((ch, cw, 3), 245, dtype=np.uint8)
    pos = [(0, band), (w + gap, band), (0, band + h + gap), (w + gap, band + h + gap)]
    for (panel, _label), (px, py) in zip(panels, pos):
        canvas[py:py + h, px:px + w] = panel
    img = Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
    dr = ImageDraw.Draw(img)
    f_title = ImageFont.truetype(FONT, 27)
    f_sub = ImageFont.truetype(FONT, 19)
    f_badge = ImageFont.truetype(FONT, 24)
    title = "AWB = PCA  |  %s  |  CCM x CSE saturation" % name
    tw = dr.textlength(title, font=f_title)
    dr.text(((cw - tw) / 2, 8), title, fill=(20, 20, 20), font=f_title)
    # 列头：saturation 1.5 | 1.0（行信息由每个面板的徽标承载）
    for cx, t in ((w / 2, "CSE sat 1.5"), (w + gap + w / 2, "CSE sat 1.0")):
        wt = dr.textlength(t, font=f_sub)
        dr.text((cx - wt / 2, 44), t, fill=(90, 90, 90), font=f_sub)
    for (panel, label), (px, py) in zip(panels, pos):
        draw_badge(dr, px + 14, py + 14, label, f_badge, pad_h=11, pad_v=6)
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
