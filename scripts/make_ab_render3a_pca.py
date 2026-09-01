# -*- coding: utf-8 -*-
"""
生成 render_3a=true + 白平衡算法=pca 配置下的 原ISP(E) vs 优化ISP(D) 对比图。

流程：对每张图（ColorChecker / Indoor1 / Outdoor1-4）——
  1. 在 E(原项目) 与 D(优化项目) 的 configs.yml 里设 filename、isp_pipeline.py 里设 RAW_DATA
  2. 各跑一次 isp_pipeline.py，解析 "Pipeline Elapsed Time"
  3. 计算两输出图 PSNR（优化 vs 原，参考=原输出）
  4. 拼成左右对比图（左=原ISP，右=优化ISP），顶栏居中标注 PSNR，两图左上角各标处理时间
  5. 保存到 result/04_测试图像对比/内置RAW/<图名>/
结束后恢复两个项目的 configs.yml 与 isp_pipeline.py。

用法：python make_ab_render3a_pca.py [name_filter]
      name_filter 非空时只处理名称包含它的场景（如 "outdoor" 只跑 Outdoor1-4）。
"""
import os, sys, glob, re, time, subprocess
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont

ROOT_E = r"E:\Infinite-ISP-main\Infinite-ISP-main"
ROOT_D = r"D:\Infinite-ISP-main"
RESULT = r"F:\Claude code\ISP_learning\result\04_测试图像对比\内置RAW"
FONT = r"C:\Windows\Fonts\arialbd.ttf"
if not os.path.exists(FONT):
    FONT = r"C:\Windows\Fonts\msyhbd.ttc"

IMAGES = [
    ("ColorChecker", "ColorChecker_2592x1536_12bits_RGGB.raw", "./in_frames/normal"),
    ("Indoor1", "Indoor1_2592x1536_12bit_RGGB.raw", "./in_frames/normal/data"),
    ("Outdoor1", "Outdoor1_2592x1536_12bit_RGGB.raw", "./in_frames/normal/data"),
    ("Outdoor2", "Outdoor2_2592x1536_12bit_RGGB.raw", "./in_frames/normal/data"),
    ("Outdoor3", "Outdoor3_2592x1536_12bit_RGGB.raw", "./in_frames/normal/data"),
    ("Outdoor4", "Outdoor4_2592x1536_12bit_RGGB.raw", "./in_frames/normal/data"),
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

def set_config(proj, fname, rawdata):
    cfg = os.path.join(proj, "config", "configs.yml")
    s = read_text(cfg)
    s = re.sub(r'filename:\s*"[^"]*"', 'filename: "%s"' % fname, s, count=1)
    write_text(cfg, s)
    pipe = os.path.join(proj, "isp_pipeline.py")
    s = read_text(pipe)
    s = re.sub(r'RAW_DATA\s*=\s*"[^"]*"', 'RAW_DATA = "%s"' % rawdata, s, count=1)
    write_text(pipe, s)

def run_pipeline(proj):
    """运行 isp_pipeline.py，返回 (输出png路径, 解析到的耗时秒数)"""
    before = glob.glob(os.path.join(proj, "out_frames", "*.png"))
    t0 = time.time()
    r = subprocess.run([sys.executable, "isp_pipeline.py"], cwd=proj,
                       capture_output=True, text=True, encoding="utf-8", errors="replace",
                       timeout=600)
    wall = time.time() - t0
    if r.returncode != 0:
        raise RuntimeError("pipeline failed in %s\n%s" % (proj, r.stderr[-1500:]))
    m = re.search(r"Pipeline Elapsed Time:\s*([\d.]+)s", r.stdout)
    parsed = float(m.group(1)) if m else wall
    after = glob.glob(os.path.join(proj, "out_frames", "*.png"))
    new = sorted(set(after) - set(before), key=os.path.getmtime)
    if not new:
        raise RuntimeError("no new output png in %s" % proj)
    return new[-1], parsed

def psnr(a, b):
    a = a.astype(np.float64); b = b.astype(np.float64)
    mse = np.mean((a - b) ** 2)
    return float("inf") if mse == 0 else 10 * np.log10(255.0 ** 2 / mse)

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

def make_cmp(name, left, right, tE, tD, psnr_v, out_path, target_w=2568):
    """左=原ISP(E)，右=优化ISP(D)；顶栏标题+居中PSNR；两图左上角各标处理时间。

    输出画布宽度约 target_w：面板先等比缩小，文字按 GitHub 内联显示宽度(~950px)
    反算到约 20-30px 有效字号（可读但不挡画面），徽标半透明。"""
    h, w = left.shape[:2]
    gap = 8
    scale = (target_w - gap) / (2.0 * w)
    nw, nh = int(w * scale), int(h * scale)
    left = cv2.resize(left, (nw, nh), interpolation=cv2.INTER_AREA)
    right = cv2.resize(right, (nw, nh), interpolation=cv2.INTER_AREA)
    band = 160
    canvas = np.full((band + nh, nw * 2 + gap, 3), 245, dtype=np.uint8)
    canvas[band:, :nw] = left
    canvas[band:, nw + gap:] = right
    img = Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)).convert("RGBA")
    dr = ImageDraw.Draw(img)
    f_title = ImageFont.truetype(FONT, 80)
    f_psnr = ImageFont.truetype(FONT, 70)
    f_badge = ImageFont.truetype(FONT, 60)
    title = "AWB = PCA  |  %s" % name
    tw = dr.textlength(title, font=f_title)
    dr.text(((nw * 2 + gap - tw) / 2, 12), title, fill=(20, 20, 20, 255), font=f_title)
    ptxt = "PSNR = ∞" if psnr_v == float("inf") else "PSNR = %.2f dB" % psnr_v
    pw = dr.textlength(ptxt, font=f_psnr)
    dr.text(((nw * 2 + gap - pw) / 2, 90), ptxt, fill=(200, 30, 30, 255), font=f_psnr)
    draw_badge(img, 24, band + 16, "Original ISP   %.2f s" % tE, f_badge)
    draw_badge(img, nw + gap + 24, band + 16, "Optimized ISP  %.2f s" % tD, f_badge)
    img = img.convert("RGB")
    img.save(out_path)
    print("  保存对比图:", out_path)

def main():
    filter_ = sys.argv[1].lower() if len(sys.argv) > 1 else None
    scenes = [(n, f, r) for n, f, r in IMAGES
              if not filter_ or filter_ in n.lower()]
    originals = {}
    for tag, proj in (("E", ROOT_E), ("D", ROOT_D)):
        originals[tag] = (read_bytes(os.path.join(proj, "config", "configs.yml")),
                          read_bytes(os.path.join(proj, "isp_pipeline.py")))
    try:
        for name, fname, rawdata in scenes:
            print("===== %s =====" % name)
            outs, times = {}, {}
            for tag, proj in (("E", ROOT_E), ("D", ROOT_D)):
                set_config(proj, fname, rawdata)
                out, parsed = run_pipeline(proj)
                outs[tag] = out
                times[tag] = parsed
                print("  [%s] 耗时 %.2f s  ->  %s" % (tag, parsed, os.path.basename(out)))
            a = cv2.imread(outs["E"]); b = cv2.imread(outs["D"])
            if a.shape != b.shape:
                b = cv2.resize(b, (a.shape[1], a.shape[0]))
            p = psnr(a, b)
            print("  PSNR(opt vs orig) = %s dB" % ("∞" if p == float("inf") else "%.2f" % p))
            folder = os.path.join(RESULT, name)
            os.makedirs(folder, exist_ok=True)
            out_path = os.path.join(folder, "compare_render3a_pca_原vs优化.png")
            make_cmp(name, a, b, times["E"], times["D"], p, out_path)
    finally:
        for tag, proj in (("E", ROOT_E), ("D", ROOT_D)):
            cfg_b, pipe_b = originals[tag]
            write_bytes(os.path.join(proj, "config", "configs.yml"), cfg_b)
            write_bytes(os.path.join(proj, "isp_pipeline.py"), pipe_b)
        print("已恢复两个项目的 configs.yml 与 isp_pipeline.py")

if __name__ == "__main__":
    main()
