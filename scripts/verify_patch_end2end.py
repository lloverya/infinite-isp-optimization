# -*- coding: utf-8 -*-
"""
端到端位级一致性验证: 在 util.save_pipeline_output 处拦截 out_rgb 数组,
存成 npy, 用于比较 补丁版 vs 原版 的最终输出。
用法: python verify_patch_end2end.py <raw_name> <out.npy>
"""
import os, sys
import numpy as np

ISP_DIR = "E:/Infinite-ISP-main/Infinite-ISP-main"
sys.path.append(ISP_DIR)

from util import utils as uu   # infinite_isp 里是 'import util.utils as util', 要打补丁的是这个模块

_capture = {}
def _capture_save(out_file, out_rgb, c_yaml):
    _capture["array"] = np.asarray(out_rgb).copy()
    _capture["shape"] = _capture["array"].shape
    _capture["dtype"] = str(_capture["array"].dtype)

uu.save_pipeline_output = _capture_save

from infinite_isp import InfiniteISP

def main():
    raw_name, out_path = sys.argv[1], sys.argv[2]
    raw_dir = os.path.join(ISP_DIR, "in_frames", "normal")
    config_path = os.path.join(ISP_DIR, "config", "configs.yml")

    isp = InfiniteISP(data_path=raw_dir, config_path=config_path)
    isp.platform["filename"] = f"{raw_name}.raw"
    isp.load_raw()
    isp.run_pipeline(visualize_output=True)

    arr = _capture["array"]
    np.save(out_path, arr)
    print(f"out_rgb: {_capture['shape']} {_capture['dtype']}  -> {out_path}")

if __name__ == "__main__":
    main()
