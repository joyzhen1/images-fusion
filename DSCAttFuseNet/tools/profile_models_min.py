"""
CPU-optimized minimal profiler for TF1.x models (Params + Latency/FPS).

特点：
- 面向 CPU：默认分辨率 256x256、warmup=3、runs=20，并启用并行线程参数
- 不加载 checkpoint，只构建前向图并用随机输入测延迟
- 直接在文件底部填你的两个 model.py 路径，PyCharm 点运行即可

测完 Ours(DSC) 与 StdConv 两个版本的：
  - Params(M)
  - Latency(ms) 平均 & P90
  - FPS (= 1000 / Latency)

最终论文里 ΔLatency(%) = (Latency_std - Latency_dsc) / Latency_std * 100%
"""

import os, time, importlib.util
import numpy as np
import tensorflow as tf

tf.compat.v1.disable_eager_execution()

# ========= 1) 在这里填你的两个 model.py 绝对路径 =========
MODELS = [
    ("Ours(DSC)", r".\models\ours\model.py"),        # 带 DSC 的版本
    ("StdConv",   r".\models\StdCon\model.py"),      # 将 DSC 改为标准卷积的版本
]

# ========= 2) CPU 极速默认参数（可按需调整） =========
H, W   = 256, 256   # 初次跑建议 256x256；最后一轮可改回 480x640
BS     = 1
WARMUP = 3
RUNS   = 20
DEVICE = -1         # -1 表示 CPU；如果装了 GPU 且想指定第0张卡：改为 0

# 可选：设置 CPU 并行线程（对 Intel MKL/OneDNN 一般有效）
os.environ.setdefault("OMP_NUM_THREADS", "8")           # 改成你的 CPU 合适线程数
os.environ.setdefault("KMP_BLOCKTIME", "0")
os.environ.setdefault("KMP_AFFINITY", "granularity=fine,compact,1,0")

def dynamic_import_model(path, module_name):
    spec = importlib.util.spec_from_file_location(module_name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

def profile_one(model_py, name):
    # 设备选择：CPU 用 -1（将 CUDA_VISIBLE_DEVICES 设成空）
    if DEVICE >= 0:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(DEVICE)
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

    tf.compat.v1.reset_default_graph()

    cfg = tf.compat.v1.ConfigProto()
    # CPU 并行（这两行对 CPU 有帮助；GPU 环境不影响）
    cfg.intra_op_parallelism_threads = max(1, os.cpu_count() or 4)
    cfg.inter_op_parallelism_threads = 2
    # 即使 CPU 也无妨
    cfg.gpu_options.allow_growth = True

    with tf.compat.v1.Session(config=cfg) as sess:
        ir  = tf.compat.v1.placeholder(tf.float32, [BS, H, W, 1], name='ir')
        viY = tf.compat.v1.placeholder(tf.float32, [BS, H, W, 1], name='viY')

        # 要求你的 model.py 暴露：decomposition(viY, ir) -> (ir_r, vi_e_r, l_r)
        ir_r, vi_e_r, l_r = model_py.decomposition(viY, ir)

        # 以融合亮度为输出节点（覆盖主干路径）；若你想测彩色输出，可改为对应张量
        out = tf.maximum(ir_r, vi_e_r * l_r)

        sess.run(tf.compat.v1.global_variables_initializer())

        # Params (M)
        params = int(np.sum([np.prod(v.shape.as_list()) for v in tf.compat.v1.trainable_variables()]))
        params_m = params / 1e6

        # 随机输入
        feed = {
            ir:  np.random.rand(BS, H, W, 1).astype(np.float32),
            viY: np.random.rand(BS, H, W, 1).astype(np.float32)
        }

        # 预热
        for _ in range(WARMUP):
            sess.run(out, feed_dict=feed)

        # 多次测量
        times = []
        for _ in range(RUNS):
            t0 = time.perf_counter()
            sess.run(out, feed_dict=feed)
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000.0)  # ms

        avg_ms = float(np.mean(times))
        p90_ms = float(np.percentile(times, 90))
        fps = 1000.0 / avg_ms if avg_ms > 0 else 0.0

        return {"Method": name, "Params(M)": params_m, "Latency(ms)": avg_ms, "P90(ms)": p90_ms, "FPS": fps}

def main():
    results = []
    for idx, (name, path) in enumerate(MODELS):
        print("=" * 60)
        print(f"[{idx+1}/{len(MODELS)}] Profiling {name}\n  from {path}")
        if not os.path.isfile(path):
            print(f"!! 找不到文件：{path}")
            continue
        mod = dynamic_import_model(path, f"model_variant_{idx}")
        res = profile_one(mod, name)
        print(f"Result: {res}")
        results.append(res)

    # 打印一个 ΔLatency(%) 便捷行（基于 StdConv 与 Ours 命名）
    names = {r["Method"]: r for r in results}
    if "StdConv" in names and "Ours(DSC)" in names:
        lat_std = names["StdConv"]["Latency(ms)"]
        lat_dsc = names["Ours(DSC)"]["Latency(ms)"]
        delta = (lat_std - lat_dsc) / lat_std * 100.0
        print(f"\nΔLatency vs StdConv: {delta:.2f}% （= (Std - DSC)/Std × 100%）")

    # 保存 CSV/Markdown 到当前目录
    import csv
    csv_path = os.path.abspath("efficiency_results.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["Method","Params(M)","Latency(ms)","P90(ms)","FPS"])
        writer.writeheader()
        for r in results:
            writer.writerow(r)
    print(f"[Saved] CSV -> {csv_path}")

    md_path = os.path.abspath("efficiency_results.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("| Method | Params (M) ↓ | Latency (ms) ↓ | P90 (ms) ↓ | FPS ↑ |\n")
        f.write("|---|---:|---:|---:|---:|\n")
        for r in results:
            f.write(f"| {r['Method']} | {r['Params(M)']:.3f} | {r['Latency(ms)']:.2f} | {r['P90(ms)']:.2f} | {r['FPS']:.2f} |\n")
    print(f"[Saved] Markdown -> {md_path}")

if __name__ == "__main__":
    main()
