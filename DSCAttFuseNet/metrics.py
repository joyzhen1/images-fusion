import os
import sys
import cv2
import numpy as np
import pandas as pd
from glob import glob

# 导入 compute_metrics 中的指标函数
from Metrics import compute_metrics  # ✅ 修正关键点

# 路径设置
ir_dir = './TestDatasets/TNO/ir/'
vi_dir = './TestDatasets/TNO/vi/'
fused_dir = './Results/TNO/'

# 支持多种图像格式
img_formats = ('*.jpg', '*.png', '*.bmp', '*.tif', '*.tiff')
fused_images = []
for fmt in img_formats:
    fused_images.extend(sorted(glob(os.path.join(fused_dir, fmt))))

# 遍历每一张融合图像
results = []
for fused_path in fused_images:
    name = os.path.basename(fused_path)
    ir_path = os.path.join(ir_dir, name)
    vi_path = os.path.join(vi_dir, name)

    if not os.path.exists(ir_path) or not os.path.exists(vi_path):
        print(f"[跳过] 缺少源图像: {name}")
        continue

    img_ir = cv2.imread(ir_path)
    img_vi = cv2.imread(vi_path)
    img_fused = cv2.imread(fused_path)

    try:
        entropy = compute_metrics.compute_entropy(img_fused)
        ag = compute_metrics.compute_ag(img_fused)
        mi = compute_metrics.compute_mi(img_ir, img_vi, img_fused)
        ssim = compute_metrics.compute_ssim(img_ir, img_vi, img_fused)
        qabf = compute_metrics.compute_qabf(img_ir, img_vi, img_fused)
        var = compute_metrics.compute_variance(img_ir, img_vi, img_fused)

        results.append({
            'Image': name,
            'Entropy': entropy,
            'AG': ag,
            'MI': mi,
            'SSIM': ssim,
            'Qabf': qabf,
            'Variance': var
        })
        print(f"[成功] 计算指标: {name}")
    except Exception as e:
        print(f"[错误] 处理 {name} 时失败: {str(e)}")

# 保存 CSV 文件
os.makedirs('./Metrics/output', exist_ok=True)
csv_path = './Metrics/output/fusion_metrics_TNO.csv'
df = pd.DataFrame(results)
df.to_csv(csv_path, index=False)
print(f"[完成] 结果已保存至: {csv_path}")
