import os
import sys
import cv2
import numpy as np
import pandas as pd
from glob import glob
import re
from Metrics import compute_metrics  # ✅ 确保 compute_metrics.py 已包含 compute_vif 函数

# 添加 Metrics 模块路径（若必要）
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'Metrics')))

# 自然排序函数
def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', os.path.basename(s))]

# 路径设置
ir_dir = './TestDatasets/LLVIP/ir/'
vi_dir = './TestDatasets/LLVIP/vi/'
fused_root = './Results/ablation/'
output_dir = './Metrics/output/no_eca'
os.makedirs(output_dir, exist_ok=True)

# 获取所有融合方法文件夹名
methods = [d for d in os.listdir(fused_root) if os.path.isdir(os.path.join(fused_root, d))]

# 图像格式支持
img_formats = ('*.jpg', '*.png', '*.bmp', '*.tif', '*.tiff')

# 总表容器
all_results = []

for method in methods:
    print(f"\n🔍 正在处理方法: {method}")
    method_dir = os.path.join(fused_root, method)
    fused_images = []
    for fmt in img_formats:
        fused_images.extend(glob(os.path.join(method_dir, fmt)))

    # 自然排序
    fused_images = sorted(fused_images, key=natural_sort_key)

    method_results = []

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
            vif = compute_metrics.compute_vif(img_ir, img_vi, img_fused)
            # qabf = compute_metrics.compute_qabf(img_ir, img_vi, img_fused)
            qcv = compute_metrics.compute_qcv(img_ir, img_vi, img_fused)
            var = compute_metrics.compute_variance(img_ir, img_vi, img_fused)
            fmi = compute_metrics.compute_fmi(img_ir, img_vi, img_fused)
            sf = compute_metrics.compute_sf(img_fused)

            sd = compute_metrics.compute_sd(img_fused)
            cc = compute_metrics.compute_cc(img_ir, img_vi, img_fused)
            scd = compute_metrics.compute_scd(img_ir, img_vi, img_fused)
            # mef_ssim = compute_metrics.compute_mef_ssim(img_ir, img_vi, img_fused)
            # psnr = compute_metrics.compute_psnr(img_ir, img_vi, img_fused)

            row = {
                'Method': method,
                'Image': name,
                'Entropy': entropy,
                'AG': ag,
                'VIF': vif,
                'Variance': var,
                'SF': sf,
                'CC': cc,
                'QCV':qcv
            }

            method_results.append(row)
            all_results.append(row)

            print(f"[成功] 计算指标: {name}")
        except Exception as e:
            print(f"[错误] 处理 {name} 时失败: {str(e)}")

    # 每方法单独保存 CSV
    df_method = pd.DataFrame(method_results)

    # 计算平均值与标准差
    avg_row = df_method.mean(numeric_only=True)
    std_row = df_method.std(numeric_only=True)

    avg_row['Method'] = method
    avg_row['Image'] = 'Average'
    std_row['Method'] = method
    std_row['Image'] = 'StdDev'

    df_method = pd.concat([df_method, pd.DataFrame([avg_row, std_row])], ignore_index=True)

    # 保存
    per_method_path = os.path.join(output_dir, f'fusion_metrics_TNO_{method}.csv')
    df_method.to_csv(per_method_path, index=False)
    print(f"[完成] {method} 结果已保存至: {per_method_path}")

# 总表 CSV
df_all = pd.DataFrame(all_results)

# 排序
df_all_sorted = pd.concat([
    df_all[df_all['Method'] == m].sort_values(by='Image', key=lambda col: col.map(lambda x: natural_sort_key(x)))
    for m in sorted(df_all['Method'].unique())
])

summary_path = os.path.join(output_dir, 'fusion_metrics_TNO_all_methods.csv')

# 添加平均值和标准差
mean_rows = []
std_rows = []

for method in sorted(df_all['Method'].unique()):
    subset = df_all[df_all['Method'] == method]
    mean_row = subset.mean(numeric_only=True)
    std_row = subset.std(numeric_only=True)

    mean_row['Method'] = method
    mean_row['Image'] = 'Average'
    std_row['Method'] = method
    std_row['Image'] = 'StdDev'

    mean_rows.append(mean_row)
    std_rows.append(std_row)

df_all_final = pd.concat([df_all_sorted, pd.DataFrame(mean_rows + std_rows)], ignore_index=True)
df_all_final.to_csv(summary_path, index=False)

print(f"\n✅ 所有方法总表已保存至: {summary_path}")
