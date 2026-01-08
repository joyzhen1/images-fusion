# -*- coding: utf-8 -*-
# author:  (xkz)
# time   : 2025/08/10
"""
DSCAttFuseNet 训练脚本（保存最优模型 & 每100 epoch保存一次）
建议文件名：train_dscattfusenet.py
"""

from __future__ import print_function

import os
import random
from glob import glob

import losses
from model import *  # uses your decomposition(vi, ir) defined in model.py

# =============== Constants ===============
batch_size   = 2
patch_size_x = 128
patch_size_y = 128

# =============== Utils ===============
def get_image_paths(directory, extensions=('.jpg', '.png', '.jpeg', '.bmp', '.tif', '.tiff')):
    paths = []
    for ext in extensions:
        paths.extend(glob(os.path.join(directory, '*' + ext)))
    return sorted(paths)

def load_image_auto_gray(image_path):
    """
    加载图像，统一转为灰度图（[H, W]）
    """
    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError("[错误] 无法加载图像: {}".format(image_path))
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img

def load_image_auto_rgb(image_path):
    """
    加载图像，统一转为 RGB 图像（[H, W, 3]）
    """
    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError("[错误] 无法加载图像: {}".format(image_path))
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def to_float01(x):
    # 统一归一化到 [0,1]
    if x.dtype != np.float32:
        x = x.astype(np.float32)
    if x.max() > 1.0:
        x = x / 255.0
    return x

def rgb_ycbcr_np_3(img_rgb):
    """
    输入: [H,W,3] RGB in [0,1] or [0,255]
    输出: [H,W,3] (Y, Cb, Cr)
    """
    if img_rgb.max() > 1.0:
        img_rgb = img_rgb / 255.0
    R = np.expand_dims(img_rgb[:, :, 0], axis=-1)
    G = np.expand_dims(img_rgb[:, :, 1], axis=-1)
    B = np.expand_dims(img_rgb[:, :, 2], axis=-1)
    Y  = 0.299 * R + 0.587 * G + 0.114 * B
    Cb = -0.1687 * R - 0.3313 * G + 0.5 * B + 128/255.0
    Cr =  0.5   * R - 0.4187 * G - 0.0813 * B + 128/255.0
    return np.concatenate([Y, Cb, Cr], axis=-1)

def random_crop_pair(ir_img, vi_y, vi_3, ph, pw):
    """
    ir_img: [H,W]
    vi_y  : [H,W,1]
    vi_3  : [H,W,3]
    返回裁剪块（随机裁剪，训练用）
    """
    H, W = ir_img.shape[:2]
    if H < ph or W < pw:
        pad_h = max(0, ph - H)
        pad_w = max(0, pw - W)
        ir_img = np.pad(ir_img, ((0, pad_h), (0, pad_w)), mode='reflect')
        vi_y   = np.pad(vi_y,   ((0, pad_h), (0, pad_w), (0, 0)), mode='reflect')
        vi_3   = np.pad(vi_3,   ((0, pad_h), (0, pad_w), (0, 0)), mode='reflect')
        H, W = ir_img.shape[:2]
    y0 = np.random.randint(0, H - ph + 1)
    x0 = np.random.randint(0, W - pw + 1)
    return (ir_img[y0:y0+ph, x0:x0+pw],
            vi_y[y0:y0+ph,   x0:x0+pw,   :],
            vi_3[y0:y0+ph,   x0:x0+pw,   :])

def center_crop_pair(ir_img, vi_y, vi_3, ph, pw):
    """
    中心裁剪（验证用，确保每次一致）
    """
    H, W = ir_img.shape[:2]
    if H < ph or W < pw:
        pad_h = max(0, ph - H)
        pad_w = max(0, pw - W)
        ir_img = np.pad(ir_img, ((0, pad_h), (0, pad_w)), mode='reflect')
        vi_y   = np.pad(vi_y,   ((0, pad_h), (0, pad_w), (0, 0)), mode='reflect')
        vi_3   = np.pad(vi_3,   ((0, pad_h), (0, pad_w), (0, 0)), mode='reflect')
        H, W = ir_img.shape[:2]
    y0 = (H - ph) // 2
    x0 = (W - pw) // 2
    return (ir_img[y0:y0+ph, x0:x0+pw],
            vi_y[y0:y0+ph,   x0:x0+pw,   :],
            vi_3[y0:y0+ph,   x0:x0+pw,   :])

# =============== Graph ===============
# 避免一次性占满显存
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.InteractiveSession(config=config)

# 输入占位符（与你现有工程保持一致）
vi         = tf.placeholder(tf.float32, [None, None, None, 1], name='vi')         # VI 的 Y
vi_hist    = tf.placeholder(tf.float32, [None, None, None, 1], name='vi_hist')    # 可选：直方图均衡的 Y（若不用，可直接用 vi）
ir         = tf.placeholder(tf.float32, [None, None, None, 1], name='ir')         # IR 灰度
vi_3       = tf.placeholder(tf.float32, [None, None, None, 3], name='vi_3')       # Y 复制成 3 通道，供感知损失
vi_hist_3  = tf.placeholder(tf.float32, [None, None, None, 3], name='vi_hist_3')  # 与上对应的直方图增强版（可与 vi_3 相同）

# 主网络（来自 model.py 的 decomposition()：encoder(DSC+ECA) + CAM + decoders）
[ir_r, vi_e_r, l_r] = decomposition(vi, ir)

# 将可见增强结果转为 3 通道，供感知损失
vi_e_r_3 = tf.concat([vi_e_r, vi_e_r, vi_e_r], axis=3)

# =============== Losses ===============
def gradient_dir(input_tensor, direction):
    """
    方向梯度并归一化（用于 mutual_i_input_loss / mutual_i_loss）
    """
    smooth_kernel_x = tf.reshape(tf.constant([[0, 0], [-1, 1]], tf.float32), [2, 2, 1, 1])
    smooth_kernel_y = tf.transpose(smooth_kernel_x, [1, 0, 2, 3])
    if direction == "x":
        kernel = smooth_kernel_x
    else:
        kernel = smooth_kernel_y
    gradient_orig = tf.abs(tf.nn.conv2d(input_tensor, kernel, strides=[1, 1, 1, 1], padding='SAME'))
    grad_min = tf.reduce_min(gradient_orig)
    grad_max = tf.reduce_max(gradient_orig)
    grad_norm = tf.div((gradient_orig - grad_min), (grad_max - grad_min + 1e-4))
    return grad_norm

def mutual_i_input_loss(input_I_low, input_im):
    """
    照度-输入一致性
    """
    input_gray = input_im  # 这里 input_im 已经是 [N,H,W,1] 的 Y
    low_gradient_x   = gradient_dir(input_I_low, "x")
    input_gradient_x = gradient_dir(input_gray,   "x")
    x_loss = tf.abs(tf.div(low_gradient_x, tf.maximum(input_gradient_x, 0.01)))

    low_gradient_y   = gradient_dir(input_I_low, "y")
    input_gradient_y = gradient_dir(input_gray,   "y")
    y_loss = tf.abs(tf.div(low_gradient_y, tf.maximum(input_gradient_y, 0.01)))
    mut_loss = tf.reduce_mean(x_loss + y_loss)
    return mut_loss

def mutual_i_loss(input_I_low):
    """
    相互一致性
    """
    low_gradient_x = gradient_dir(input_I_low, "x")
    x_loss = (low_gradient_x) * tf.exp(-10 * (low_gradient_x))
    low_gradient_y = gradient_dir(input_I_low, "y")
    y_loss = (low_gradient_y) * tf.exp(-10 * (low_gradient_y))
    return tf.reduce_mean(x_loss + y_loss)

# --- 原始损失 ---
recon_loss_vi         = tf.reduce_mean(tf.square(vi_e_r * l_r - vi))   # 可见重构
recon_loss_ir         = tf.reduce_mean(tf.square(ir_r - ir))           # 红外重构
i_input_mutual_loss   = mutual_i_input_loss(l_r, vi)                   # 照度-输入一致
per_loss              = losses.Perceptual_Loss(vi_e_r_3, vi_hist_3)    # 感知损失（VGG；需要3通道）
mutual_loss           = mutual_i_loss(l_r)                             # 照度一致

loss_Dsca = 1000.0 * recon_loss_vi \
           + 2000.0 * recon_loss_ir \
           + 7.0    * i_input_mutual_loss \
           + 40.0   * per_loss \
           + 9.0    * mutual_loss

# ==== Edge (Charbonnier) Loss ====
def sobel_xy(x):
    kx = tf.reshape(tf.constant([[1,0,-1],[2,0,-2],[1,0,-1]], tf.float32), [3,3,1,1])
    ky = tf.transpose(kx, [1,0,2,3])
    gx = tf.nn.conv2d(x, kx, [1,1,1,1], 'SAME')
    gy = tf.nn.conv2d(x, ky, [1,1,1,1], 'SAME')
    return gx, gy

def grad_mag(x):
    gx, gy = sobel_xy(x)
    return tf.sqrt(gx*gx + gy*gy + 1e-6)

# 融合亮度（与你现有的一致）
l_r_clip = tf.clip_by_value(l_r, 0.0, 1.0)
fused_vi = vi_e_r * l_r_clip
target_y = tf.maximum(vi, ir)                     # 目标边缘 = 两模态较强者

gF = grad_mag(fused_vi)
gT = grad_mag(tf.stop_gradient(target_y))

eps = 1e-3
with tf.device('/CPU:0'):                         # 避免占显存，可放 CPU
    L_edge = tf.reduce_mean(tf.sqrt(tf.square(gF - gT) + eps*eps), name='L_edge_char')

# 总损失（只改这行权重即可）
lambda_edge = 0.3
loss_total = loss_Dsca + lambda_edge * L_edge

# --- 总损失 ---

# ==== END ====

# =============== Optimizer & Saver ===============
lr = tf.placeholder(tf.float32, name='learning_rate')
optimizer   = tf.train.AdamOptimizer(learning_rate=lr, name='AdamOptimizer')

# 只训练 DSCAttFuseNet 变量（与你工程一致）
var_DSCAttFuseNet   = [v for v in tf.trainable_variables() if 'DSCAttFuseNet' in v.name]
train_op_DSCAttFuseNet = optimizer.minimize(loss_total, var_list=var_DSCAttFuseNet)

sess.run(tf.global_variables_initializer())

# 限制最多保留 5 组普通 ckpt，避免磁盘爆炸
saver_DSCAttFuseNet = tf.train.Saver(var_list=var_DSCAttFuseNet, max_to_keep=5)
print("[*] Initialize model successfully...")

# =============== Data (mixed formats & RGB/Gray) ===============
# 目录可按需修改
train_ir_paths = get_image_paths('./train/infrared')
train_vi_paths = get_image_paths('./train/visible')
eval_ir_paths  = get_image_paths('./dscattfusenet_val/ir/')
eval_vi_paths  = get_image_paths('./dscattfusenet_val/vi')

print('[*] Number of training data_ir/vi: %d / %d' % (len(train_ir_paths), len(train_vi_paths)))
print('[*] Number of eval     data_ir/vi: %d / %d' % (len(eval_ir_paths),  len(eval_vi_paths)))

# 加载训练集（IR 灰度；VI 统一转 Y，并复制成 3 通道副本）
train_ir_data   = []
train_vi_Y_data = []
train_vi_3_data = []

for ir_p, vi_p in zip(train_ir_paths, train_vi_paths):
    ir_img = load_image_auto_gray(ir_p)       # [H,W]
    vi_rgb = load_image_auto_rgb(vi_p)        # [H,W,3] (RGB)
    ir_img = to_float01(ir_img)
    vi_rgb = to_float01(vi_rgb)
    ycbcr  = rgb_ycbcr_np_3(vi_rgb)
    vi_y   = ycbcr[:, :, 0:1]                 # [H,W,1]
    vi_3c  = np.repeat(vi_y, 3, axis=2)       # [H,W,3]

    train_ir_data.append(ir_img)
    train_vi_Y_data.append(vi_y)
    train_vi_3_data.append(vi_3c)

# 加载评估集
eval_ir_data   = []
eval_vi_Y_data = []
eval_vi_3_data = []

for ir_p, vi_p in zip(eval_ir_paths, eval_vi_paths):
    ir_img = load_image_auto_gray(ir_p)
    vi_rgb = load_image_auto_rgb(vi_p)
    ir_img = to_float01(ir_img)
    vi_rgb = to_float01(vi_rgb)
    ycbcr  = rgb_ycbcr_np_3(vi_rgb)
    vi_y   = ycbcr[:, :, 0:1]
    vi_3c  = np.repeat(vi_y, 3, axis=2)

    eval_ir_data.append(ir_img)
    eval_vi_Y_data.append(vi_y)
    eval_vi_3_data.append(vi_3c)

# =============== Training Config ===============
epoch            = 180
learning_rate    = 1e-4
train_phase      = 'decomposition'
numBatch         = max(1, len(train_ir_data) // int(batch_size))
checkpoint_dir   = './checkpoint/dscattfusenet/'  # 新目录名，避免覆盖旧权重
if not os.path.isdir(checkpoint_dir):
    os.makedirs(checkpoint_dir)

# 断点续训
# === 自动判断：若目录中存在可用 ckpt 就续训，否则从零开始 ===
ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
ckpt_ok = (
    ckpt is not None
    and ckpt.model_checkpoint_path is not None
    and os.path.exists(ckpt.model_checkpoint_path + '.index')  # TF1.x 标准索引文件
)

if ckpt_ok:
    print("[*] Found existing checkpoint. Resuming from:", ckpt.model_checkpoint_path)
    saver_DSCAttFuseNet.restore(sess, ckpt.model_checkpoint_path)
else:
    print("[*] No checkpoint found. Training from scratch.")


# -------- 保存频率与最优记录 --------
SAVE_EVERY = 100                # 每 100 个 epoch 保存一次普通 ckpt
best_val   = np.inf             # 当前最优验证损失
best_path  = os.path.join(checkpoint_dir, 'model_best.ckpt')
best_info  = os.path.join(checkpoint_dir, 'best.txt')

# 如果已有 best.txt，读一下，便于继续训练时延续“最优”
if os.path.exists(best_info):
    try:
        with open(best_info, 'r', encoding='utf-8') as f:
            line = f.readline().strip()
            parts = line.split(',')
            # 形如：epoch=80, val=0.123456
            if len(parts) == 2 and parts[0].startswith('epoch=') and parts[1].startswith('val='):
                best_val = float(parts[1].split('=')[1])
                print(f"[*] Found previous best: val={best_val:.6f}")
    except Exception:
        pass

def compute_val_loss(sess, patch_h, patch_w, max_samples=20):
    """
    在验证集上计算平均 loss_total（中心裁剪，batch_size=1）
    """
    n = min(len(eval_ir_data), len(eval_vi_Y_data), max_samples)
    if n == 0:
        return np.inf
    losses_val = []
    for i in range(n):
        ir_img = eval_ir_data[i]
        vi_y   = eval_vi_Y_data[i]
        vi3    = eval_vi_3_data[i]

        ir_p, vi_y_p, vi3_p = center_crop_pair(ir_img, vi_y, vi3, patch_size_y, patch_size_x)

        feed = {
            ir:   ir_p[None, ..., None].astype(np.float32),
            vi:   vi_y_p[None, ...].astype(np.float32),
            vi_3: vi3_p[None, ...].astype(np.float32),
            vi_hist:    vi_y_p[None, ...].astype(np.float32),
            vi_hist_3:  vi3_p[None, ...].astype(np.float32),
            lr: 0.0,  # 评估不更新
        }
        lv = sess.run(loss_total, feed_dict=feed)
        losses_val.append(lv)
    return float(np.mean(losses_val))

# =============== Training Loop ===============
patch_h, patch_w = patch_size_y, patch_size_x
print("[*] Start training for phase %s" % train_phase)

for ep in range(epoch):
    # 一个 epoch 内随机打乱索引
    idxs = list(range(len(train_ir_data)))
    random.shuffle(idxs)

    for b in range(numBatch):
        b_ir   = np.zeros((batch_size, patch_h, patch_w, 1), dtype=np.float32)
        b_vi   = np.zeros((batch_size, patch_h, patch_w, 1), dtype=np.float32)
        b_vi3  = np.zeros((batch_size, patch_h, patch_w, 3), dtype=np.float32)
        b_vih  = np.zeros((batch_size, patch_h, patch_w, 1), dtype=np.float32)  # 如果不做直方图均衡，直接等于 b_vi
        b_vih3 = np.zeros((batch_size, patch_h, patch_w, 3), dtype=np.float32)

        for i in range(batch_size):
            if b * batch_size + i >= len(idxs):
                rnd = np.random.randint(0, len(idxs))
                sel = idxs[rnd]
            else:
                sel = idxs[b * batch_size + i]

            ir_img = train_ir_data[sel]      # [H,W]
            vi_y   = train_vi_Y_data[sel]    # [H,W,1]
            vi3    = train_vi_3_data[sel]    # [H,W,3]

            ir_p, vi_y_p, vi3_p = random_crop_pair(ir_img, vi_y, vi3, patch_h, patch_w)

            # 直方图均衡（可选）：这里为了稳定，默认不用，直接赋值
            vih  = vi_y_p
            vih3 = vi3_p

            b_ir[i, ..., 0] = ir_p
            b_vi[i, ...]    = vi_y_p
            b_vi3[i, ...]   = vi3_p
            b_vih[i, ...]   = vih
            b_vih3[i, ...]  = vih3

        feed = {
            ir: b_ir,
            vi: b_vi,
            vi_3: b_vi3,
            vi_hist: b_vih,
            vi_hist_3: b_vih3,
            lr: learning_rate
        }
        _, loss_val, loss_base, loss_g = sess.run(
            [train_op_DSCAttFuseNet, loss_total, loss_Dsca, L_edge],
            feed_dict=feed
        )

    print("Epoch %d/%d | total=%.6f  base=%.6f  L_grad=%.6f" %
          (ep + 1, epoch, loss_val, loss_base, loss_g))

    # ---- 验证并挑 best ----
    val_loss = compute_val_loss(sess, patch_h, patch_w, max_samples=20)
    print("          | val_total=%.6f" % val_loss)

    if val_loss < best_val:
        best_val = val_loss
        saver_DSCAttFuseNet.save(sess, best_path)
        with open(best_info, 'w', encoding='utf-8') as f:
            f.write(f'epoch={ep+1}, val={best_val:.6f}\n')
        print(f"[*] New BEST at epoch {ep+1}: {best_val:.6f} -> saved to {best_path}")

    # ---- 每 100 个 epoch 保存一次普通 ckpt ----
    if (ep + 1) % SAVE_EVERY == 0:
        saver_DSCAttFuseNet.save(sess, os.path.join(checkpoint_dir, 'model.ckpt'), global_step=ep + 1)
        print(f"[*] Saved checkpoint at epoch {ep+1}")

# 训练结束后保存最后一版
last_path = os.path.join(checkpoint_dir, 'model_last.ckpt')
saver_DSCAttFuseNet.save(sess, last_path)
print(f"[*] Training finished. Last model saved to: {last_path}")
print(f"[*] Best model   saved to: {best_path} (val={best_val:.6f})")
