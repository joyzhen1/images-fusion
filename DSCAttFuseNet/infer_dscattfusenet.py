# -*- coding: utf-8 -*-
# 只用 DSCAttFuseNet 做推理并输出彩色融合结果（按类别分目录保存）
from __future__ import print_function
import os
import argparse
import cv2
import numpy as np
import tensorflow as tf
from glob import glob
from model import decomposition

# ========== 工具 ==========
def get_image_paths(directory, exts=('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')):
    paths = []
    for e in exts:
        paths.extend(glob(os.path.join(directory, '*' + e)))
    paths.sort()
    return paths

def load_image_auto_gray(path):
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError("无法读取图像: {}".format(path))
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = img.astype(np.float32) / 255.0
    return img  # [H,W]

def load_image_auto_rgb(path):
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError("无法读取图像: {}".format(path))
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # BGR -> RGB
    img = img.astype(np.float32) / 255.0
    return img  # [H,W,3]

def rgb_to_ycbcr_np(rgb):  # rgb [0,1]
    R = rgb[..., 0:1]; G = rgb[..., 1:2]; B = rgb[..., 2:3]
    Y  = 0.299 * R + 0.587 * G + 0.114 * B
    Cb = -0.1687 * R - 0.3313 * G + 0.5   * B + 128.0/255.0
    Cr = 0.5   * R - 0.4187 * G - 0.0813 * B + 128.0/255.0
    return np.concatenate([Y, Cb, Cr], axis=-1)  # [H,W,3]

def ycbcr_to_rgb_np(ycbcr):
    Y  = ycbcr[..., 0:1]
    Cb = ycbcr[..., 1:2]
    Cr = ycbcr[..., 2:3]
    R = Y + 1.402   * (Cr - 128.0/255.0)
    G = Y - 0.34414 * (Cb - 128.0/255.0) - 0.71414 * (Cr - 128.0/255.0)
    B = Y + 1.772   * (Cb - 128.0/255.0)
    rgb = np.concatenate([R, G, B], axis=-1)
    return np.clip(rgb, 0.0, 1.0)

def save_rgb(path, rgb01):
    img = (np.clip(rgb01, 0.0, 1.0) * 255.0).astype(np.uint8)
    bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(path, bgr)

def save_gray(path, gray01):
    img = (np.clip(gray01, 0.0, 1.0) * 255.0).astype(np.uint8)
    cv2.imwrite(path, img)

# ========== 构图（只用分解网络） ==========
def build_graph():
    vi = tf.placeholder(tf.float32, [1, None, None, 1], name='vi')
    ir = tf.placeholder(tf.float32, [1, None, None, 1], name='ir')


    ir_r, vi_e_r, l_r = decomposition(vi, ir)

    yf = tf.maximum(ir_r, vi_e_r * l_r, name='Y_f')  # 融合亮度

    fetches = {'ir_r': ir_r, 'vi_e_r': vi_e_r, 'l_r': l_r, 'Y_f': yf}
    return vi, ir, fetches

# ========== 恢复 checkpoint ==========
def restore_dscattfusenet(sess, ckpt_path):
    if os.path.isdir(ckpt_path):
        ckpt = tf.train.get_checkpoint_state(ckpt_path)
        if not ckpt or not ckpt.model_checkpoint_path:
            raise FileNotFoundError("未找到有效 checkpoint: {}".format(ckpt_path))
        ckpt_path = ckpt.model_checkpoint_path
    if not (os.path.exists(ckpt_path) or os.path.exists(ckpt_path + '.index')):
        raise FileNotFoundError("未找到 checkpoint: {}".format(ckpt_path))

    reader = tf.train.NewCheckpointReader(ckpt_path)
    ckpt_keys = set(reader.get_variable_to_shape_map().keys())
    graph_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='DSCAttFuseNet')
    vars_ok = [v for v in graph_vars if v.op.name in ckpt_keys]

    saver = tf.train.Saver(var_list=vars_ok)
    saver.restore(sess, ckpt_path)
    print('[*] Decom restored: {} vars from {}'.format(len(vars_ok), ckpt_path))

# ========== 主流程 ==========
def run_infer(ir_dir, vi_dir, ckpt, out_dir):
    # 创建 5 个子目录
    subdirs = ['ir_r', 'vi_e_r', 'l_r', 'yf', 'yf_color']
    for sd in subdirs:
        os.makedirs(os.path.join(out_dir, sd), exist_ok=True)

    ir_list = get_image_paths(ir_dir)
    vi_list = get_image_paths(vi_dir)
    n = min(len(ir_list), len(vi_list))

    tf.reset_default_graph()
    vi_ph, ir_ph, fetches = build_graph()

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        restore_dscattfusenet(sess, ckpt)

        for i in range(n):
            ir_path = ir_list[i]
            vi_path = vi_list[i]
            base = os.path.splitext(os.path.basename(ir_path))[0]

            ir_gray = load_image_auto_gray(ir_path)
            vi_rgb  = load_image_auto_rgb(vi_path)
            vi_ycbcr = rgb_to_ycbcr_np(vi_rgb)
            cb = vi_ycbcr[..., 1:2]
            cr = vi_ycbcr[..., 2:3]

            feed = {
                vi_ph: np.expand_dims(vi_ycbcr[...,0:1], axis=0),
                ir_ph: np.expand_dims(ir_gray, axis=(0, -1))
            }

            outs = sess.run(fetches, feed_dict=feed)
            ir_r = np.squeeze(outs['ir_r'])
            vi_e = np.squeeze(outs['vi_e_r'])
            l_r  = np.squeeze(outs['l_r'])
            Yf   = np.squeeze(outs['Y_f'])

            ycbcr_f = np.concatenate([Yf[...,None], cb, cr], axis=-1)
            rgb_f = ycbcr_to_rgb_np(ycbcr_f)

            # 获取红外源图的扩展名，例如 ".jpg"、".png"
            ext = os.path.splitext(ir_path)[1]

            save_gray(os.path.join(out_dir, 'ir_r', f'{base}{ext}'), ir_r)
            save_gray(os.path.join(out_dir, 'vi_e_r', f'{base}{ext}'), vi_e)
            save_gray(os.path.join(out_dir, 'l_r', f'{base}{ext}'), l_r)
            save_gray(os.path.join(out_dir, 'yf', f'{base}{ext}'), Yf)
            save_rgb(os.path.join(out_dir, 'yf_color', f'{base}{ext}'), rgb_f)

            print(f'[*] Done: {i+1}/{n}')

# ========== CLI ==========
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ir_dir', type=str, default='./TestDatasets/LLVIP/ir')
    ap.add_argument('--vi_dir', type=str, default='./TestDatasets/LLVIP/vi')
    ap.add_argument('--ckpt',   type=str, default='./checkpoint/dscattfusenet/model_best.ckpt')
    ap.add_argument('--out',    type=str, default='./results/LLVIP')
    return ap.parse_args()

if __name__ == '__main__':
    args = parse_args()
    run_infer(args.ir_dir, args.vi_dir, args.ckpt, args.out)
