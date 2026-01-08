import argparse
import tensorflow as tf
import os
from model import *  # 从 model.py 导入模型
from losses import *  # 从 losses.py 导入损失函数
from train_dscattfusenet import *  # 如果你希望直接使用 train 脚本中的训练代码

# 解析命令行参数
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--use_dsc', type=bool, default=True, help="是否使用深度可分离卷积")
    ap.add_argument('--use_eca', type=bool, default=True, help="是否使用通道注意力（ECA）")
    ap.add_argument('--use_scam', type=bool, default=True, help="是否使用SCAM")
    ap.add_argument('--use_guided_fusion', type=bool, default=True, help="是否使用亮度引导融合")
    ap.add_argument('--ir_dir', type=str, default='./TestDatasets/LLVIP/ir', help="红外图像目录")
    ap.add_argument('--vi_dir', type=str, default='./TestDatasets/LLVIP/vi', help="可见光图像目录")
    ap.add_argument('--ckpt', type=str, default='./checkpoint/dscattfusenet/model_best.ckpt', help="模型检查点路径")
    ap.add_argument('--out', type=str, default='./Results', help="输出结果目录")
    return ap.parse_args()

# 训练代码，基于命令行参数进行模块控制
def encoder(img, use_dsc=True, use_eca=True):
    with tf.variable_scope('encoder'):
        with tf.variable_scope('layer1'):
            weights = tf.get_variable("w1", [3, 3, 2, 64], initializer=tf.truncated_normal_initializer(stddev=1e-3))
            bias = tf.get_variable("b1", [64], initializer=tf.constant_initializer(0.0))
            conv1 = tf.contrib.layers.batch_norm(
                tf.nn.conv2d(img, weights, strides=[1, 1, 1, 1], padding='SAME') + bias,
                decay=0.9, updates_collections=None, epsilon=1e-5, scale=True)
            conv1 = lrelu(conv1)

        # 根据 use_dsc 参数决定是否使用深度可分离卷积
        if use_dsc:
            conv2 = dw_sep_conv(conv1, 128, name='layer2_dsc')
            conv3 = dw_sep_conv(conv2, 256, name='layer3_dsc')
            conv4 = dw_sep_conv(conv3, 256, name='layer4_dsc')
        else:
            conv2 = regular_conv(conv1, 128, name='layer2')
            conv3 = regular_conv(conv2, 256, name='layer3')
            conv4 = regular_conv(conv3, 256, name='layer4')

        # 根据 use_eca 参数决定是否使用通道注意力
        if use_eca:
            feature = eca(conv4, k=3, name='eca')
        else:
            feature = conv4
    return feature

def decomposition(vi, ir, use_dsc=True, use_eca=True, use_scam=True, use_guided_fusion=True):
    with tf.variable_scope('DSCAttFuseNet', reuse=tf.AUTO_REUSE):
        img = tf.concat([vi, ir], axis=-1)
        feature = encoder(img, use_dsc=use_dsc, use_eca=use_eca)

        if use_scam:
            vector_ir = SCAM_IR(feature)
            feature_ir = get_sf_ir(vector_ir, feature)
            ir_r = decoder_ir(feature_ir)

            vector_vi_e = SCAM_VI(feature)
            feature_vi_e = get_sf_vi_e(vector_vi_e, feature)

            vector_l = SCAM_L(feature)
            feature_l = get_sf_l(vector_l, feature)
            vi_e_r, l_r = decoder_vi_l(feature_vi_e, feature_l)
        else:
            # 如果不使用 SCAM，则仅返回融合亮度通道
            ir_r = decoder_ir(feature)
            vi_e_r, l_r = decoder_vi_l(feature, feature)

        if use_guided_fusion:
            yf = tf.maximum(ir_r, vi_e_r * l_r, name='Y_f')  # 使用亮度引导融合
        else:
            yf = tf.maximum(ir_r, vi_e_r, name='Y_f')  # 不使用亮度引导融合

    return ir_r, vi_e_r, l_r, yf

# 运行推理并保存结果
def run_infer(args):
    # 创建输出目录
    subdirs = ['ir_r', 'vi_e_r', 'l_r', 'yf', 'yf_color']
    for sd in subdirs:
        os.makedirs(os.path.join(args.out, sd), exist_ok=True)

    ir_list = get_image_paths(args.ir_dir)
    vi_list = get_image_paths(args.vi_dir)
    n = min(len(ir_list), len(vi_list))

    tf.reset_default_graph()
    vi_ph, ir_ph, fetches = build_gr aph()

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        restore_dscattfusenet(sess, args.ckpt)

        for i in range(n):
            ir_path = ir_list[i]
            vi_path = vi_list[i]
            base = os.path.splitext(os.path.basename(ir_path))[0]

            ir_gray = load_image_auto_gray(ir_path)
            vi_rgb = load_image_auto_rgb(vi_path)
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
            l_r = np.squeeze(outs['l_r'])
            Yf = np.squeeze(outs['Y_f'])

            ycbcr_f = np.concatenate([Yf[..., None], cb, cr], axis=-1)
            rgb_f = ycbcr_to_rgb_np(ycbcr_f)

            # 保存结果
            ext = os.path.splitext(ir_path)[1]
            save_gray(os.path.join(args.out, 'ir_r', f'{base}{ext}'), ir_r)
            save_gray(os.path.join(args.out, 'vi_e_r', f'{base}{ext}'), vi_e)
            save_gray(os.path.join(args.out, 'l_r', f'{base}{ext}'), l_r)
            save_gray(os.path.join(args.out, 'yf', f'{base}{ext}'), Yf)
            save_rgb(os.path.join(args.out, 'yf_color', f'{base}{ext}'), rgb_f)

            print(f'[*] Done: {i+1}/{n}')

# 运行不同的实验
if __name__ == '__main__':
    args = parse_args()
    run_infer(args)
