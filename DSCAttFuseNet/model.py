
import numpy as np
from PIL import Image
import tensorflow as tf
import scipy.stats as st
from skimage import io,data,color
from functools import reduce
import cv2

############ 常量的预定义 ############
batch_size = 5
patch_size_x = 224
patch_size_y = 224
############ Encoder ############
# 输入img为concat红外可见光图像的结果，通道数为2
# 输出为256个feature—map
def encoder(img):
    with tf.variable_scope('encoder'):
        with tf.variable_scope('layer1'):
            weights = tf.get_variable("w1", [3, 3, 2, 64], initializer=tf.truncated_normal_initializer(stddev=1e-3))
            bias = tf.get_variable("b1", [64], initializer=tf.constant_initializer(0.0))
            conv1 = tf.contrib.layers.batch_norm(
                tf.nn.conv2d(img, weights, strides=[1, 1, 1, 1], padding='SAME') + bias,
                decay=0.9, updates_collections=None, epsilon=1e-5, scale=True)
            conv1 = lrelu(conv1)

        # === 用 DSC 替换 layer2/3/4 ===
        conv2 = dw_sep_conv(conv1, 128, name='layer2_dsc')
        conv3 = dw_sep_conv(conv2, 256, name='layer3_dsc')
        conv4 = dw_sep_conv(conv3, 256, name='layer4_dsc')

        # === 轻量通道注意力（全局级） ===
        feature = eca(conv4, k=3, name='eca')

    return feature



############ Decoder ############
def decoder_ir(feature_ir):
    with tf.variable_scope('decoder_ir'):
        with tf.variable_scope('layer1'):
            weights = tf.get_variable("w1", [3, 3, 256, 128], initializer=tf.truncated_normal_initializer(stddev=1e-3))
            bias = tf.get_variable("b1", [128], initializer=tf.constant_initializer(0.0))
            conv1 = tf.contrib.layers.batch_norm(
                tf.nn.conv2d(feature_ir, weights, strides=[1, 1, 1, 1], padding='SAME') + bias, decay=0.9,
                updates_collections=None, epsilon=1e-5, scale=True)
            conv1 = lrelu(conv1)
        with tf.variable_scope('layer2'):
            weights = tf.get_variable("w2", [3, 3, 128, 64], initializer=tf.truncated_normal_initializer(stddev=1e-3))
            bias = tf.get_variable("b2", [64], initializer=tf.constant_initializer(0.0))
            conv2 = tf.contrib.layers.batch_norm(
                tf.nn.conv2d(conv1, weights, strides=[1, 1, 1, 1], padding='SAME') + bias, decay=0.9,
                updates_collections=None, epsilon=1e-5, scale=True)
            conv2 = lrelu(conv2)
        with tf.variable_scope('layer3'):
            weights = tf.get_variable("w3", [3, 3, 64, 32], initializer=tf.truncated_normal_initializer(stddev=1e-3))
            bias = tf.get_variable("b3", [32], initializer=tf.constant_initializer(0.0))
            conv3 = tf.contrib.layers.batch_norm(
                tf.nn.conv2d(conv2, weights, strides=[1, 1, 1, 1], padding='SAME') + bias, decay=0.9,
                updates_collections=None, epsilon=1e-5, scale=True)
            conv3 = lrelu(conv3)
        with tf.variable_scope('layer4'):
            weights = tf.get_variable("w4", [3, 3, 32, 1], initializer=tf.truncated_normal_initializer(stddev=1e-3))
            bias = tf.get_variable("b4", [1], initializer=tf.constant_initializer(0.0))
            conv4 = tf.contrib.layers.batch_norm(
                tf.nn.conv2d(conv3, weights, strides=[1, 1, 1, 1], padding='SAME') + bias, decay=0.9,
                updates_collections=None, epsilon=1e-5, scale=True)
            ir_r = tf.sigmoid(conv4)
        return ir_r



def decoder_vi_l(feature_vi_e, feature_l):
    with tf.variable_scope('decoder_vi_l'):
        with tf.variable_scope('layer1'):
            weights = tf.get_variable("w1", [3, 3, 256, 128],
                                      initializer=tf.truncated_normal_initializer(stddev=1e-3))
            bias = tf.get_variable("b1", [128], initializer=tf.constant_initializer(0.0))
            conv1 = tf.contrib.layers.batch_norm(
                tf.nn.conv2d(feature_vi_e, weights, strides=[1, 1, 1, 1], padding='SAME') + bias, decay=0.9,
                updates_collections=None, epsilon=1e-5, scale=True)
            conv1 = lrelu(conv1)
        with tf.variable_scope('layer2'):
            weights = tf.get_variable("w2", [3, 3, 128, 64],
                                      initializer=tf.truncated_normal_initializer(stddev=1e-3))
            bias = tf.get_variable("b2", [64], initializer=tf.constant_initializer(0.0))
            conv2 = tf.contrib.layers.batch_norm(
                tf.nn.conv2d(conv1, weights, strides=[1, 1, 1, 1], padding='SAME') + bias, decay=0.9,
                updates_collections=None, epsilon=1e-5, scale=True)
            conv2 = lrelu(conv2)
        with tf.variable_scope('layer3'):
            weights = tf.get_variable("w3", [3, 3, 64, 32],
                                      initializer=tf.truncated_normal_initializer(stddev=1e-3))
            bias = tf.get_variable("b3", [32], initializer=tf.constant_initializer(0.0))
            conv3 = tf.contrib.layers.batch_norm(
                tf.nn.conv2d(conv2, weights, strides=[1, 1, 1, 1], padding='SAME') + bias, decay=0.9,
                updates_collections=None, epsilon=1e-5, scale=True)
            conv3 = lrelu(conv3)
        with tf.variable_scope('layer4'):
            weights = tf.get_variable("w4", [3, 3, 32, 1], initializer=tf.truncated_normal_initializer(stddev=1e-3))
            bias = tf.get_variable("b4", [1], initializer=tf.constant_initializer(0.0))
            conv4 = tf.contrib.layers.batch_norm(
                tf.nn.conv2d(conv3, weights, strides=[1, 1, 1, 1], padding='SAME') + bias, decay=0.9,
                updates_collections=None, epsilon=1e-5, scale=True)
            vi_e_r = tf.sigmoid(conv4)
    with tf.variable_scope('decoder_l'):
        with tf.variable_scope('layer1'):
            weights = tf.get_variable("w1", [3, 3, 256, 128], initializer=tf.truncated_normal_initializer(stddev=1e-3))
            bias = tf.get_variable("b1", [128], initializer=tf.constant_initializer(0.0))
            l_conv1 = tf.contrib.layers.batch_norm(
                tf.nn.conv2d(feature_l, weights, strides=[1, 1, 1, 1], padding='SAME') + bias, decay=0.9,
                updates_collections=None, epsilon=1e-5, scale=True)
            l_conv1 = lrelu(l_conv1)
            l_conv1 = tf.concat([l_conv1, conv1], axis=3)
        with tf.variable_scope('layer2'):
            weights = tf.get_variable("w2", [3, 3, 256, 64], initializer=tf.truncated_normal_initializer(stddev=1e-3))
            bias = tf.get_variable("b2", [64], initializer=tf.constant_initializer(0.0))
            l_conv2 = tf.contrib.layers.batch_norm(
                tf.nn.conv2d(l_conv1, weights, strides=[1, 1, 1, 1], padding='SAME') + bias, decay=0.9,
                updates_collections=None, epsilon=1e-5, scale=True)
            l_conv2 = lrelu(l_conv2)
        with tf.variable_scope('layer3'):
            weights = tf.get_variable("w3", [3, 3, 64, 32], initializer=tf.truncated_normal_initializer(stddev=1e-3))
            bias = tf.get_variable("b3", [32], initializer=tf.constant_initializer(0.0))
            l_conv3 = tf.contrib.layers.batch_norm(
                tf.nn.conv2d(l_conv2, weights, strides=[1, 1, 1, 1], padding='SAME') + bias, decay=0.9,
                updates_collections=None, epsilon=1e-5, scale=True)
            l_conv3 = lrelu(l_conv3)
            l_conv3 = tf.concat([l_conv3, conv3],axis=3)
        with tf.variable_scope('layer4'):
            weights = tf.get_variable("w4", [3, 3, 64, 1], initializer=tf.truncated_normal_initializer(stddev=1e-3))
            bias = tf.get_variable("b4", [1], initializer=tf.constant_initializer(0.0))
            l_conv4 = tf.contrib.layers.batch_norm(
                tf.nn.conv2d(l_conv3, weights, strides=[1, 1, 1, 1], padding='SAME') + bias, decay=0.9,
                updates_collections=None, epsilon=1e-5, scale=True)
            l_r = tf.sigmoid(l_conv4)
        return vi_e_r, l_r


############  #############
# BCA (Branch-specific Channel Attention) — 强调每个分支单独的注意力。
def SCAM_IR(input_feature):
    with tf.variable_scope('SCAM_IR'):
        with tf.variable_scope('layer'):
            weights = tf.get_variable("w1", [3, 3, 256, 32], initializer=tf.truncated_normal_initializer(stddev=1e-3))
            bias = tf.get_variable("b1", [32], initializer=tf.constant_initializer(0.0))
            conv1 = tf.contrib.layers.batch_norm(
                tf.nn.conv2d(input_feature, weights, strides=[1, 1, 1, 1], padding='SAME') + bias,
                decay=0.9, updates_collections=None, epsilon=1e-5, scale=True)
            conv1 = lrelu(conv1)
            weights = tf.get_variable("w2", [3, 3, 32, 256], initializer=tf.truncated_normal_initializer(stddev=1e-3))
            bias = tf.get_variable("b2", [256], initializer=tf.constant_initializer(0.0))
            conv1 = tf.contrib.layers.batch_norm(
                tf.nn.conv2d(conv1, weights, strides=[1, 1, 1, 1], padding='SAME') + bias,
                decay=0.9, updates_collections=None, epsilon=1e-5, scale=True)
            vector_ir = tf.reduce_mean(conv1, [1, 2], name='global_pool', keep_dims=True)
            vector_ir = tf.nn.softmax(vector_ir)
    return vector_ir
# MSCA (Modality-specific Channel Attention) — 强调模态差异化建模。
def SCAM_VI(input_feature):
    with tf.variable_scope('SCAM_VI'):
        with tf.variable_scope('layer'):
            weights = tf.get_variable("w1", [3, 3, 256, 32], initializer=tf.truncated_normal_initializer(stddev=1e-3))
            bias = tf.get_variable("b1", [32], initializer=tf.constant_initializer(0.0))
            conv1 = tf.contrib.layers.batch_norm(
                tf.nn.conv2d(input_feature, weights, strides=[1, 1, 1, 1], padding='SAME') + bias,
                decay=0.9, updates_collections=None, epsilon=1e-5, scale=True)
            conv1 = lrelu(conv1)
            weights = tf.get_variable("w2", [3, 3, 32, 256], initializer=tf.truncated_normal_initializer(stddev=1e-3))
            bias = tf.get_variable("b2", [256], initializer=tf.constant_initializer(0.0))
            conv1 = tf.contrib.layers.batch_norm(
                tf.nn.conv2d(conv1, weights, strides=[1, 1, 1, 1], padding='SAME') + bias,
                decay=0.9, updates_collections=None, epsilon=1e-5, scale=True)
            vector_vi_e = tf.reduce_mean(conv1, [1, 2], name='global_pool', keep_dims=True)
            vector_vi_e = tf.nn.softmax(vector_vi_e)
    return vector_vi_e
#TCA (Targeted Channel Attention) — 强调对目标特征的选择性增强。
def SCAM_L(input_feature):
    with tf.variable_scope('SCAM_L'):
        with tf.variable_scope('layer'):
            weights = tf.get_variable("w1", [3, 3, 256, 32], initializer=tf.truncated_normal_initializer(stddev=1e-3))
            bias = tf.get_variable("b1", [32], initializer=tf.constant_initializer(0.0))
            conv1 = tf.contrib.layers.batch_norm(
                tf.nn.conv2d(input_feature, weights, strides=[1, 1, 1, 1], padding='SAME') + bias,
                decay=0.9, updates_collections=None, epsilon=1e-5, scale=True)
            conv1 = lrelu(conv1)
            weights = tf.get_variable("w2", [3, 3, 32, 256], initializer=tf.truncated_normal_initializer(stddev=1e-3))
            bias = tf.get_variable("b2", [256], initializer=tf.constant_initializer(0.0))
            conv1 = tf.contrib.layers.batch_norm(
                tf.nn.conv2d(conv1, weights, strides=[1, 1, 1, 1], padding='SAME') + bias,
                decay=0.9, updates_collections=None, epsilon=1e-5, scale=True)
            vector_l = tf.reduce_mean(conv1, [1, 2], name='global_pool', keep_dims=True)
            vector_l = tf.nn.softmax(vector_l)
    return vector_l



############ Special Feature ############
def get_sf_ir(vector_ir, feature):
    with tf.variable_scope('special_feature_ir'):
        # new_vector_ir = tf.broadcast_to(vector_ir, feature.shape)
        feature_ir = tf.multiply(vector_ir, feature)
    return feature_ir

def get_sf_l(vector_l, feature):
    with tf.variable_scope('special_feature_l'):
        # new_vector_l = tf.broadcast_to(vector_l, feature.shape)
        feature_l = tf.multiply(vector_l, feature)
    return feature_l

def get_sf_vi_e(vector_vi_e, feature):
    with tf.variable_scope('special_feature_vi_e'):
        # new_vector_vi_e = tf.broadcast_to(vector_vi_e, feature.shape)
        feature_vi_e = tf.multiply(vector_vi_e, feature)
    return feature_vi_e


############ All_model ############
def decomposition(vi,ir):
    with tf.variable_scope('DSCAttFuseNet', reuse=tf.AUTO_REUSE):
        # 两个图像都得要是通道为1的
        img = tf.concat([vi,ir],axis=-1)
        feature = encoder(img)
        vector_ir = SCAM_IR(feature)
        feature_ir = get_sf_ir(vector_ir, feature)
        ir_r = decoder_ir(feature_ir)

        vector_vi_e = SCAM_VI(feature)
        feature_vi_e = get_sf_vi_e(vector_vi_e, feature)
        vector_l = SCAM_L(feature)
        feature_l = get_sf_l(vector_l, feature)
        [vi_e_r, l_r] = decoder_vi_l(feature_vi_e, feature_l)

        # vector_l = CAM_L(feature)
        # feature_l = get_sf_l(vector_l, feature)
        # l_r = decoder_l(feature_l)
    return ir_r, vi_e_r, l_r


############ Tool ############
def lrelu(x, leak=0.2):
    return tf.maximum(x, leak * x)


def gradient(input_tensor, direction):
    smooth_kernel_x = tf.reshape(tf.constant([[0, 0], [-1, 1]], tf.float32), [2, 2, 1, 1])
    smooth_kernel_y = tf.transpose(smooth_kernel_x, [1, 0, 2, 3])
    if direction == "x":
        kernel = smooth_kernel_x
    elif direction == "y":
        kernel = smooth_kernel_y
    gradient_orig = tf.abs(tf.nn.conv2d(input_tensor, kernel, strides=[1, 1, 1, 1], padding='SAME'))
    grad_min = tf.reduce_min(gradient_orig)
    grad_max = tf.reduce_max(gradient_orig)
    grad_norm = tf.div((gradient_orig - grad_min), (grad_max - grad_min + 0.0001))
    return grad_norm

def laplacian(input_tensor):
    k = tf.reshape(tf.constant([[0.,1.,0.],[1.,-4.,1.],[0.,1.,0.]], tf.float32), [3,3,1,1])
    # 如果要支持多通道输入，可以 broadcast 到 in_ch
    in_ch = input_tensor.get_shape().as_list()[-1]
    k = tf.tile(k, [1,1,in_ch,1])  # [3,3,in_ch,1]
    g = tf.abs(tf.nn.conv2d(input_tensor, k, strides=[1,1,1,1], padding='SAME'))
    g_min = tf.reduce_min(g); g_max = tf.reduce_max(g)
    return tf.div(g - g_min, g_max - g_min + 1e-4)



def load_images(file):
    im = Image.open(file)
    img = np.array(im, dtype="float16") / 255.0
    # img_max = np.max(img)
    # img_min = np.min(img)
    # img_norm = np.float32((img - img_min) / np.maximum((img_max - img_min), 0.001))
    img_norm = np.float32(img)
    return img_norm

def blur_downsample(x, stride=2):
    if stride == 1: return x
    k = tf.reshape(tf.constant([[1,4,6,4,1],
                                [4,16,24,16,4],
                                [6,24,36,24,6],
                                [4,16,24,16,4],
                                [1,4,6,4,1]], tf.float32)/256.0, [5,5,int(x.get_shape()[-1]),1])
    x = tf.nn.depthwise_conv2d(x, k, strides=[1,1,1,1], padding='SAME')
    return tf.nn.avg_pool(x, [1,stride,stride,1], [1,stride,stride,1], 'SAME')

def hist(input):
    input_int = np.uint8((input*255.0))
    input_hist = cv2.equalizeHist(input_int)
    input_hist = (input_hist/255.0).astype(np.float32)
    return input_hist


def save_images(filepath, result_1, result_2 = None, result_3 = None):
    result_1 = np.squeeze(result_1)
    result_2 = np.squeeze(result_2)
    result_3 = np.squeeze(result_3)

    if not result_2.any():
        cat_image = result_1
    else:
        cat_image = np.concatenate([result_1, result_2], axis=1)
    if not result_3.any():
        cat_image = cat_image
    else:
        cat_image = np.concatenate([cat_image, result_3], axis=1)

    im = Image.fromarray(np.clip(cat_image * 255.0, 0, 255.0).astype('uint8'))
    im.save(filepath, 'png')


def rgb_ycbcr(img_rgb):
    R = tf.expand_dims(img_rgb[:, :, :, 0], axis=-1)
    G = tf.expand_dims(img_rgb[:, :, :, 1], axis=-1)
    B = tf.expand_dims(img_rgb[:, :, :, 2], axis=-1)
    Y  = 0.299 * R + 0.587 * G + 0.114 * B
    Cb = -0.1687 * R - 0.3313 * G + 0.5 * B + 128/255
    Cr =  0.5   * R - 0.4187 * G - 0.0813 * B + 128/255
    return tf.concat([Y, Cb, Cr], axis=-1)



def rgb_ycbcr_np(img_rgb):
    R = np.expand_dims(img_rgb[:, :, 0], axis=-1)
    G = np.expand_dims(img_rgb[:, :, 1], axis=-1)
    B = np.expand_dims(img_rgb[:, :, 2], axis=-1)
    Y = 0.299 * R + 0.587 * G + 0.114 * B
    Cb = -0.1687 * R - 0.3313 * G + 0.5 * B + 128/255.0
    Cr = 0.5 * R - 0.4187 * G - 0.0813 * B + 128/255.0
    img_ycbcr = np.concatenate([Y, Cb, Cr], axis=-1)
    return img_ycbcr


# === Lightweight building blocks (TF1.x friendly) ===
def dw_sep_conv(x, out_ch, name, depth_multiplier=1):
    with tf.variable_scope(name):
        in_ch = x.get_shape().as_list()[-1]
        # depthwise: [Kh, Kw, in_ch, depth_multiplier]
        dw_w = tf.get_variable("dw_w", [3, 3, in_ch, depth_multiplier],
                               initializer=tf.truncated_normal_initializer(stddev=1e-3))
        dw = tf.nn.depthwise_conv2d(x, dw_w, strides=[1, 1, 1, 1], padding='SAME')

        # pointwise: [1,1, in_ch*dm, out_ch]
        pw_w = tf.get_variable("pw_w", [1, 1, in_ch * depth_multiplier, out_ch],
                               initializer=tf.truncated_normal_initializer(stddev=1e-3))
        pw = tf.nn.conv2d(dw, pw_w, strides=[1, 1, 1, 1], padding='SAME')
        pw = tf.contrib.layers.batch_norm(pw, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True)
        return lrelu(pw)

def eca(x, k=3, name="eca"):
    # Efficient Channel Attention: GAP + 1D局部卷积，几乎零开销
    with tf.variable_scope(name):
        ch = x.get_shape().as_list()[-1]
        gap = tf.reduce_mean(x, axis=[1, 2], keepdims=True)  # [N,1,1,C]
        v = tf.reshape(gap, [-1, ch, 1])                     # [N,C,1]
        # 用2D卷积模拟1D same-padding卷积
        w = tf.get_variable("conv1d_w", [k, 1, 1],
                            initializer=tf.truncated_normal_initializer(stddev=1e-3))
        v_pad = tf.pad(v, [[0, 0], [k // 2, k // 2], [0, 0]])   # [N,C+k-1,1]
        v_pad = tf.expand_dims(v_pad, axis=1)                   # [N,1,C+k-1,1]
        w2d = tf.reshape(w, [1, k, 1, 1])                       # [1,k,1,1]
        y = tf.nn.conv2d(v_pad, w2d, strides=[1, 1, 1, 1], padding='VALID')  # [N,1,C,1]
        y = tf.reshape(y, [-1, 1, 1, ch])
        s = tf.nn.sigmoid(y)
        return x * s
