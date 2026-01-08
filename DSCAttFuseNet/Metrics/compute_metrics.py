import numpy as np
import cv2
import math
from math import pi
import numpy as np
import cv2
from scipy.ndimage import gaussian_filter

from scipy.ndimage import uniform_filter, gaussian_filter
from scipy.signal import convolve2d
from scipy.fft import fft2, ifft2, fftshift, ifftshift
def compute_vifp_mscale(ref, dist):
    """
    Multi-scale Visual Information Fidelity (VIF) metric between reference and distorted image.
    Ported from MATLAB by Sheikh & Bovik.
    """
    sigma_nsq = 2.0
    eps = 1e-10
    num = 0.0
    den = 0.0

    for scale in range(4):
        N = 2 ** (4 - scale) + 1
        sigma = N / 5.0
        win = cv2.getGaussianKernel(N, sigma)
        win = win @ win.T

        if scale > 0:
            ref = cv2.filter2D(ref, -1, win)[::2, ::2]
            dist = cv2.filter2D(dist, -1, win)[::2, ::2]

        mu1 = cv2.filter2D(ref, -1, win)
        mu2 = cv2.filter2D(dist, -1, win)

        mu1_sq = mu1 * mu1
        mu2_sq = mu2 * mu2
        mu1_mu2 = mu1 * mu2

        sigma1_sq = cv2.filter2D(ref * ref, -1, win) - mu1_sq
        sigma2_sq = cv2.filter2D(dist * dist, -1, win) - mu2_sq
        sigma12 = cv2.filter2D(ref * dist, -1, win) - mu1_mu2

        sigma1_sq[sigma1_sq < 0] = 0
        sigma2_sq[sigma2_sq < 0] = 0

        g = sigma12 / (sigma1_sq + eps)
        sv_sq = sigma2_sq - g * sigma12

        g[sigma1_sq < eps] = 0
        sv_sq[sigma1_sq < eps] = sigma2_sq[sigma1_sq < eps]
        sigma1_sq[sigma1_sq < eps] = 0

        g[sigma2_sq < eps] = 0
        sv_sq[sigma2_sq < eps] = 0

        g[g < 0] = 0
        sv_sq[g < 0] = sigma2_sq[g < 0]
        sv_sq[sv_sq <= eps] = eps

        num += np.sum(np.log10(1 + (g ** 2) * sigma1_sq / (sv_sq + sigma_nsq)))
        den += np.sum(np.log10(1 + sigma1_sq / sigma_nsq))

    vifp = num / den if den != 0 else 0
    return vifp

def ensure_gray(img):
    """将 RGB 图像转灰度"""
    if img.ndim == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    elif img.ndim == 2:
        return img
    else:
        raise ValueError("图像维度错误：{}".format(img.shape))



def compute_ag(img):
    """
    Compute Average Gradient (AG) of a fused image.
    Args:
        img: np.ndarray, shape (H, W) or (H, W, C), grayscale or color image.
    Returns:
        ag_val: float, average gradient value.
    """
    img = img.astype(np.float64)
    if img.ndim == 2:  # Grayscale
        img = np.expand_dims(img, axis=-1)

    H, W, C = img.shape
    gradients = []

    for k in range(C):
        band = img[:, :, k]
        dzdx, dzdy = np.gradient(band)
        s = np.sqrt((dzdx ** 2 + dzdy ** 2) / 2)
        g = np.sum(s) / ((H - 1) * (W - 1))
        gradients.append(g)

    ag_val = np.mean(gradients)
    return ag_val


def compute_entropy(img):
    """
    Compute entropy of a grayscale image.
    """
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hist, _ = np.histogram(img.ravel(), bins=256, range=(0, 256), density=True)
    hist = hist[hist > 0]  # Remove zeros
    entropy = -np.sum(hist * np.log2(hist))
    return entropy


def compute_joint_entropy(img1, img2, grey_level=256):
    """
    Compute joint entropy between two grayscale images.
    """
    if img1.ndim == 3:
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    if img2.ndim == 3:
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    joint_hist = np.zeros((grey_level, grey_level))

    for i in range(img1.shape[0]):
        for j in range(img1.shape[1]):
            a = img1[i, j]
            b = img2[i, j]
            joint_hist[a, b] += 1

    joint_hist /= np.sum(joint_hist)
    joint_hist = joint_hist[joint_hist > 0]
    joint_entropy = -np.sum(joint_hist * np.log2(joint_hist))
    return joint_entropy


def compute_mi(imgA, imgB, imgF, grey_level=256):
    """
    Compute mutual information between source images and fused image.
    Args:
        imgA: Source image A (e.g., IR)
        imgB: Source image B (e.g., VI)
        imgF: Fused image
    Returns:
        MI score
    """
    HA = compute_entropy(imgA)
    HB = compute_entropy(imgB)
    HF = compute_entropy(imgF)
    HFA = compute_joint_entropy(imgF, imgA, grey_level)
    HFB = compute_joint_entropy(imgF, imgB, grey_level)
    MI_FA = HA + HF - HFA
    MI_FB = HB + HF - HFB
    MI = MI_FA + MI_FB
    return MI


def compute_vif(img_ir, img_vi, fused):
    """
    VIF fusion metric: compute VIF(IR, Fused) + VIF(VI, Fused)
    """
    def to_gray(img):
        if img.ndim == 3:
            return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return img

    ref_ir = to_gray(img_ir).astype(np.float64)
    ref_vi = to_gray(img_vi).astype(np.float64)
    fused = to_gray(fused).astype(np.float64)

    # Normalize to 0–255 if necessary
    if fused.max() <= 1.0: fused *= 255
    if ref_ir.max() <= 1.0: ref_ir *= 255
    if ref_vi.max() <= 1.0: ref_vi *= 255

    vif_ir = compute_vifp_mscale(ref_ir, fused)
    vif_vi = compute_vifp_mscale(ref_vi, fused)
    return vif_ir + vif_vi


def compute_qabf(img1, img2, fused):
    """
    Compute the Qabf metric for image fusion
    Reference: C.S. Xydeas and V. Petrovic, "Objective image fusion performance measure"
    """

    def to_gray(img):
        if img.ndim == 3 and img.shape[2] == 3:
            return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return img

    img1 = to_gray(img1).astype(np.float64)
    img2 = to_gray(img2).astype(np.float64)
    fused = to_gray(fused).astype(np.float64)

    # Normalize to 0–255 if not already
    if fused.max() <= 1.0: fused *= 255
    if img1.max() <= 1.0: img1 *= 255
    if img2.max() <= 1.0: img2 *= 255

    # Sobel operator
    h1 = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    h3 = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])

    def gradient_and_orientation(img):
        gx = cv2.filter2D(img, -1, h3)
        gy = cv2.filter2D(img, -1, h1)
        grad = np.sqrt(gx**2 + gy**2)
        angle = np.arctan2(gy, gx)
        angle[np.isnan(angle)] = pi / 2
        return grad, angle

    gA, aA = gradient_and_orientation(img1)
    gB, aB = gradient_and_orientation(img2)
    gF, aF = gradient_and_orientation(fused)

    # Qabf parameters
    Tg = 0.9994
    kg = -15
    Dg = 0.5
    Ta = 0.9879
    ka = -22
    Da = 0.8

    def compute_Q(gX, aX):
        GXF = np.minimum(gX, gF) / np.maximum(gX, gF + 1e-12)
        AXF = 1 - np.abs(aX - aF) / (pi / 2)

        Qg = Tg / (1 + np.exp(kg * (GXF - Dg)))
        Qa = Ta / (1 + np.exp(ka * (AXF - Da)))
        Q = Qg * Qa
        return Q

    QAF = compute_Q(gA, aA)
    QBF = compute_Q(gB, aB)

    numerator = np.sum(QAF * gA + QBF * gB)
    denominator = np.sum(gA + gB)

    qabf_score = numerator / (denominator + 1e-12)
    return qabf_score



def compute_variance(img1, img2, fused):
    """
    计算融合图像的亮度方差指标，反映细节对比度水平
    Reference: Y.-J. Rao, “In-fibre bragg grating sensors,” 1997.
    """

    def to_gray(img):
        if img.ndim == 3 and img.shape[2] == 3:
            return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return img

    fused = to_gray(fused).astype(np.float64)

    # 若为 float 格式图像，确保范围在 0~255
    if fused.max() <= 1.0:
        fused *= 255.0

    mean_val = np.mean(fused)
    variance = np.sqrt(np.mean((fused - mean_val) ** 2))  # 方差（标准差）

    return variance
def compute_fmi(img1, img2, fused):
    """
    Compute Feature Mutual Information (FMI) between fused image and source images.
    Reference: Haghighat et al., "Fast-FMI: Non-reference image fusion quality metric"
    """
    def to_gray(img):
        if img.ndim == 3:
            return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return img

    img1 = to_gray(img1).astype(np.float64)
    img2 = to_gray(img2).astype(np.float64)
    fused = to_gray(fused).astype(np.float64)

    # Normalize to [0,1]
    img1 /= 255.0
    img2 /= 255.0
    fused /= 255.0

    def sobel_feature(img):
        gx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
        grad = np.sqrt(gx ** 2 + gy ** 2)
        return grad

    feat1 = sobel_feature(img1)
    feat2 = sobel_feature(img2)
    feat_fused = sobel_feature(fused)

    def mutual_info(x, y, bins=256):
        x = x.ravel()
        y = y.ravel()
        joint_hist, _, _ = np.histogram2d(x, y, bins=bins)
        joint_prob = joint_hist / np.sum(joint_hist)
        x_marginal = np.sum(joint_prob, axis=1)
        y_marginal = np.sum(joint_prob, axis=0)
        nonzero = joint_prob > 0
        mi = np.sum(joint_prob[nonzero] * np.log2(joint_prob[nonzero] / (
            x_marginal[:, None] @ y_marginal[None, :] + 1e-12
        )[nonzero]))
        return mi

    fmi_val = mutual_info(feat1, feat_fused) + mutual_info(feat2, feat_fused)
    return fmi_val
def compute_sf(img):
    """
    计算图像的空间频率 (Spatial Frequency, SF)

    """
    import numpy as np
    import cv2

    if img.ndim == 3 and img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    img = img.astype(np.float64)

    RF = np.diff(img, axis=0)  # 行频率
    CF = np.diff(img, axis=1)  # 列频率

    RF1 = np.sqrt(np.mean(RF ** 2))
    CF1 = np.sqrt(np.mean(CF ** 2))

    SF = np.sqrt(RF1 ** 2 + CF1 ** 2)
    return SF
def compute_sd(img):
    """
    计算图像的标准差 (Standard Deviation, SD)

    """
    import numpy as np
    import cv2

    if img.ndim == 3 and img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    img = img.astype(np.float64)
    mean_val = np.mean(img)
    sd = np.sqrt(np.mean((img - mean_val) ** 2))

    return sd
def compute_cc(img_ir, img_vi, fused):
    """
    计算图像融合的相关系数 (Correlation Coefficient, CC)"""
    import numpy as np
    import cv2

    def to_gray(img):
        if img.ndim == 3 and img.shape[2] == 3:
            return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return img

    A = to_gray(img_ir).astype(np.float64)
    B = to_gray(img_vi).astype(np.float64)
    F = to_gray(fused).astype(np.float64)

    A_mean = np.mean(A)
    B_mean = np.mean(B)
    F_mean = np.mean(F)

    # 避免除以0
    eps = 1e-12

    rAF_num = np.sum((A - A_mean) * (F - F_mean))
    rAF_den = np.sqrt(np.sum((A - A_mean)**2) * np.sum((F - F_mean)**2) + eps)
    rAF = rAF_num / rAF_den

    rBF_num = np.sum((B - B_mean) * (F - F_mean))
    rBF_den = np.sqrt(np.sum((B - B_mean)**2) * np.sum((F - F_mean)**2) + eps)
    rBF = rBF_num / rBF_den

    CC = (rAF + rBF) / 2.0
    return CC

import numpy as np
import cv2

def compute_scd(img_ir, img_vi, fused):
    """
    Sum of the Correlations of Differences (SCD) 指标

    """

    def to_gray(img):
        if img.ndim == 3 and img.shape[2] == 3:
            return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return img

    img1 = to_gray(img_ir).astype(np.float64)
    img2 = to_gray(img_vi).astype(np.float64)
    fus = to_gray(fused).astype(np.float64)

    diff1 = fus - img2
    diff2 = fus - img1

    def corr2(a, b):
        a_mean = np.mean(a)
        b_mean = np.mean(b)
        numerator = np.sum((a - a_mean) * (b - b_mean))
        denominator = np.sqrt(np.sum((a - a_mean) ** 2) * np.sum((b - b_mean) ** 2))
        return numerator / (denominator + 1e-12)

    r = corr2(diff1, img1) + corr2(diff2, img2)
    return r


import cv2
import numpy as np

def compute_mef_ssim(img_ir, img_vi, fused, K=0.03, window_size=11, sigma=1.5):
    """
    MEF-SSIM: Multi-exposure Fusion Structural Similarity Index
    Reference: https://ieeexplore.ieee.org/document/7410820

    Args:
        img_ir: infrared image
        img_vi: visible image
        fused: fused image
        K: stability constant (default 0.03)
        window_size: Gaussian window size (default 11)
        sigma: Gaussian sigma (default 1.5)

    Returns:
        Q: MEF-SSIM score
    """

    def to_gray(img):
        if img.ndim == 3:
            return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float64)
        return img.astype(np.float64)

    # Resize if necessary to match
    h = min(img_ir.shape[0], img_vi.shape[0], fused.shape[0])
    w = min(img_ir.shape[1], img_vi.shape[1], fused.shape[1])
    img_ir = cv2.resize(img_ir, (w, h), interpolation=cv2.INTER_AREA)
    img_vi = cv2.resize(img_vi, (w, h), interpolation=cv2.INTER_AREA)
    fused = cv2.resize(fused, (w, h), interpolation=cv2.INTER_AREA)

    img1 = to_gray(img_ir)
    img2 = to_gray(img_vi)
    fused = to_gray(fused)

    # Normalize to [0, 255] if needed
    if img1.max() <= 1.0: img1 *= 255
    if img2.max() <= 1.0: img2 *= 255
    if fused.max() <= 1.0: fused *= 255

    imgSeq = np.stack([img1, img2], axis=-1)
    s1, s2, s3 = imgSeq.shape
    wSize = window_size
    bd = wSize // 2
    eps = 1e-12

    # Gaussian window
    window = cv2.getGaussianKernel(wSize, sigma)
    window = window @ window.T
    window /= np.sum(window)
    win_flat = window.flatten()

    # Uniform filter for local stats
    sWindow = np.ones((wSize, wSize)) / (wSize ** 2)
    mu = np.zeros((s1 - 2 * bd, s2 - 2 * bd, s3))
    ed = np.zeros_like(mu)

    for i in range(s3):
        img = imgSeq[..., i]
        mu[..., i] = cv2.filter2D(img, -1, sWindow)[bd:-bd, bd:-bd]
        muSq = mu[..., i] ** 2
        sigmaSq = cv2.filter2D(img * img, -1, sWindow)[bd:-bd, bd:-bd] - muSq
        ed[..., i] = np.sqrt(np.maximum(wSize ** 2 * sigmaSq, 0)) + 0.001

    R = np.zeros((s1 - 2 * bd, s2 - 2 * bd))
    for i in range(bd, s1 - bd):
        for j in range(bd, s2 - bd):
            vecs = imgSeq[i-bd:i+bd+1, j-bd:j+bd+1, :].reshape(-1, s3)
            denominator = sum(np.linalg.norm(vecs[:, k] - mu[i - bd, j - bd, k]) for k in range(s3))
            numerator = np.linalg.norm(np.sum(vecs, axis=1) - np.mean(np.sum(vecs, axis=1)))
            R[i - bd, j - bd] = (numerator + eps) / (denominator + eps)

    R = np.clip(R, eps, 1 - eps)
    p = np.tan(np.pi / 2 * R)
    p = np.clip(p, 0, 10)

    p = np.repeat(p[:, :, np.newaxis], s3, axis=2)
    wMap = (ed / wSize) ** p + eps
    wMap /= np.sum(wMap, axis=2, keepdims=True)

    maxEd = np.max(ed, axis=2)
    fI = fused[bd:-bd, bd:-bd]
    qMap = np.zeros_like(R)
    C = (K * 255) ** 2

    for i in range(bd, s1 - bd):
        for j in range(bd, s2 - bd):
            blocks = imgSeq[i-bd:i+bd+1, j-bd:j+bd+1, :]
            rBlock = sum(wMap[i-bd, j-bd, k] * (blocks[..., k] - mu[i-bd, j-bd, k]) / ed[i-bd, j-bd, k]
                         for k in range(s3))
            if np.linalg.norm(rBlock) > 0:
                rBlock = rBlock / np.linalg.norm(rBlock) * maxEd[i-bd, j-bd]

            fBlock = fI[i-bd:i+bd+1, j-bd:j+bd+1]

            if rBlock.shape != (wSize, wSize) or fBlock.shape != (wSize, wSize):
                continue  # Skip border if shape mismatch

            rVec = rBlock.flatten()
            fVec = fBlock.flatten()

            # Ensure same length
            min_len = min(len(win_flat), len(rVec), len(fVec))
            w = win_flat[:min_len]
            r = rVec[:min_len]
            f = fVec[:min_len]

            mu1 = np.sum(w * r)
            mu2 = np.sum(w * f)
            sigma1Sq = np.sum(w * (r - mu1) ** 2)
            sigma2Sq = np.sum(w * (f - mu2) ** 2)
            sigma12 = np.sum(w * (r - mu1) * (f - mu2))

            qMap[i - bd, j - bd] = (2 * sigma12 + C) / (sigma1Sq + sigma2Sq + C)

    Q = np.mean(qMap)
    return Q


def compute_psnr(img_ir, img_vi, fused):
    """

        PSNR
    """

    def to_gray(img):
        if img.ndim == 3:
            return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return img

    ir = to_gray(img_ir).astype(np.float64)
    vi = to_gray(img_vi).astype(np.float64)
    f = to_gray(fused).astype(np.float64)

    # Normalize to [0, 255] if needed
    if ir.max() <= 1.0: ir *= 255.0
    if vi.max() <= 1.0: vi *= 255.0
    if f.max() <= 1.0: f *= 255.0

    ir /= 255.0
    vi /= 255.0
    f /= 255.0

    mse_af = np.mean((f - ir) ** 2)
    mse_bf = np.mean((f - vi) ** 2)
    mse = 0.5 * (mse_af + mse_bf)

    psnr = 20 * np.log10(1.0 / np.sqrt(mse)) if mse != 0 else float('inf')
    return psnr


def normalize1(data):
    data = data.astype(np.float64)
    min_val = data.min()
    max_val = data.max()
    if max_val == 0 and min_val == 0:
        return data
    norm = (data - min_val) / (max_val - min_val + 1e-12)
    return np.round(norm * 255)

def compute_qcv(img1, img2, fused, window_size=16, alpha=5):
    """
    Python implementation of Qcv fusion metric.
    """
    img1 = normalize1(cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY) if img1.ndim == 3 else img1)
    img2 = normalize1(cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY) if img2.ndim == 3 else img2)
    fused = normalize1(cv2.cvtColor(fused, cv2.COLOR_BGR2GRAY) if fused.ndim == 3 else fused)

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    fused = fused.astype(np.float64)

    # Step 1: Edge maps
    flt1 = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    flt2 = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

    def compute_gradient(img):
        gx = convolve2d(img, flt1, mode='same')
        gy = convolve2d(img, flt2, mode='same')
        return np.sqrt(gx**2 + gy**2)

    grad1 = compute_gradient(img1)
    grad2 = compute_gradient(img2)

    # Step 2: Local energy (ramda)
    def local_energy(img):
        h, w = img.shape
        H, W = h // window_size, w // window_size
        img = img[:H*window_size, :W*window_size]
        img_blocks = img.reshape(H, window_size, W, window_size).transpose(0, 2, 1, 3)
        energy = np.sum(np.power(img_blocks, alpha), axis=(2, 3))
        return energy

    ramda1 = local_energy(grad1)
    ramda2 = local_energy(grad2)

    # Step 3: Perceptual filter
    h, w = fused.shape
    u, v = np.meshgrid(np.fft.fftfreq(w), np.fft.fftfreq(h))
    r = np.sqrt((u * w / 8)**2 + (v * h / 8)**2)
    r[r == 0] = 1e-6

    theta_m = 2.6 * (0.0192 + 0.144 * r) * np.exp(-(0.144 * r)**1.1)
    ff1 = fft2(img1 - fused)
    ff2 = fft2(img2 - fused)

    Df1 = np.real(ifft2(ifftshift(fftshift(ff1) * theta_m)))
    Df2 = np.real(ifft2(ifftshift(fftshift(ff2) * theta_m)))

    def block_mean2(img):
        h, w = img.shape
        H, W = h // window_size, w // window_size
        img = img[:H*window_size, :W*window_size]
        img_blocks = img.reshape(H, window_size, W, window_size).transpose(0, 2, 1, 3)
        return np.mean(np.square(img_blocks), axis=(2, 3))

    D1 = block_mean2(Df1)
    D2 = block_mean2(Df2)

    # Step 4: Qcv metric
    Qcv_value = np.sum(ramda1 * D1 + ramda2 * D2) / (np.sum(ramda1 + ramda2) + 1e-12)
    return float(Qcv_value)




