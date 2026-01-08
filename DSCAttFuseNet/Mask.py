# -*- coding: utf-8 -*-
import cv2
import numpy as np
import os
import glob
from collections import defaultdict
from datetime import datetime

# =============== 可调参数（批量） ===============
FOLDER_PATH = r'./Results/ablationdata/our'   # ← 放融合图的文件夹
OUT_DIR     = r'./SeAFusion'               # ← 输出文件夹
EXTS = ['*.png', '*.jpg', '*.jpeg', '*.bmp']

AUTOSAVE_ON_SWITCH = True                   # 切换上一张/下一张时自动保存当前图
FILENAME_SUFFIX = '_marked'                 # 保存文件的后缀

# =============== 可调参数（显示/放大） ===============
MAX_DISPLAY = 1100            # 屏幕显示时最长边上限（仅影响显示，不影响保存）
RECT_THICK = 2                # 主图矩形框粗细
SELECT_HL_THICK = 3           # 选中框高亮粗细
CORNER_THRESH = 10            # 命中角点阈值 (px)

# 右下角放大框布局（固定 2×，放不下再整体缩小）
INSET_MARGIN = 16             # 放大框距画面边缘的外边距
INSET_BORDER_THICK = 2        # 放大框边框粗细
RB_MAX_W_RATIO = 0.35         # 放大框相对画面宽的最大比例
RB_MAX_H_RATIO = 0.35         # 放大框相对画面高的最大比例

# =============== 全局状态（不要改） ===============
start_point = None
end_point = None
is_drawing = False
is_moving = False
is_resizing = False
selected_rectangle = -1
color = (0, 0, 255)           # 当前绘制颜色，红色
move_start_point = None
DISP_SCALE = 1.0              # 显示缩放映射

# 每张图的矩形框： {idx: [(p1, p2, color), ...]}
rectangles_map = defaultdict(list)
# 当前图像索引/路径/像素
img_paths = []
img_idx = 0
img0 = None                   # 当前原图像素（不随显示缩放而变）

# =============== 工具函数 ===============
def ensure_out_dir():
    os.makedirs(OUT_DIR, exist_ok=True)

def list_images(folder):
    files = []
    for pat in EXTS:
        files += glob.glob(os.path.join(folder, pat))
    files = sorted(files)
    return files

def basename_noext(p):
    return os.path.splitext(os.path.basename(p))[0]

def resize_to_fit(img, max_size=1000):
    """显示用等比缩放，最长边不超过 max_size；返回缩放图和比例。"""
    h, w = img.shape[:2]
    scale = max_size / float(max(h, w))
    if scale < 1.0:
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    else:
        scale = 1.0
    return img, scale

def norm_rect(p1, p2):
    x1, y1 = p1; x2, y2 = p2
    return (min(x1,x2), min(y1,y2)), (max(x1,x2), max(y1,y2))

def point_in_rect(x, y, r0, r1):
    (x1,y1),(x2,y2) = norm_rect(r0, r1)
    return x1 <= x <= x2 and y1 <= y <= y2

def near_corners(x, y, r0, r1, thr=CORNER_THRESH):
    (x1,y1),(x2,y2) = norm_rect(r0, r1)
    corners = {'tl':(x1,y1), 'tr':(x2,y1), 'bl':(x1,y2), 'br':(x2,y2)}
    for name,(cx,cy) in corners.items():
        if abs(x-cx) < thr and abs(y-cy) < thr:
            return name
    return None

def clip_rect_to_image(r0, r1, w, h):
    (x1,y1),(x2,y2) = norm_rect(r0, r1)
    x1 = max(0, min(x1, w-1)); x2 = max(0, min(x2, w-1))
    y1 = max(0, min(y1, h-1)); y2 = max(0, min(y2, h-1))
    return (x1,y1), (x2,y2)

def draw_rectangles_on(img, rects, selected_idx=-1, preview=None, preview_color=None):
    """在 img 上画所有矩形；可选预览框。"""
    h, w = img.shape[:2]
    for i, (r0, r1, c) in enumerate(rects):
        r0c, r1c = clip_rect_to_image(r0, r1, w, h)
        if i == selected_idx:
            cv2.rectangle(img, r0c, r1c, (0,255,255), SELECT_HL_THICK, cv2.LINE_AA)
        cv2.rectangle(img, r0c, r1c, c, RECT_THICK, cv2.LINE_AA)
    if preview is not None and preview_color is not None:
        r0p, r1p = clip_rect_to_image(preview[0], preview[1], h=img.shape[0], w=img.shape[1])
        cv2.rectangle(img, r0p, r1p, preview_color, RECT_THICK, cv2.LINE_AA)
    return img

def build_inset_from_roi_fixed2x(roi, color):
    """固定 2× 放大（宽高都乘2），只加细边框，不加留白/文字。"""
    if roi is None or roi.size == 0:
        return None
    h, w = roi.shape[:2]
    if w <= 0 or h <= 0:
        return None
    new_w, new_h = int(w * 2), int(h * 2)         # ——固定 1:2 比例——
    resized = cv2.resize(roi, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
    block = resized.copy()
    cv2.rectangle(block, (0, 0), (new_w-1, new_h-1), color, max(1, INSET_BORDER_THICK), cv2.LINE_AA)
    return block

def paste_inset_right_bottom(canvas, inset):
    """
    把 inset 粘贴到右下角；若放不下，按比例整体缩小（保持2×与ROI的相对比例）。
    """
    if inset is None:
        return canvas
    H, W = canvas.shape[:2]
    ih, iw = inset.shape[:2]

    # 可占的最大宽/高（相对整图）
    max_w = int(RB_MAX_W_RATIO * W)
    max_h = int(RB_MAX_H_RATIO * H)

    scale = min(1.0, max_w / float(iw), max_h / float(ih))
    if scale < 1.0:
        inset = cv2.resize(inset, (max(1, int(iw*scale)), max(1, int(ih*scale))),
                           interpolation=cv2.INTER_NEAREST)
        ih, iw = inset.shape[:2]

    x1 = max(0, W - iw - INSET_MARGIN)
    y1 = max(0, H - ih - INSET_MARGIN)
    canvas[y1:y1+ih, x1:x1+iw] = inset
    return canvas

# =============== 图片切换/保存 ===============
def load_image(index):
    """载入第 index 张图片，返回像素。"""
    global img0
    path = img_paths[index]
    img0 = cv2.imread(path)
    if img0 is None:
        raise RuntimeError(f'无法读取图像：{path}')
    return img0

def reset_interaction_state():
    global start_point, end_point, is_drawing, is_moving, is_resizing, selected_rectangle, move_start_point
    start_point = None
    end_point = None
    is_drawing = False
    is_moving = False
    is_resizing = False
    selected_rectangle = -1
    move_start_point = None

def save_current_image():
    """按当前矩形绘制+红框2×放大，保存到 OUT_DIR。"""
    ensure_out_dir()
    base = img0.copy()
    rects = rectangles_map[img_idx]
    base = draw_rectangles_on(base, rects, selected_idx=-1, preview=None, preview_color=None)

    H, W = base.shape[:2]
    # 取第一个红框
    fr = next(((r0, r1, c) for (r0, r1, c) in rects if c==(0,0,255)), None)
    if fr is not None:
        (r0, r1, c) = fr
        (x1,y1),(x2,y2) = clip_rect_to_image(r0, r1, W, H)
        roi = base[y1:y2, x1:x2]
        inset = build_inset_from_roi_fixed2x(roi, c)
        base = paste_inset_right_bottom(base, inset)

    name = basename_noext(img_paths[img_idx]) + FILENAME_SUFFIX + '.png'
    outpath = os.path.join(OUT_DIR, name)
    cv2.imwrite(outpath, base)
    print(f'✅ 保存：{outpath}')

# =============== 鼠标交互 ===============
def draw_rectangle(event, x, y, flags, param):
    global start_point, end_point, is_drawing, is_moving, is_resizing
    global selected_rectangle, color, move_start_point, DISP_SCALE

    # 显示坐标 → 原图坐标
    x = int(x / DISP_SCALE)
    y = int(y / DISP_SCALE)

    rects = rectangles_map[img_idx]

    def hit_which(px, py):
        for i, (r0, r1, _) in enumerate(rects):
            if point_in_rect(px, py, r0, r1):
                return i
        return -1

    if event == cv2.EVENT_LBUTTONDOWN:
        idx = hit_which(x, y)
        selected_rectangle = idx
        if idx >= 0:
            r0, r1, _ = rects[idx]
            corner = near_corners(x, y, r0, r1)
            start_point, end_point = r0, r1
            if corner:
                is_resizing = True
                move_start_point = (x, y, corner)
            else:
                is_moving = True
                move_start_point = (x, y)
        else:
            is_drawing = True
            start_point = (x, y)
            end_point = (x, y)

    elif event == cv2.EVENT_MOUSEMOVE:
        if is_drawing and start_point is not None:
            end_point = (x, y)
        elif is_moving and selected_rectangle >= 0 and move_start_point is not None:
            dx = x - move_start_point[0]
            dy = y - move_start_point[1]
            r0, r1, c = rects[selected_rectangle]
            (x1,y1),(x2,y2) = norm_rect(r0, r1)
            new_r0 = (x1+dx, y1+dy)
            new_r1 = (x2+dx, y2+dy)
            rects[selected_rectangle] = (new_r0, new_r1, c)
            move_start_point = (x, y)
            start_point, end_point = new_r0, new_r1
        elif is_resizing and selected_rectangle >= 0 and move_start_point is not None:
            r0, r1, c = rects[selected_rectangle]
            (x1,y1),(x2,y2) = norm_rect(r0, r1)
            corner = move_start_point[2]
            if corner == 'tl':
                r0 = (x, y); r1 = (x2, y2)
            elif corner == 'tr':
                r0 = (x1, y); r1 = (x, y2)
            elif corner == 'bl':
                r0 = (x, y1); r1 = (x2, y)
            elif corner == 'br':
                r0 = (x1, y1); r1 = (x, y)
            r0, r1 = norm_rect(r0, r1)
            rects[selected_rectangle] = (r0, r1, c)
            start_point, end_point = r0, r1

    elif event == cv2.EVENT_LBUTTONUP:
        if is_drawing:
            is_drawing = False
            r0, r1 = norm_rect(start_point, end_point)
            rects.append((r0, r1, color))
            selected_rectangle = len(rects) - 1
        elif is_moving:
            is_moving = False
        elif is_resizing:
            is_resizing = False

    elif event == cv2.EVENT_RBUTTONDOWN:
        color = (0,255,0) if color == (0,0,255) else (0,0,255)

# =============== 主流程 ===============
def main():
    global img_paths, img_idx, img0, DISP_SCALE, MAX_DISPLAY
    global selected_rectangle, color

    img_paths = list_images(FOLDER_PATH)
    if not img_paths:
        raise RuntimeError(f'在文件夹中未找到图像：{FOLDER_PATH}')

    print(f'共找到 {len(img_paths)} 张图片。')
    ensure_out_dir()

    img0 = load_image(img_idx)
    cv2.namedWindow('Batch Fusion ROI (red 2x)', cv2.WINDOW_NORMAL)
    cv2.setMouseCallback('Batch Fusion ROI (red 2x)', draw_rectangle)

    while True:
        base = img0.copy()
        rects = rectangles_map[img_idx]

        # 主图矩形（含预览/选中高亮）
        preview = (start_point, end_point) if (is_drawing or is_moving or is_resizing) and start_point and end_point else None
        base = draw_rectangles_on(base, rects, selected_idx=selected_rectangle,
                                  preview=preview, preview_color=(color if preview else None))

        # 红框 2× 放大贴右下角
        H, W = base.shape[:2]
        fr = next(((r0, r1, c) for (r0, r1, c) in rects if c==(0,0,255)), None)
        if fr is not None:
            (r0, r1, c) = fr
            (x1,y1),(x2,y2) = clip_rect_to_image(r0, r1, W, H)
            roi = base[y1:y2, x1:x2]
            inset = build_inset_from_roi_fixed2x(roi, c)
            base = paste_inset_right_bottom(base, inset)

        # 显示缩放
        disp, DISP_SCALE = resize_to_fit(base, max_size=MAX_DISPLAY)
        title = f'Batch Fusion ROI (red 2x)  [{img_idx+1}/{len(img_paths)}]  {os.path.basename(img_paths[img_idx])}'
        cv2.setWindowTitle('Batch Fusion ROI (red 2x)', title)
        cv2.imshow('Batch Fusion ROI (red 2x)', disp)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

        elif key == ord('s'):
            save_current_image()

        elif key in (ord('n'), 83):   # 'n' 或 右方向键（部分系统返回83）
            if AUTOSAVE_ON_SWITCH:
                save_current_image()
            rectangles_map[img_idx] = rectangles_map[img_idx]  # 保留
            img_idx = min(len(img_paths)-1, img_idx+1)
            img0 = load_image(img_idx)
            reset_interaction_state()

        elif key in (ord('p'), 81):   # 'p' 或 左方向键（部分系统返回81）
            if AUTOSAVE_ON_SWITCH:
                save_current_image()
            rectangles_map[img_idx] = rectangles_map[img_idx]
            img_idx = max(0, img_idx-1)
            img0 = load_image(img_idx)
            reset_interaction_state()

        elif key == ord('d') and selected_rectangle >= 0:
            rects = rectangles_map[img_idx]
            del rects[selected_rectangle]
            selected_rectangle = -1

        elif key == ord('c'):
            rectangles_map[img_idx].clear()
            selected_rectangle = -1

        elif key == ord('r'):
            color = (0,0,255)
        elif key == ord('g'):
            color = (0,255,0)

        elif key == ord('+') or key == ord('='):
            MAX_DISPLAY = min(MAX_DISPLAY + 100, 2400)
        elif key == ord('-') or key == ord('_'):
            MAX_DISPLAY = max(MAX_DISPLAY - 100, 600)

    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
