# -*- coding: utf-8 -*-
import os
import numpy as np
import cv2
from PIL import Image
from tqdm import tqdm

# 路径配置
npy_dir = './final/ablation/voc_baseline'
img_root = 'D:/bxm/VOCdevkit/VOC2012/JPEGImages'
save_dir = os.path.join(npy_dir, 'cam_overlay_png')  # 新保存路径
os.makedirs(save_dir, exist_ok=True)

# 类别名称（确保与 class_id 对应）
from clip.clip_text import new_class_names

class_names = new_class_names

# 获取所有 npy 文件
npy_files = [f for f in os.listdir(npy_dir) if f.endswith('.npy')]

for npy_file in tqdm(npy_files):
    npy_path = os.path.join(npy_dir, npy_file)
    try:
        data = np.load(npy_path, allow_pickle=True).item()
    except Exception as e:
        print(f"[Error] Failed to load {npy_path}: {e}")
        continue

    keys = data.get("keys", [])
    cams = data.get("attn_highres", None)

    if cams is None or len(keys) == 0:
        print(f"[Warning] Empty CAM or keys in {npy_file}")
        continue

    image_name = npy_file.replace('.npy', '.jpg')
    image_path = os.path.join(img_root, image_name)
    if not os.path.exists(image_path):
        print(f"[Missing] Image not found: {image_path}")
        continue

    ori_img = np.array(Image.open(image_path).convert("RGB"))
    ori_img = cv2.resize(ori_img, (cams.shape[2], cams.shape[1]))

    # 初始化一个空的融合热力图
    combined_cam = np.zeros_like(cams[0], dtype=np.float32)

    for i, class_id in enumerate(keys):
        cam = cams[i]
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        combined_cam = np.maximum(combined_cam, cam)  # 保留每个像素的最大响应值

    # 将融合的 CAM 映射为伪彩色并叠加到原图
    combined_cam = (combined_cam - combined_cam.min()) / (combined_cam.max() - combined_cam.min() + 1e-8)
    cam_uint8 = np.uint8(255 * combined_cam)
    cam_color = cv2.applyColorMap(cam_uint8, cv2.COLORMAP_JET)
    overlayed = cv2.addWeighted(ori_img, 0.5, cam_color, 0.5, 0)

    # 保存图像
    save_name = image_name.replace('.jpg', '_overlay.png')
    save_path = os.path.join(save_dir, save_name)
    cv2.imwrite(save_path, overlayed)
