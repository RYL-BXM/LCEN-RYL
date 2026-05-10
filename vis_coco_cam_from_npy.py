# # -*- coding: utf-8 -*-
# import os
# import numpy as np
# import cv2
# from PIL import Image
# from tqdm import tqdm
#
# # ==== 修改路径（根据实际情况） ====
# npy_dir = 'F:/WeCLIP/test_msc_flip_coco/results/val/logit'  # 存放 .npy 的路径
# img_root = 'D:/ToCo-main/datasets/coco/MSCOCO/JPEGImages/val'  # 原图像路径
# save_dir = os.path.join(npy_dir, 'cam_png')  # 输出 PNG 的路径
# os.makedirs(save_dir, exist_ok=True)
#
# # ==== 类别名列表（COCO） ====
# from clip.clip_text import new_class_names_coco
# class_names = new_class_names_coco
#
# # ==== 遍历 .npy 文件 ====
# npy_files = [f for f in os.listdir(npy_dir) if f.endswith('.npy')]
#
# for npy_file in tqdm(npy_files, desc="Saving CAM PNGs"):
#     npy_path = os.path.join(npy_dir, npy_file)
#     data = np.load(npy_path, allow_pickle=True).item()
#
#     # === 提取数据 ===
#     keys = data["segs"]
#     cams = data["msc_segs"]  # shape: [C, H, W]
#
#     image_name = npy_file.replace('.npy', '.jpg')
#     image_path = os.path.join(img_root, image_name)
#
#     if not os.path.exists(image_path):
#         print(f"[❌] Missing image: {image_path}")
#         continue
#
#     # 加载原图并调整大小
#     ori_img = np.array(Image.open(image_path).convert("RGB"))
#     ori_img = cv2.resize(ori_img, (cams.shape[2], cams.shape[1]))
#
#     for i in range(len(keys)):
#         cam = cams[i]
#         cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)  # 归一化到 0~1
#         cam = np.uint8(255 * cam)
#         cam_color = cv2.applyColorMap(cam, cv2.COLORMAP_JET)
#         overlayed = cv2.addWeighted(ori_img, 0.5, cam_color, 0.5, 0)
#
#         class_id = keys[i]
#         if class_id >= len(class_names):
#             print(f"[⚠️] Invalid class ID {class_id} in {npy_file}")
#             continue
#
#         class_name = class_names[class_id]
#         save_name = image_name.replace('.jpg', f'_{class_name}.png')
#         save_path = os.path.join(save_dir, save_name)
#         cv2.imwrite(save_path, overlayed)

# -*- coding: utf-8 -*-
import os
import numpy as np
import cv2
from PIL import Image
from tqdm import tqdm

# 配置路径（请根据你的实际路径修改）
npy_dir = 'E:/ryl/WeCLIP+3-coco/final/ablation/coco_test'  # 存放 .npy 的路径
img_root = 'D:/ToCo-main/datasets/coco/MSCOCO/JPEGImages/val'  # 原始 COCO 图像路径
save_dir = os.path.join(npy_dir, 'cam_png_combined')
os.makedirs(save_dir, exist_ok=True)

# 使用 COCO 类别
from clip.clip_text import new_class_names_coco as class_names

# 获取所有 .npy 文件
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

    # 构造 COCO 原图路径
    image_name = npy_file.replace('.npy', '.jpg')
    image_path = os.path.join(img_root, image_name)
    if not os.path.exists(image_path):
        print(f"[Missing] Image not found: {image_path}")
        continue

    ori_img = np.array(Image.open(image_path).convert("RGB"))
    H, W = cams.shape[1], cams.shape[2]
    ori_img = cv2.resize(ori_img, (W, H))

    # CAM 累加图
    combined_cam = np.zeros((H, W), dtype=np.float32)
    for cam in cams:
        cam_norm = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        combined_cam += cam_norm

    # 平均归一化
    combined_cam /= len(keys)
    combined_cam = np.uint8(255 * combined_cam)

    # 生成彩色热力图并叠加
    cam_color = cv2.applyColorMap(combined_cam, cv2.COLORMAP_JET)
    overlayed = cv2.addWeighted(ori_img, 0.5, cam_color, 0.5, 0)

    # 保存叠加图
    save_name = image_name.replace('.jpg', '_combined.png')
    save_path = os.path.join(save_dir, save_name)
    cv2.imwrite(save_path, overlayed)

