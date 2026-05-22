# LCEN: Lightweight End-to-End Weakly Supervised Semantic Segmentation via Completeness Enhancement with Noise Suppression

This repository provides the official implementation of **LCEN**, a lightweight end-to-end weakly supervised semantic segmentation framework designed to improve pseudo-label completeness and suppress semantic noise under image-level supervision.

The corresponding paper:

> **LCEN: Lightweight End-to-End Weakly Supervised Semantic Segmentation via Completeness Enhancement with Noise Suppression**  
> Under review at *Pattern Analysis and Applications*.

If you find this repository useful for your research, please consider citing our work.

---

# 1. Introduction

Weakly Supervised Semantic Segmentation (WSSS) aims to generate accurate pixel-level semantic predictions using only image-level annotations. Despite recent progress in end-to-end WSSS frameworks, existing approaches still suffer from two critical issues:

- incomplete activation in low-response regions,
- erroneous activation caused by semantic co-occurrence,
- blurred semantic boundaries,
- and background interference during pseudo-label generation.

To address these challenges, we propose **LCEN**, a lightweight end-to-end WSSS framework built upon frozen CLIP semantic priors. LCEN introduces two collaborative decoder enhancement modules:

### DCCM (Dual-Channel Contextual Modulation)

DCCM explicitly partitions features into semantic and structural subspaces and performs adaptive spatial-channel modulation to enhance low-response regions and preserve boundary structures.

### HCSA (Hybrid Channel-Spatial Attention)

HCSA integrates multi-scale spatial attention with multi-head channel self-attention to suppress background noise and improve semantic discriminability.

Extensive experiments on the **PASCAL VOC 2012** and **MS COCO 2014** benchmarks demonstrate that LCEN achieves superior segmentation performance compared with existing state-of-the-art single-stage WSSS methods while maintaining lightweight computation and efficient end-to-end optimization.

---

# 2. Framework Overview

<p align="center">
<img src="assets/framework.png" width="95%">
</p>

> Overall architecture of the proposed LCEN framework.

---

# 3. Visualization Results

## PASCAL VOC 2012

<p align="center">
<img src="assets/voc_vis.png" width="95%">
</p>

---

## MS COCO 2014

<p align="center">
<img src="assets/coco_vis.png" width="95%">
</p>

---

# 4. Experimental Results

## PASCAL VOC 2012

| Method | Backbone | Val mIoU | Test mIoU |
|---|---|---|---|
| WeCLIP | ViT-B | 76.4 | 77.2 |
| ExCEL | ViT-B | 78.4 | 78.5 |
| **LCEN (Ours)** | ViT-B | **81.2** | **83.3** |

---

## MS COCO 2014

| Method | Backbone | mIoU |
|---|---|---|
| WeCLIP | ViT-B | 47.1 |
| ExCEL | ViT-B | 50.3 |
| **LCEN (Ours)** | ViT-B | **53.2** |

---

# 5. Repository Structure

```text
LCEN-RYL/
├── datasets/
│   ├── VOC2012/
│   ├── SBD/
│   └── COCO2014/
│
├── pretrained/
│
├── WeCLIP_Plus/
│   ├── models/
│   │   ├── WeCLIP_Plus.py
│   │   ├── SegFormer_head.py
│   │   ├── TransDecoder.py
│   │   ├── PAR.py
│   │   ├── FCM.py
│   │   └── SCSA.py
│   │
│   ├── utils/
│   ├── scripts/
│   └── datasets/
│
├── dist_clip_voc.py
├── dist_clip_coco.py
├── eval_voc.py
├── eval_coco.py
│
├── requirements.txt
└── README.md
```

---

# 6. Installation

## Clone Repository

```bash
git clone https://github.com/RYL-BXM/LCEN-RYL.git
cd LCEN-RYL
```

---

## Create Conda Environment

```bash
conda create -n lcen python=3.8 -y
conda activate lcen
```

---

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

# 7. Requirements

```text
mmcv_full==1.2.7
matplotlib==3.3.3
tqdm==4.46.1
omegaconf==2.0.0
numpy==1.18.5
timm==0.3.2
imageio==2.9.0
mmcv==1.3.17
Pillow==8.4.0
scikit_learn==1.0.1
```

---

# 8. Dataset Preparation

Our experiments are conducted on the following benchmark datasets:

- PASCAL VOC 2012
- MS COCO 2014

## Download Links

### PASCAL VOC 2012
http://host.robots.ox.ac.uk/pascal/VOC/

### SBD Dataset
http://home.bharathh.info/pubs/codes/SBD/download.html

### MS COCO 2014
https://cocodataset.org/

---

## Recommended Dataset Structure

```text
datasets/
├── VOC2012/
├── SBD/
└── COCO2014/
```

Please modify the dataset root path in the corresponding configuration files before training or evaluation.

---

# 9. Pretrained Weights

Please place pretrained models into:

```text
pretrained/
```

Supported pretrained backbones:

- CLIP ViT-B/16
- DINOv2

---

# 10. Model Training

The project provides independent distributed training scripts for different datasets.

## Train on PASCAL VOC 2012

```bash
python dist_clip_voc.py
```

---

## Train on MS COCO 2014

```bash
python dist_clip_coco.py
```

Training checkpoints are automatically saved to:

```text
checkpoints/
```

Training logs are stored in:

```text
logs/
```

---

# 11. Evaluation

## Evaluate on PASCAL VOC 2012

```bash
python eval_voc.py
```

---

## Evaluate on MS COCO 2014

```bash
python eval_coco.py
```

Evaluation results and visualization outputs are automatically saved to:

```text
results/
```

---

# 12. Experimental Settings

| Item | Setting |
|---|---|
| Backbone | ViT-B/16 |
| Optimizer | AdamW |
| Learning Rate | 2e-5 |
| Weight Decay | 0.01 |
| Batch Size | 16 / 32 |
| Crop Size | 320×320 |
| Scheduler | Poly LR |
| Training Iterations | 40K / 80K |

---

# 13. Reproducibility

To ensure reproducibility, this repository provides:

- complete training scripts,
- evaluation scripts,
- preprocessing pipelines,
- fixed hyperparameter settings,
- official evaluation protocols,
- and identical experimental configurations reported in the manuscript.

All experiments are conducted using the same settings described in the paper.

---

# 14. Citation

If you use this repository in your research, please cite:

```bibtex
@article{ren2026lcen,
  title={LCEN: Lightweight End-to-End Weakly Supervised Semantic Segmentation via Completeness Enhancement with Noise Suppression},
  author={Ren, YiLong and Zhao, XueZhuan and Li, LingLing and Shao, XiaoYan and Ren, Ning and Zhang, Jian},
  journal={Pattern Analysis and Applications},
  year={2026}
}
```

---

# 15. Acknowledgements

This repository is built upon several outstanding open-source projects and frameworks:

- WeCLIP
- CLIP
- DINOv2
- PyTorch
- MMCV

We sincerely thank the authors for their valuable contributions to the computer vision community.

---

# 16. Contact

If you encounter any problems during code reproduction or implementation, please contact:

📩 Email: renyilong2026@163.com

---

# 17. License

This project is released under the MIT License.
