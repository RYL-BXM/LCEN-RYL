# LCEN: Lightweight End-to-End Weakly Supervised Semantic Segmentation via Completeness Enhancement with Noise Suppression

This repository provides the official implementation of **LCEN**, a lightweight end-to-end weakly supervised semantic segmentation framework designed to improve pseudo-label completeness and suppress semantic noise under image-level supervision.

The corresponding paper:

> **LCEN: Lightweight End-to-End Weakly Supervised Semantic Segmentation via Completeness Enhancement with Noise Suppression**  
> Under review at *Pattern Analysis and Applications*.

If you find this repository useful for your research, please consider citing our work.

---

# 1. Introduction

Weakly Supervised Semantic Segmentation (WSSS) aims to generate accurate pixel-level semantic predictions using only image-level annotations. Despite recent progress in end-to-end WSSS frameworks, existing approaches still suffer from incomplete object activation, blurred semantic boundaries, semantic ambiguity, and background interference during pseudo-label generation.

To address these limitations, we propose LCEN, a lightweight end-to-end WSSS framework built upon frozen CLIP semantic priors. The proposed method introduces two collaborative enhancement modules:

### DCCM (Dual-Channel Contextual Modulation)

DCCM explicitly partitions feature representations into semantic and structural subspaces and performs adaptive spatial-channel modulation to enhance low-response regions while preserving object boundary consistency.

### HCSA (Hybrid Channel-Spatial Attention)

HCSA integrates multi-scale spatial attention with channel self-attention mechanisms to suppress semantic noise and improve discriminative feature representation.

Extensive experiments on PASCAL VOC 2012 and MS COCO 2014 demonstrate that LCEN achieves superior segmentation performance compared with existing state-of-the-art single-stage WSSS methods while maintaining lightweight computation and efficient end-to-end optimization.

---

# 2. Main Contributions

- A lightweight end-to-end weakly supervised semantic segmentation framework based on frozen CLIP semantic priors.
- A Dual-Channel Contextual Modulation (DCCM) module for enhancing activation completeness and preserving structural consistency.
- A Hybrid Channel-Spatial Attention (HCSA) module for suppressing semantic ambiguity and background interference.
- Superior segmentation performance on both PASCAL VOC 2012 and MS COCO 2014 benchmarks.



---

# 3. Experimental Results

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

# 4. Repository Structure

```text
LCEN-RYL/
├── WeCLIP_Plus/
│   ├── Decoder/
│   │   ├── clip/
│   │   ├── MaskMultiheadAttention.py
│   │   ├── TransDecoder.py
│   │   ├── TransDecoder_clip_dino.py
│   │   ├── TransDecoder_cls.py
│   │   ├── TransDecoder_seg.py
│   │   ├── TransDecoder_seg_tsne.py
│   │   └── Transformer.py
│   ├── DCCM.py
│   ├── HCSA.py
│   ├── PAR.py
│   ├── __init__.py
│   ├── conv_head.py
│   ├── dice_loss.py
│   ├── model_attn_aff_coco.py
│   ├── model_attn_aff_voc.py
│   ├── model_attn_aff_voc_seg.py
│   ├── segformer_head.py
│   ├── segformer_head_seg.py
│   └── test_msc_flip_voc_seg.py
├── .gitignore
├── LICENSE
├── README.md
├── coco 实验数据.log
├── 实验数据.log
├── dist_clip_coco.py
├── dist_clip_voc.py
├── generate_cams_coco14.py
├── generate_cams_voc12.py
├── requirements.txt
├── test_msc_flip_coco.py
├── test_msc_flip_seg.py
├── test_msc_flip_voc.py
├── vis_cam_from_npy.py
└── vis_coco_cam_from_npy.py
```
---

# 5. Installation

## Recommended Environment

- Ubuntu 20.04
- Python 3.8
- PyTorch 1.8.0
- CUDA 11.1
- NVIDIA RTX 4090

---

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

# 6. Requirements

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

## 7. Dataset Preparation

This repository does not host any datasets.
Please download the standard public datasets and organize them according to the following structure.

### Supported Datasets
- PASCAL VOC 2012
- MS COCO 2014

### Dataset Download
- PASCAL VOC 2012: http://host.robots.ox.ac.uk/pascal/VOC/
- MS COCO 2014: https://cocodataset.org/

### Recommended Dataset Structure
datasets/
├── VOC2012/
└── COCO2014/

After downloading, place the datasets in the above folders.
Please modify the dataset path in the configuration files before training.

---

# 8. Pretrained Weights

Please place pretrained models into:

```text
pretrained/
```

Supported pretrained backbones:

- CLIP ViT-B/16

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
