# Learning Mask-Aware Offsets: Two-branch Deformable Attention Networks for Inpainting with Masked Region Avoidance (WACV 2026)

Official implementation of "Learning Mask-Aware Offsets: Two-branch Deformable Attention Networks for Inpainting with Masked Region Avoidance", accepted at **WACV 2026**.

This repository provides the core code for the MADIN architecture, focusing on efficient and high-quality image inpainting by utilizing mask-aware deformable attention.

## 🚀 Key Features
- **MADIN Architecture**: A novel inpainting network using Mask-Aware Deformable Attention (MADA).
- **Masked Region Avoidance**: Improved attention mechanism that avoids sampling from masked/missing regions.
- **Memory Efficient**: Optimized attention calculation using `F.scaled_dot_product_attention`.
- **Flexible Demo**: Supports both single-image and folder-wise inference with manual or random masks.

---

## 🛠 Installation

### 1. Environment Setup
We recommend using Conda to manage your environment:

```bash
conda create -n mkMADIN python=3.10 -y
conda activate mkMADIN

# Install PyTorch (matching your CUDA version)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install requirements
pip install -r requirements.txt
```

### 2. Requirements
- Python 3.10+
- PyTorch 2.5.1+ (with CUDA support)
- einops
- PyYAML
- Pillow
- Matplotlib

---

## 📂 Dataset Preparation

The model is designed to work with standard inpainting datasets like CelebA-HQ or Places2. 

### Recommended Structure:
```text
/datasets/
  ├── CelebA-HQ/
  │   ├── train/ (RGB images)
  │   └── val/   (RGB images)
  └── masks/
      └── eval_masks/ (Binary masks: 255 for hole, 0 for background)
```

Adjust the paths in `config/config.yaml` to match your local directory structure.

---

## 🏋️ Training

To start training the model, use the `train.py` script:

```bash
conda activate mkMADIN
python train.py
```

- Hyperparameters such as `batch_size`, `num_epochs`, and `lambdas` (loss weights) can be modified in `config/config.yaml`.
- Checkpoints will be saved in the directory specified under `paths.checkpoint_save_path`.

---

## 🎨 Demo / Inference

You can run inference on a single image or an entire folder using `demo .py`.

### Single Image with Manual Mask
```bash
python "demo .py" --image ./sample.jpg --mask ./mask.png --checkpoint ./epoch_200.pth --output result.png
```

### Folder-wise Processing (Batch Inference)
```bash
# Process all images in a folder with random masks
python "demo .py" --image ./input_folder --checkpoint ./epoch_200.pth --output ./output_folder

# Process all images matching them with masks in another folder
python "demo .py" --image ./input_folder --mask ./mask_folder --checkpoint ./epoch_200.pth --output ./output_folder
```

---

## ⚙️ Configuration (`config.yaml`)
Key parameters in the configuration file:
- `img_size`: Resolution stages for the tiered architecture (e.g., [256, 128, 64, 32]).
- `batch_size`: Adjusted for VRAM usage (recommended: 2 or 4 for 256x256 resolution).
- `lambdas`: Weights for L1, Perceptual, Style, and Adversarial losses.

---

## 📝 Citation
If you find this work useful, please cite:
```bibtex
@inproceedings{madin2026,
  title={Learning Mask-Aware Offsets: Two-branch Deformable Attention Networks for Inpainting with Masked Region Avoidance},
  author={H. Oh et al.},
  booktitle={WACV},
  year={2026}
}
```