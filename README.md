# Learning Mask-Aware Offsets: Two-branch Deformable Attention Networks for Inpainting with Masked Region Avoidance (WACV 2026)

Official implementation of ["Learning Mask-Aware Offsets: Two-branch Deformable Attention Networks for Inpainting with Masked Region Avoidance"](https://openaccess.thecvf.com/content/WACV2026/html/Oh_Learning_Mask-Aware_Offsets_Two-branch_Deformable_Attention_Networks_for_Inpainting_with_WACV_2026_paper.html), accepted at **WACV 2026**.

## 🛠 Installation

### 1. Environment Setup

We recommend using Conda to manage your environment:

```bash
conda create -n mkMADIN python=3.10 -y
conda activate mkMADIN

Install PyTorch (matching your CUDA version)

pip install -r requirements.txt
```

### 2. Requirements

- Python 3.10
- PyTorch 2.5.1
- einops
- PyYAML
- Pillow
- Matplotlib

---

## 📦 Pretrained Weights

You can download the pretrained checkpoints from the following links:

- [CelebA-HQ](https://drive.google.com/file/d/16om30aVwsm1RGscF6supd-mokW3T7hJi/view?usp=sharing)
- [Places2](https://drive.google.com/file/d/1cfwh-DlUkwg6PwPqrp400vrmgdWKMj9O/view?usp=sharing)

---

## 📂 Dataset Preparation

The model is designed to work with standard inpainting datasets like CelebA-HQ or Places2.

### Recommended Structure

```text
/datasets/
  ├── (dataset_name)/
  │   ├── train/ 
  │   └── val/   
  └── masks/
      └── eval_masks/ (Binary masks: 255 for hole, 0 for background)
```

Adjust the paths in `config/config.yaml` to match your local directory structure.

---

## 🏋️ Training
>
> Currently, this implementation supports **256x256 resolution** only.

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
python "demo .py" --image ./sample.jpg --mask ./mask.png --checkpoint (path/to/pth) --output result.png
```

### Folder-wise Processing (Batch Inference)

```bash
# Process all images in a folder with random masks
python "demo .py" --image ./input_folder --checkpoint (path/to/pth) --output ./output_folder

# Process all images matching them with masks in another folder
python "demo .py" --image ./input_folder --mask ./mask_folder --checkpoint (path/to/pth) --output ./output_folder
```

---

## ⚙️ Configuration (`config.yaml`)

Key parameters in the configuration file:

- `img_size`: Resolution of input image.
- `batch_size`: Adjusted for VRAM usage (recommended: 2 or 4 for 256x256 resolution).
- `lambdas`: Weights for L1, Perceptual, Style, and Adversarial losses.

---

## 📝 Citation

If you find this work useful, please cite:

```bibtex
@inproceedings{oh2026learning,
  title={Learning Mask-Aware Offsets: Two-branch Deformable Attention Networks for Inpainting with Masked Region Avoidance},
  author={Oh, Hyeongseok and Paik, Joonki},
  booktitle={Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision},
  pages={1022--1031},
  year={2026}
}
```
