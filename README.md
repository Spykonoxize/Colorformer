# Colorformer

Colorformer is a black-and-white image colorization project built with a Transformer-based U-Net generator and a PatchGAN discriminator in PyTorch.

The repository provides a modular Python package (`src/colorformer/`) for reusable training and inference.

## What This Project Does

- Trains a colorization generator (`L -> ab`) in Lab color space.
- Uses a conditional PatchGAN discriminator for more realistic colors.
- Combines WGAN-GP, L1 reconstruction, and VGG perceptual losses.
- Supports both quick smoke tests and longer GPU runs.

## Technical Overview

1. Input:
- `L` channel (luminance, normalized grayscale)

2. Output:
- `ab` channels (predicted chrominance)

3. Generator:
- U-Net structure with `ColorFormerBlock`
- Local window attention (`LWMHSA`)
- Residual paths and skip connections

4. Discriminator:
- Conditional PatchGAN on `[L, ab]`

5. Total loss:
- `lambda_gan * L_GAN + lambda_l1 * L1 + lambda_vgg * L_VGG`

## Repository Structure

- `src/colorformer/data.py`: preprocessing and dataset utilities
- `src/colorformer/model.py`: generator and discriminator architectures
- `src/colorformer/losses.py`: GAN, L1, and VGG losses
- `src/colorformer/trainer.py`: training loop and checkpointing
- `src/colorformer/inference.py`: inference helpers
- `scripts/train.py`: modular training entrypoint on a local dataset
- `scripts/infer.py`: inference from a trained checkpoint
- `docs/project_structure.md`: project structure notes

## Installation

1. Create a Python environment (recommended: Python 3.10+).
2. Install dependencies:

```bash
pip install -r requirements.txt
pip install -e .
```

## Quick Usage

### Local Training

Set your training images directory:

```bash
set COLORFORMER_DATASET_DIR=E:\path\to\images
python scripts/train.py
```

### Inference

```bash
python scripts/infer.py --checkpoint outputs/colorformer_training/checkpoints/checkpoint_epoch_010.pt --image <image.jpg> --output outputs/colorized.png
```

## Generated Outputs

- Checkpoints: `outputs/.../checkpoints/*.pt`
- Per-epoch previews: `outputs/.../samples/epoch_XXX.png`
- Inference preview: `outputs/.../inference_preview.png`

## GPU Compatibility Note

For RTX 50-series cards, use a PyTorch build compatible with your GPU architecture. A working setup used in this repo:

```bash
pip install --upgrade --index-url https://download.pytorch.org/whl/cu130 torch torchvision torchaudio
```

## What You Can Do With This Repo

- Colorize grayscale images.
- Run quick hyperparameter experiments.
- Reuse `src/colorformer/` as a base for research or demos.

## Current Limitations

- Quality strongly depends on dataset size and training time.
- Some scenes remain ambiguous due to the ill-posed nature of colorization.

## License

This project is released under the MIT License. See `LICENSE`.
