# 🔬 WaferVision

## AI-Based Restoration of Degraded Semiconductor Inspection Images

WaferVision is a deep-learning-based image restoration system designed to recover visual information from degraded semiconductor inspection images affected by noise, blur, and reduced spatial resolution.

The system uses a lightweight residual deep-learning architecture to restore degraded 128×128 images into higher-resolution 256×256 images.

---

## Problem Statement

Semiconductor inspection images can suffer from:

- Gaussian noise
- Speckle noise
- Blur
- Reduced spatial resolution
- Loss of fine inspection details

These degradations can make defects and structural features difficult to inspect.

WaferVision addresses this problem by using a trained neural network to reconstruct a clearer and higher-resolution image from a degraded input.

---

##  Proposed Solution

The proposed system follows this pipeline:

```text
Degraded Wafer Image
        │
        ▼
   Pre-processing
        │
        ▼
Feature Extraction
        │
        ▼
Residual Restoration
        │
        ▼
2× Pixel Shuffle
Super Resolution
        │
        ▼
Restored Wafer Image
        │
        ├──────────────► PSNR
        │
        └──────────────► SSIM