import os
import time
import numpy as np
import torch

from model import WaferRestorationNet


# ============================================================
# SETTINGS
# ============================================================

MODEL_PATH = "checkpoints/wafer_restoration.pth"

DATA_DIR = "data/train"

NUM_IMAGES = 3200

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# START
# ============================================================

print("=" * 60)
print("WaferVision Final Performance Benchmark")
print("=" * 60)

print("Using device:", DEVICE)


# ============================================================
# LOAD MODEL
# ============================================================

model = WaferRestorationNet().to(DEVICE)

checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE
)

model.load_state_dict(checkpoint)

model.eval()

print("Model loaded successfully")


# ============================================================
# DATA DIRECTORIES
# ============================================================

gt_dir = os.path.join(
    DATA_DIR,
    "GT"
)

lr_dir = os.path.join(
    DATA_DIR,
    "NoisyLR"
)


# ============================================================
# GET FILES
# ============================================================

files = sorted([
    f
    for f in os.listdir(gt_dir)
    if f.endswith(".npy")
])


# Make sure we don't request more images than available
NUM_IMAGES = min(
    NUM_IMAGES,
    len(files)
)

files = files[:NUM_IMAGES]

print("Images selected:", len(files))


# ============================================================
# RESULT STORAGE
# ============================================================

psnr_values = []
ssim_values = []
inference_times = []


# ============================================================
# PROCESS DATASET
# ============================================================

for i, filename in enumerate(files):

    # --------------------------------------------------------
    # Load images
    # --------------------------------------------------------

    lr = np.load(
        os.path.join(lr_dir, filename)
    ).astype(np.float32)

    gt = np.load(
        os.path.join(gt_dir, filename)
    ).astype(np.float32)


    # --------------------------------------------------------
    # Convert input to tensor
    # --------------------------------------------------------

    input_tensor = (
        torch.from_numpy(lr)
        .unsqueeze(0)
        .unsqueeze(0)
        .to(DEVICE)
    )


    # --------------------------------------------------------
    # Model inference
    # --------------------------------------------------------

    if DEVICE.type == "cuda":
        torch.cuda.synchronize()

    start_time = time.perf_counter()

    with torch.no_grad():
        output = model(input_tensor)

    if DEVICE.type == "cuda":
        torch.cuda.synchronize()

    end_time = time.perf_counter()

    inference_time = (
        end_time - start_time
    )

    inference_times.append(
        inference_time
    )


    # --------------------------------------------------------
    # Convert output
    # --------------------------------------------------------

    restored = (
        output
        .squeeze()
        .cpu()
        .numpy()
    )


    # --------------------------------------------------------
    # Normalize restored image
    # --------------------------------------------------------

    restored_min = restored.min()
    restored_max = restored.max()

    restored_norm = (
        (restored - restored_min)
        /
        (restored_max - restored_min + 1e-8)
    )


    # --------------------------------------------------------
    # Normalize ground truth
    # --------------------------------------------------------

    gt_min = gt.min()
    gt_max = gt.max()

    gt_norm = (
        (gt - gt_min)
        /
        (gt_max - gt_min + 1e-8)
    )


    # ========================================================
    # PSNR
    # ========================================================

    mse = np.mean(
        (restored_norm - gt_norm) ** 2
    )

    if mse < 1e-12:

        psnr = float("inf")

    else:

        psnr = (
            10
            *
            np.log10(
                1.0 / mse
            )
        )


    psnr_values.append(psnr)


    # ========================================================
    # SSIM
    # ========================================================

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    mu_x = restored_norm.mean()
    mu_y = gt_norm.mean()

    sigma_x = restored_norm.var()
    sigma_y = gt_norm.var()

    sigma_xy = np.mean(
        (restored_norm - mu_x)
        *
        (gt_norm - mu_y)
    )

    numerator = (
        (2 * mu_x * mu_y + C1)
        *
        (2 * sigma_xy + C2)
    )

    denominator = (
        (mu_x ** 2 + mu_y ** 2 + C1)
        *
        (sigma_x + sigma_y + C2)
    )

    ssim = numerator / (
        denominator + 1e-8
    )

    ssim_values.append(ssim)


    # ========================================================
    # PROGRESS
    # ========================================================

    if (i + 1) % 100 == 0:

        print(
            f"Processed {i + 1}/{NUM_IMAGES} images"
        )


# ============================================================
# FINAL RESULTS
# ============================================================

average_psnr = np.mean(
    psnr_values
)

average_ssim = np.mean(
    ssim_values
)

average_inference = np.mean(
    inference_times
)

fps = 1.0 / average_inference


# ============================================================
# DISPLAY FINAL RESULTS
# ============================================================

print()
print("=" * 60)
print("WaferVision Final Benchmark Results")
print("=" * 60)

print(
    f"Images tested       : {NUM_IMAGES}"
)

print(
    f"Average PSNR        : {average_psnr:.4f} dB"
)

print(
    f"Average SSIM        : {average_ssim:.4f}"
)

print(
    f"Average inference   : "
    f"{average_inference * 1000:.2f} ms"
)

print(
    f"Approx. FPS         : {fps:.2f}"
)

print("=" * 60)
print("Benchmark completed successfully.")
print("=" * 60)