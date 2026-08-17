import torch
import numpy as np
from torch.utils.data import DataLoader

from dataset import WaferDataset
from model import WaferRestorationNet


# -----------------------------
# Settings
# -----------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATA_PATH = "data/train"
MODEL_PATH = "checkpoints/wafer_restoration.pth"


# -----------------------------
# PSNR calculation
# -----------------------------
def calculate_psnr(pred, target):
    mse = torch.mean((pred - target) ** 2)

    if mse == 0:
        return float("inf")

    psnr = 10 * torch.log10(1.0 / mse)
    return psnr.item()


# -----------------------------
# Simple SSIM calculation
# -----------------------------
def calculate_ssim(pred, target):
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    mu_x = torch.mean(pred)
    mu_y = torch.mean(target)

    sigma_x = torch.var(pred)
    sigma_y = torch.var(target)

    sigma_xy = torch.mean((pred - mu_x) * (target - mu_y))

    numerator = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
    denominator = (
        (mu_x ** 2 + mu_y ** 2 + C1)
        * (sigma_x + sigma_y + C2)
    )

    return (numerator / denominator).item()


# -----------------------------
# Load model
# -----------------------------
print("Using device:", DEVICE)

model = WaferRestorationNet().to(DEVICE)

checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE
)

model.load_state_dict(checkpoint)

model.eval()

print("Model loaded successfully")


# -----------------------------
# Load dataset
# -----------------------------
dataset = WaferDataset(DATA_PATH)

loader = DataLoader(
    dataset,
    batch_size=1,
    shuffle=False
)


# -----------------------------
# Evaluation
# -----------------------------
total_psnr = 0
total_ssim = 0

with torch.no_grad():

    for i, (lr, gt) in enumerate(loader):

        lr = lr.to(DEVICE)
        gt = gt.to(DEVICE)

        # Model prediction
        restored = model(lr)

        # Normalize to 0-1 for evaluation
        gt_min = gt.min()
        gt_max = gt.max()

        gt_norm = (gt - gt_min) / (gt_max - gt_min + 1e-8)

        restored_min = restored.min()
        restored_max = restored.max()

        restored_norm = (
            (restored - restored_min)
            / (restored_max - restored_min + 1e-8)
        )

        psnr = calculate_psnr(
            restored_norm,
            gt_norm
        )

        ssim = calculate_ssim(
            restored_norm,
            gt_norm
        )

        total_psnr += psnr
        total_ssim += ssim

        if i % 100 == 0:
            print(
                f"Image [{i}/{len(dataset)}] "
                f"PSNR: {psnr:.4f} dB "
                f"SSIM: {ssim:.4f}"
            )


# -----------------------------
# Final results
# -----------------------------
average_psnr = total_psnr / len(dataset)
average_ssim = total_ssim / len(dataset)

print("\n==============================")
print("WaferVision Evaluation Results")
print("==============================")
print(f"Images evaluated : {len(dataset)}")
print(f"Average PSNR     : {average_psnr:.4f} dB")
print(f"Average SSIM     : {average_ssim:.4f}")
print("==============================")