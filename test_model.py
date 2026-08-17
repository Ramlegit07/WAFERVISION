import numpy as np
import torch
import matplotlib.pyplot as plt

from model import WaferRestorationNet


# -----------------------------
# Load model
# -----------------------------

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

model = WaferRestorationNet().to(device)

model.load_state_dict(
    torch.load(
        "checkpoints/wafer_restoration.pth",
        map_location=device
    )
)

model.eval()

print("Model loaded successfully")


# -----------------------------
# Select test image
# -----------------------------

filename = "000000.npy"

lr_path = f"data/train/NoisyLR/{filename}"
gt_path = f"data/train/GT/{filename}"


# -----------------------------
# Load images
# -----------------------------

lr = np.load(lr_path).astype(np.float32)
gt = np.load(gt_path).astype(np.float32)


# -----------------------------
# Prepare input
# -----------------------------

input_tensor = torch.from_numpy(lr)
input_tensor = input_tensor.unsqueeze(0).unsqueeze(0)
input_tensor = input_tensor.to(device)


# -----------------------------
# Run inference
# -----------------------------

with torch.no_grad():
    output = model(input_tensor)


restored = output.squeeze().cpu().numpy()


# -----------------------------
# Print information
# -----------------------------

print("Input shape    :", lr.shape)
print("Restored shape :", restored.shape)
print("Ground truth   :", gt.shape)

print("Input range    :", lr.min(), lr.max())
print("Restored range :", restored.min(), restored.max())
print("GT range      :", gt.min(), gt.max())


# -----------------------------
# Save restored image
# -----------------------------

np.save(
    "outputs/restored_000000.npy",
    restored
)


# -----------------------------
# Visual comparison
# -----------------------------

plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.imshow(lr, cmap="gray")
plt.title("Degraded Input")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(restored, cmap="gray")
plt.title("AI Restored")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(gt, cmap="gray")
plt.title("Ground Truth")
plt.axis("off")

plt.tight_layout()

plt.savefig(
    "outputs/comparison_000000.png",
    dpi=150
)

plt.show()

print("Comparison saved to outputs/comparison_000000.png")