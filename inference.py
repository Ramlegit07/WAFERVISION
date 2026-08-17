import torch
import numpy as np
import matplotlib.pyplot as plt

from model import WaferRestorationNet



# Settings
MODEL_PATH = "checkpoints/wafer_restoration.pth"
IMAGE_PATH = "data/train/NoisyLR/000000.npy"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")



# Load model
print("Using device:", DEVICE)

model = WaferRestorationNet().to(DEVICE)

checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE
)

model.load_state_dict(checkpoint)
model.eval()

print("Model loaded successfully")



# Load degraded wafer image
lr = np.load(IMAGE_PATH).astype(np.float32)

print("Input shape:", lr.shape)


# Convert to tensor
lr_tensor = torch.from_numpy(lr).unsqueeze(0).unsqueeze(0)

lr_tensor = lr_tensor.to(DEVICE)



# AI restoration
with torch.no_grad():
    restored = model(lr_tensor)


restored = restored.squeeze().cpu().numpy()



# Display result

plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.imshow(lr, cmap="gray")
plt.title("Degraded Wafer Image")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(restored, cmap="gray")
plt.title("AI Restored Wafer Image")
plt.axis("off")

plt.tight_layout()
plt.show()