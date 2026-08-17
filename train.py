import torch
from torch.utils.data import DataLoader

from dataset import WaferDataset
from model import WaferRestorationNet



# Configuration

DATA_DIR = "data/train"

BATCH_SIZE = 8
EPOCHS = 20
LEARNING_RATE = 1e-4

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using device:", DEVICE)



# Dataset


dataset = WaferDataset(DATA_DIR)

loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)



# Model


model = WaferRestorationNet().to(DEVICE)



# Loss and optimizer


criterion = torch.nn.L1Loss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


# Training


for epoch in range(EPOCHS):

    model.train()

    total_loss = 0.0

    for batch_idx, (lr, gt) in enumerate(loader):

        lr = lr.to(DEVICE)
        gt = gt.to(DEVICE)

        # Forward pass
        output = model(lr)

        # Calculate loss
        loss = criterion(output, gt)

        # Backpropagation
        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

        if batch_idx % 50 == 0:
            print(
                f"Epoch [{epoch + 1}/{EPOCHS}] "
                f"Batch [{batch_idx}/{len(loader)}] "
                f"Loss: {loss.item():.6f}"
            )

    average_loss = total_loss / len(loader)

    print(
        f"Epoch [{epoch + 1}/{EPOCHS}] "
        f"Average Loss: {average_loss:.6f}"
    )



# Save trained model


torch.save(
    model.state_dict(),
    "checkpoints/wafer_restoration.pth"
)

print("Training completed.")
print("Model saved to checkpoints/wafer_restoration.pth")