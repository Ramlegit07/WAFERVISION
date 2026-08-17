import os
import numpy as np
import torch
from torch.utils.data import Dataset


class WaferDataset(Dataset):

    def __init__(self, root_dir):
        self.gt_dir = os.path.join(root_dir, "GT")
        self.lr_dir = os.path.join(root_dir, "NoisyLR")

        self.files = sorted([
            f for f in os.listdir(self.gt_dir)
            if f.endswith(".npy")
        ])

        print(f"Found {len(self.files)} training pairs")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):

        filename = self.files[idx]

        gt_path = os.path.join(self.gt_dir, filename)
        lr_path = os.path.join(self.lr_dir, filename)

        # Load numpy arrays
        gt = np.load(gt_path).astype(np.float32)
        lr = np.load(lr_path).astype(np.float32)

        # Convert to tensors
        gt = torch.from_numpy(gt).unsqueeze(0)
        lr = torch.from_numpy(lr).unsqueeze(0)

        return lr, gt


if __name__ == "__main__":

    dataset = WaferDataset("data/train")

    print("Dataset size:", len(dataset))

    lr, gt = dataset[0]

    print("Input NoisyLR:", lr.shape)
    print("Ground Truth:", gt.shape)