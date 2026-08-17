import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(self, channels=64):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1)
        )

    def forward(self, x):
        return x + self.block(x)


class WaferRestorationNet(nn.Module):

    def __init__(self):
        super().__init__()

        # Feature extraction
        self.head = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1),
            nn.ReLU(inplace=True)
        )

        # Feature restoration
        self.residual_blocks = nn.Sequential(
            ResidualBlock(64),
            ResidualBlock(64),
            ResidualBlock(64),
            ResidualBlock(64)
        )

        # 2x super-resolution
        self.upsample = nn.Sequential(
            nn.Conv2d(64, 256, 3, padding=1),
            nn.PixelShuffle(2),
            nn.ReLU(inplace=True)
        )

        # Final image reconstruction
        self.tail = nn.Conv2d(64, 1, 3, padding=1)

    def forward(self, x):

        x = self.head(x)

        residual = self.residual_blocks(x)

        x = x + residual

        x = self.upsample(x)

        x = self.tail(x)

        return x


if __name__ == "__main__":

    model = WaferRestorationNet()

    test_input = torch.randn(1, 1, 128, 128)

    output = model(test_input)

    print("Input shape :", test_input.shape)
    print("Output shape:", output.shape)

    parameters = sum(p.numel() for p in model.parameters())

    print("Parameters:", parameters)