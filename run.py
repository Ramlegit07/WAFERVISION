import os
import sys
import numpy as np
import torch
import torch.nn as nn


# =========================================================
# MODEL ARCHITECTURE
# =========================================================

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

        # Final reconstruction
        self.tail = nn.Conv2d(64, 1, 3, padding=1)

    def forward(self, x):

        x = self.head(x)

        residual = self.residual_blocks(x)

        x = x + residual

        x = self.upsample(x)

        x = self.tail(x)

        return x


# =========================================================
# CONFIGURATION
# =========================================================

INPUT_SIZE = 128
OUTPUT_SIZE = 256


# =========================================================
# FIND MODEL CHECKPOINT
# =========================================================

def find_checkpoint():

    possible_paths = [
        os.path.join("models", "wafer_restoration.pth"),
        os.path.join("checkpoints", "wafer_restoration.pth")
    ]

    for path in possible_paths:

        if os.path.isfile(path):
            return path

    raise FileNotFoundError(
        "Model checkpoint not found.\n"
        "Expected:\n"
        "models/wafer_restoration.pth"
    )


# =========================================================
# LOAD MODEL
# =========================================================

def load_model(device):

    checkpoint_path = find_checkpoint()

    print(f"Loading model from: {checkpoint_path}")

    model = WaferRestorationNet()

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False
    )

    # Handle checkpoint formats
    if isinstance(checkpoint, dict):

        if "state_dict" in checkpoint:

            state_dict = checkpoint["state_dict"]

        elif "model_state_dict" in checkpoint:

            state_dict = checkpoint["model_state_dict"]

        else:

            state_dict = checkpoint

    else:

        state_dict = checkpoint

    # Remove DataParallel "module." prefix
    cleaned_state_dict = {}

    for key, value in state_dict.items():

        if key.startswith("module."):

            key = key[len("module."):]

        cleaned_state_dict[key] = value

    model.load_state_dict(cleaned_state_dict)

    model.to(device)

    model.eval()

    print("Model loaded successfully.")

    return model


# =========================================================
# PREPARE INPUT
# =========================================================

def prepare_input(array):

    array = np.asarray(array)

    # Remove unnecessary dimensions
    array = np.squeeze(array)

    # Must be grayscale
    if array.ndim != 2:

        raise ValueError(
            f"Expected grayscale 2D array (H,W), "
            f"received {array.shape}"
        )

    # Convert to float32
    array = array.astype(np.float32)

    # Remove NaN and Inf
    array = np.nan_to_num(
        array,
        nan=0.0,
        posinf=1.0,
        neginf=0.0
    )

    # Normalize to [0,1]
    min_value = array.min()
    max_value = array.max()

    if max_value > 1.0:

        if max_value <= 255.0:

            array = array / 255.0

        else:

            if max_value > min_value:

                array = (
                    (array - min_value)
                    / (max_value - min_value)
                )

            else:

                array = np.zeros_like(array)

    array = np.clip(array, 0.0, 1.0)

    # H,W -> 1,1,H,W
    tensor = torch.from_numpy(array)

    tensor = tensor.unsqueeze(0).unsqueeze(0)

    return tensor


# =========================================================
# RESTORE IMAGE
# =========================================================

def restore_image(model, array, device):

    tensor = prepare_input(array)

    # Check input resolution
    if tensor.shape[-2:] != (
        INPUT_SIZE,
        INPUT_SIZE
    ):

        raise ValueError(
            f"Expected input resolution "
            f"{INPUT_SIZE}x{INPUT_SIZE}, "
            f"received "
            f"{tensor.shape[-2]}x{tensor.shape[-1]}"
        )

    tensor = tensor.to(device)

    with torch.no_grad():

        output = model(tensor)

    # Tensor -> NumPy
    output = output.squeeze().cpu().numpy()

    # Remove NaN / Inf
    output = np.nan_to_num(
        output,
        nan=0.0,
        posinf=1.0,
        neginf=0.0
    )

    # Ensure 2D grayscale
    if output.ndim == 3:

        output = np.squeeze(output)

    # Check output resolution
    if output.shape != (
        OUTPUT_SIZE,
        OUTPUT_SIZE
    ):

        raise ValueError(
            f"Model produced incorrect output shape "
            f"{output.shape}. "
            f"Expected "
            f"({OUTPUT_SIZE},{OUTPUT_SIZE})"
        )

    # Required range
    output = np.clip(
        output,
        0.0,
        1.0
    )

    # Required datatype
    output = output.astype(np.float32)

    return output


# =========================================================
# MAIN
# =========================================================

def main():

    # Check arguments
    if len(sys.argv) != 3:

        print(
            "\nUsage:\n"
            "python run.py <input-dir> <output-dir>\n"
        )

        print(
            "Example:\n"
            "python run.py test_input test_output\n"
        )

        sys.exit(1)

    input_dir = sys.argv[1]

    output_dir = sys.argv[2]

    # Check input directory
    if not os.path.isdir(input_dir):

        print(
            f"ERROR: Input directory not found: "
            f"{input_dir}"
        )

        sys.exit(1)

    # Create output directory
    os.makedirs(
        output_dir,
        exist_ok=True
    )

    # Select device
    if torch.cuda.is_available():

        device = torch.device("cuda")

        print("CUDA GPU detected.")
        print("Using NVIDIA GPU.")

    else:

        device = torch.device("cpu")

        print("CUDA GPU not detected.")
        print("Using CPU.")

    # Load model
    try:

        model = load_model(device)

    except Exception as error:

        print("\nERROR while loading model:")

        print(error)

        sys.exit(1)

    # Find NPY files
    input_files = sorted(
        [
            filename
            for filename in os.listdir(input_dir)
            if filename.lower().endswith(".npy")
        ]
    )

    if len(input_files) == 0:

        print(
            f"\nERROR: No .npy files found in "
            f"{input_dir}"
        )

        sys.exit(1)

    print(
        f"\nFound {len(input_files)} input files."
    )

    successful = 0

    failed = 0

    # =====================================================
    # PROCESS ALL FILES
    # =====================================================

    for index, filename in enumerate(
        input_files,
        start=1
    ):

        input_path = os.path.join(
            input_dir,
            filename
        )

        output_path = os.path.join(
            output_dir,
            filename
        )

        print(
            f"\n[{index}/{len(input_files)}] "
            f"Processing: {filename}"
        )

        try:

            # Load input
            input_array = np.load(
                input_path
            )

            print(
                f"Input shape: "
                f"{input_array.shape}"
            )

            # Restore
            restored = restore_image(
                model,
                input_array,
                device
            )

            # Save SAME filename
            np.save(
                output_path,
                restored
            )

            # Verify output
            saved = np.load(
                output_path
            )

            valid = (
                saved.shape ==
                (OUTPUT_SIZE, OUTPUT_SIZE)
                and
                np.isfinite(saved).all()
                and
                saved.min() >= 0.0
                and
                saved.max() <= 1.0
            )

            if not valid:

                raise ValueError(
                    "Output validation failed."
                )

            print(
                f"Output shape: "
                f"{saved.shape}"
            )

            print(
                f"Output range: "
                f"[{saved.min():.6f}, "
                f"{saved.max():.6f}]"
            )

            print(
                f"Saved: {output_path}"
            )

            successful += 1

        except Exception as error:

            print(
                f"FAILED: {filename}"
            )

            print(
                f"Reason: {error}"
            )

            failed += 1

    # =====================================================
    # FINAL REPORT
    # =====================================================

    print(
        "\n" + "=" * 60
    )

    print(
        "WaferVision Restoration Complete"
    )

    print(
        "=" * 60
    )

    print(
        f"Total files : {len(input_files)}"
    )

    print(
        f"Successful  : {successful}"
    )

    print(
        f"Failed      : {failed}"
    )

    print(
        "\nOutput directory:"
    )

    print(
        os.path.abspath(output_dir)
    )

    if failed == 0:

        print(
            "\nSUCCESS: All input files were restored."
        )

    else:

        print(
            "\nWARNING: Some files failed."
        )


# =========================================================
# ENTRY POINT
# =========================================================

if __name__ == "__main__":

    main()