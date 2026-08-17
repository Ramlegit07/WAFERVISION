import streamlit as st
import torch
import numpy as np
import time
import math
import matplotlib.pyplot as plt

from model import WaferRestorationNet


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="WaferVision",
    page_icon="🔬",
    layout="wide"
)


# ============================================================
# SETTINGS
# ============================================================

MODEL_PATH = "checkpoints/wafer_restoration.pth"

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():

    model = WaferRestorationNet().to(DEVICE)

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE
    )

    if (
        isinstance(checkpoint, dict)
        and "state_dict" in checkpoint
    ):
        model.load_state_dict(
            checkpoint["state_dict"]
        )
    else:
        model.load_state_dict(
            checkpoint
        )

    model.eval()

    return model


try:

    model = load_model()

except Exception as e:

    st.error("❌ Model loading failed.")
    st.exception(e)
    st.stop()


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def normalize_image(image):

    image = image.astype(
        np.float32
    )

    minimum = image.min()
    maximum = image.max()

    return np.clip(
        (image - minimum)
        /
        (maximum - minimum + 1e-8),
        0,
        1
    )


def false_color(image):

    """
    False-color visualization only.
    Does not modify the actual restoration data.
    """

    image = normalize_image(
        image
    )

    r = np.clip(
        1.5 * image,
        0,
        1
    )

    g = np.clip(
        1.5
        *
        (1 - np.abs(image - 0.5) * 2),
        0,
        1
    )

    b = np.clip(
        1.5 * (1 - image),
        0,
        1
    )

    rgb = np.stack(
        [r, g, b],
        axis=-1
    )

    return np.clip(
        rgb,
        0,
        1
    )


def calculate_psnr(
    restored,
    ground_truth
):

    mse = np.mean(
        (restored - ground_truth) ** 2
    )

    if mse <= 1e-12:

        return float("inf")

    return (
        10 *
        np.log10(
            1.0 / mse
        )
    )


def calculate_ssim(
    restored,
    ground_truth
):

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    mu_x = restored.mean()
    mu_y = ground_truth.mean()

    sigma_x = restored.var()
    sigma_y = ground_truth.var()

    sigma_xy = np.mean(
        (restored - mu_x)
        *
        (ground_truth - mu_y)
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

    return float(
        numerator /
        (denominator + 1e-8)
    )


# ============================================================
# GRAPH 1 - GAUSSIAN NOISE
# ============================================================

def gaussian_noise_graph():

    sigmas = [
        0.05,
        0.16,
        0.28,
        0.39,
        0.50
    ]

    x = np.linspace(
        -1.5,
        1.5,
        600
    )

    fig, ax = plt.subplots(
        figsize=(9, 4)
    )

    for sigma in sigmas:

        y = (
            1 /
            (
                sigma *
                np.sqrt(2 * np.pi)
            )
        ) * np.exp(
            -(x ** 2)
            /
            (2 * sigma ** 2)
        )

        ax.plot(
            x,
            y,
            label=f"σ = {sigma:.2f}"
        )

    ax.set_title(
        "Gaussian Noise Probability Distributions"
    )

    ax.set_xlabel(
        "Noise value"
    )

    ax.set_ylabel(
        "Probability density"
    )

    ax.grid(
        alpha=0.25
    )

    ax.legend(
        ncol=3
    )

    fig.tight_layout()

    return fig


# ============================================================
# GRAPH 2 - SPECKLE NOISE
# ============================================================

def speckle_noise_graph():

    sigmas = [
        0.05,
        0.16,
        0.28,
        0.39,
        0.50
    ]

    x = np.linspace(
        0.001,
        3.0,
        600
    )

    fig, ax = plt.subplots(
        figsize=(9, 4)
    )

    for sigma in sigmas:

        # Log-normal distribution used only
        # as a visual representation of
        # multiplicative speckle characteristics.

        sigma_log = sigma

        mu_log = -(
            sigma_log ** 2
        ) / 2

        pdf = (
            1 /
            (
                x *
                sigma_log *
                np.sqrt(2 * np.pi)
            )
        ) * np.exp(
            -(
                np.log(x)
                - mu_log
            ) ** 2
            /
            (
                2 *
                sigma_log ** 2
            )
        )

        ax.plot(
            x,
            pdf,
            label=f"σ = {sigma:.2f}"
        )

    ax.set_title(
        "Representative Speckle Noise Distributions"
    )

    ax.set_xlabel(
        "Speckle intensity"
    )

    ax.set_ylabel(
        "Probability density"
    )

    ax.grid(
        alpha=0.25
    )

    ax.legend(
        ncol=3
    )

    fig.tight_layout()

    return fig


# ============================================================
# GRAPH 3 - GAUSSIAN BLUR KERNEL
# ============================================================

def blur_kernel_graph():

    sigma = 2.0

    size = 31

    axis = np.arange(
        -(size // 2),
        size // 2 + 1
    )

    xx, yy = np.meshgrid(
        axis,
        axis
    )

    kernel = np.exp(
        -(
            xx ** 2 +
            yy ** 2
        )
        /
        (
            2 * sigma ** 2
        )
    )

    kernel /= kernel.sum()

    fig, ax = plt.subplots(
        figsize=(6, 5)
    )

    image = ax.imshow(
        kernel,
        cmap="viridis"
    )

    ax.set_title(
        "2D Gaussian Blur Kernel (σ = 2.0)"
    )

    ax.set_xlabel(
        "Pixel offset (x)"
    )

    ax.set_ylabel(
        "Pixel offset (y)"
    )

    fig.colorbar(
        image,
        ax=ax,
        label="Kernel weight"
    )

    fig.tight_layout()

    return fig


# ============================================================
# GRAPH 4 - RESOLUTION DEGRADATION
# ============================================================

def resolution_graph():

    factors = [
        1.5,
        1.75,
        2.0,
        3.0,
        4.0
    ]

    labels = [
        "1.5×",
        "1.75×",
        "2×",
        "3×",
        "4×"
    ]

    fig, ax = plt.subplots(
        figsize=(8, 4)
    )

    bars = ax.bar(
        labels,
        factors
    )

    ax.set_title(
        "Resolution Degradation / Downsampling"
    )

    ax.set_xlabel(
        "Downsampling factor"
    )

    ax.set_ylabel(
        "Factor"
    )

    ax.grid(
        axis="y",
        alpha=0.25
    )

    for bar, value in zip(
        bars,
        factors
    ):

        ax.text(
            bar.get_x()
            +
            bar.get_width() / 2,
            value + 0.05,
            f"{value:.2f}×",
            ha="center"
        )

    fig.tight_layout()

    return fig


# ============================================================
# HEADER
# ============================================================

st.title(
    "🔬 WaferVision"
)

st.subheader(
    "AI-Based Restoration of Degraded Semiconductor Inspection Images"
)

st.write(
    "Deep-learning based restoration of degraded "
    "wafer inspection images affected by noise, "
    "blur and reduced spatial resolution."
)

st.divider()


# ============================================================
# SYSTEM INFORMATION
# ============================================================

col1, col2, col3, col4 = st.columns(4)

with col1:

    st.info(
        "🧠 MODEL\n\n"
        "WaferRestorationNet"
    )

with col2:

    st.info(
        "🖥️ DEVICE\n\n"
        + str(DEVICE).upper()
    )

with col3:

    st.info(
        "📥 INPUT\n\n"
        "128 × 128"
    )

with col4:

    parameter_count = sum(
        p.numel()
        for p in model.parameters()
    )

    st.info(
        "⚙️ PARAMETERS\n\n"
        f"{parameter_count:,}"
    )


st.divider()


# ============================================================
# DEGRADATION ANALYSIS
# ============================================================

st.header(
    "1. Degradation Analysis"
)

st.write(
    "The KLA problem involves restoration of degraded "
    "semiconductor inspection images. The following "
    "visualizations illustrate representative degradation "
    "characteristics."
)


# ============================================================
# GAUSSIAN + SPECKLE
# ============================================================

col1, col2 = st.columns(2)

with col1:

    st.subheader(
        "Gaussian Noise"
    )

    fig = gaussian_noise_graph()

    st.pyplot(
        fig,
        use_container_width=True
    )

    plt.close(fig)


with col2:

    st.subheader(
        "Speckle Noise"
    )

    fig = speckle_noise_graph()

    st.pyplot(
        fig,
        use_container_width=True
    )

    plt.close(fig)


# ============================================================
# BLUR + RESOLUTION
# ============================================================

col1, col2 = st.columns(2)

with col1:

    st.subheader(
        "Blur Kernel"
    )

    fig = blur_kernel_graph()

    st.pyplot(
        fig,
        use_container_width=True
    )

    plt.close(fig)


with col2:

    st.subheader(
        "Resolution Degradation"
    )

    fig = resolution_graph()

    st.pyplot(
        fig,
        use_container_width=True
    )

    plt.close(fig)


st.divider()


# ============================================================
# IMAGE SELECTION
# ============================================================

st.header(
    "2. AI Wafer Image Restoration"
)

image_number = st.number_input(
    "Enter image number",
    min_value=0,
    max_value=3199,
    value=0,
    step=1
)


image_path = (
    f"data/train/NoisyLR/"
    f"{image_number:06d}.npy"
)

gt_path = (
    f"data/train/GT/"
    f"{image_number:06d}.npy"
)


# ============================================================
# DISPLAY MODE
# ============================================================

display_mode = st.radio(
    "Image visualization",
    [
        "Grayscale",
        "False Color"
    ],
    horizontal=True
)


# ============================================================
# RESTORE
# ============================================================

if st.button(
    "🚀 Restore Wafer Image",
    use_container_width=True
):

    try:

        # ====================================================
        # LOAD DATA
        # ====================================================

        lr = np.load(
            image_path
        ).astype(
            np.float32
        )

        gt = np.load(
            gt_path
        ).astype(
            np.float32
        )


        # Remove unnecessary dimensions

        lr = np.squeeze(
            lr
        )

        gt = np.squeeze(
            gt
        )


        # ====================================================
        # INPUT TENSOR
        # ====================================================

        # IMPORTANT:
        # Raw LR data is passed to the model.
        # We do NOT normalize before inference.

        lr_tensor = (
            torch.from_numpy(
                lr
            )
            .unsqueeze(0)
            .unsqueeze(0)
            .to(DEVICE)
        )


        # ====================================================
        # INFERENCE
        # ====================================================

        if DEVICE.type == "cuda":

            torch.cuda.synchronize()


        start_time = time.perf_counter()


        with torch.no_grad():

            restored = model(
                lr_tensor
            )


        if DEVICE.type == "cuda":

            torch.cuda.synchronize()


        end_time = time.perf_counter()


        inference_time = (
            end_time -
            start_time
        )


        # ====================================================
        # OUTPUT
        # ====================================================

        restored = (
            restored
            .squeeze()
            .cpu()
            .numpy()
        )


        # ====================================================
        # NORMALIZED DISPLAY IMAGES
        # ====================================================

        lr_display = normalize_image(
            lr
        )

        restored_display = normalize_image(
            restored
        )

        gt_display = normalize_image(
            gt
        )


        # ====================================================
        # RESTORATION RESULT
        # ====================================================

        st.header(
            "3. Restoration Result"
        )


        col1, col2, col3 = st.columns(3)


        with col1:

            st.subheader(
                "Degraded Input"
            )

            if display_mode == "False Color":

                st.image(
                    false_color(lr),
                    use_container_width=True
                )

            else:

                st.image(
                    lr_display,
                    clamp=True,
                    use_container_width=True
                )


            st.caption(
                f"Input: {lr.shape}"
            )


        with col2:

            st.subheader(
                "AI Restored"
            )

            if display_mode == "False Color":

                st.image(
                    false_color(restored),
                    use_container_width=True
                )

            else:

                st.image(
                    restored_display,
                    clamp=True,
                    use_container_width=True
                )


            st.caption(
                f"Output: {restored.shape}"
            )


        with col3:

            st.subheader(
                "Ground Truth"
            )

            if display_mode == "False Color":

                st.image(
                    false_color(gt),
                    use_container_width=True
                )

            else:

                st.image(
                    gt_display,
                    clamp=True,
                    use_container_width=True
                )


            st.caption(
                f"Ground Truth: {gt.shape}"
            )


        st.divider()


        # ====================================================
        # METRICS
        # ====================================================

        # Metrics are calculated on normalized displayed
        # image values, matching your existing app behavior.

        psnr = calculate_psnr(
            restored_display,
            gt_display
        )

        ssim = calculate_ssim(
            restored_display,
            gt_display
        )

        inference_ms = (
            inference_time * 1000
        )

        fps = (
            1.0 /
            inference_time
        )


        # ====================================================
        # METRIC CARDS
        # ====================================================

        st.header(
            "4. Image Quality & Performance"
        )


        metric1, metric2, metric3, metric4 = (
            st.columns(4)
        )


        with metric1:

            st.metric(
                "PSNR",
                f"{psnr:.4f} dB"
            )


        with metric2:

            st.metric(
                "SSIM",
                f"{ssim:.4f}"
            )


        with metric3:

            st.metric(
                "Inference Time",
                f"{inference_ms:.2f} ms"
            )


        with metric4:

            st.metric(
                "Approx. FPS",
                f"{fps:.2f}"
            )


        # ====================================================
        # PERFORMANCE GRAPH
        # ====================================================

        st.subheader(
            "Performance Visualization"
        )


        col1, col2 = st.columns(2)


        # ----------------------------------------------------
        # QUALITY GRAPH
        # ----------------------------------------------------

        with col1:

            fig, ax = plt.subplots(
                figsize=(7, 4)
            )

            # PSNR is normalized ONLY for putting it beside SSIM
            # on the same 0-1 graph.

            psnr_normalized = min(
                psnr / 40.0,
                1.0
            )

            labels = [
                "PSNR\n(normalized)",
                "SSIM"
            ]

            values = [
                psnr_normalized,
                ssim
            ]

            bars = ax.bar(
                labels,
                values
            )

            ax.set_ylim(
                0,
                1.1
            )

            ax.set_ylabel(
                "Quality score"
            )

            ax.set_title(
                "Image Quality"
            )

            ax.grid(
                axis="y",
                alpha=0.25
            )

            ax.text(
                0,
                psnr_normalized + 0.03,
                f"{psnr:.2f} dB",
                ha="center"
            )

            ax.text(
                1,
                ssim + 0.03,
                f"{ssim:.4f}",
                ha="center"
            )

            fig.tight_layout()

            st.pyplot(
                fig,
                use_container_width=True
            )

            plt.close(fig)


        # ----------------------------------------------------
        # SPEED GRAPH
        # ----------------------------------------------------

        with col2:

            fig, ax = plt.subplots(
                figsize=(7, 4)
            )

            labels = [
                "Inference\n(ms)",
                "FPS"
            ]

            values = [
                inference_ms,
                fps
            ]

            bars = ax.bar(
                labels,
                values
            )

            ax.set_title(
                "Inference Performance"
            )

            ax.set_ylabel(
                "Measured value"
            )

            ax.grid(
                axis="y",
                alpha=0.25
            )

            for bar, value in zip(
                bars,
                values
            ):

                ax.text(
                    bar.get_x()
                    +
                    bar.get_width() / 2,
                    value,
                    f"{value:.2f}",
                    ha="center",
                    va="bottom"
                )

            fig.tight_layout()

            st.pyplot(
                fig,
                use_container_width=True
            )

            plt.close(fig)


        # ====================================================
        # PIXEL DISTRIBUTION
        # ====================================================

        st.subheader(
            "Pixel Intensity Distribution"
        )

        fig, ax = plt.subplots(
            figsize=(10, 4)
        )


        ax.hist(
            lr_display.flatten(),
            bins=50,
            density=True,
            alpha=0.45,
            label="Degraded Input"
        )


        ax.hist(
            restored_display.flatten(),
            bins=50,
            density=True,
            alpha=0.45,
            label="AI Restored"
        )


        ax.hist(
            gt_display.flatten(),
            bins=50,
            density=True,
            alpha=0.45,
            label="Ground Truth"
        )


        ax.set_title(
            "Pixel Intensity Distribution Comparison"
        )

        ax.set_xlabel(
            "Normalized intensity"
        )

        ax.set_ylabel(
            "Probability density"
        )

        ax.legend()

        ax.grid(
            alpha=0.25
        )

        fig.tight_layout()


        st.pyplot(
            fig,
            use_container_width=True
        )

        plt.close(fig)


        # ====================================================
        # IMAGE PROFILE GRAPH
        # ====================================================

        st.subheader(
            "Image Detail Profile"
        )


        input_row = (
            lr_display.shape[0] // 2
        )

        restored_row = (
            restored_display.shape[0] // 2
        )

        gt_row = (
            gt_display.shape[0] // 2
        )


        input_profile = (
            lr_display[input_row]
        )

        restored_profile = (
            restored_display[restored_row]
        )

        gt_profile = (
            gt_display[gt_row]
        )


        fig, ax = plt.subplots(
            figsize=(10, 4)
        )


        ax.plot(
            np.linspace(
                0,
                1,
                len(input_profile)
            ),
            input_profile,
            label="Degraded Input",
            linewidth=1.5
        )


        ax.plot(
            np.linspace(
                0,
                1,
                len(restored_profile)
            ),
            restored_profile,
            label="AI Restored",
            linewidth=1.5
        )


        ax.plot(
            np.linspace(
                0,
                1,
                len(gt_profile)
            ),
            gt_profile,
            label="Ground Truth",
            linewidth=1.5
        )


        ax.set_title(
            "Central Row Intensity Profile"
        )

        ax.set_xlabel(
            "Normalized horizontal position"
        )

        ax.set_ylabel(
            "Pixel intensity"
        )

        ax.legend()

        ax.grid(
            alpha=0.25
        )

        fig.tight_layout()


        st.pyplot(
            fig,
            use_container_width=True
        )

        plt.close(fig)


        # ====================================================
        # TECHNICAL DETAILS
        # ====================================================

        st.success(
            "✅ Wafer image restoration completed successfully!"
        )


        with st.expander(
            "📊 Technical Details"
        ):

            st.write(
                "### Input"
            )

            st.write(
                f"Shape: {lr.shape}"
            )

            st.write(
                f"Minimum: {lr.min():.6f}"
            )

            st.write(
                f"Maximum: {lr.max():.6f}"
            )


            st.write(
                "### AI Restored"
            )

            st.write(
                f"Shape: {restored.shape}"
            )

            st.write(
                f"Minimum: {restored.min():.6f}"
            )

            st.write(
                f"Maximum: {restored.max():.6f}"
            )


            st.write(
                "### Ground Truth"
            )

            st.write(
                f"Shape: {gt.shape}"
            )


            st.write(
                "### Model"
            )

            st.write(
                "WaferRestorationNet"
            )


            st.write(
                "### Parameters"
            )

            st.write(
                f"{parameter_count:,}"
            )


            st.write(
                "### Device"
            )

            st.write(
                str(DEVICE)
            )


            st.write(
                "### Inference Time"
            )

            st.write(
                f"{inference_ms:.2f} ms"
            )


            st.write(
                "### Approximate FPS"
            )

            st.write(
                f"{fps:.2f}"
            )


    except FileNotFoundError:

        st.error(
            "❌ Image file not found."
        )

        st.write(
            f"Looking for:\n"
            f"{image_path}"
        )


    except Exception as e:

        st.error(
            "❌ An error occurred."
        )

        st.exception(e)