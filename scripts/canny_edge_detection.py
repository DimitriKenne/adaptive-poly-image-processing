from typing import Optional, Dict, Union, List
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path

# --- Global Matplotlib Plotting Parameters for High Quality Output ---
MATPLOTLIB_PARAMS: Dict[str, Union[str, int, float, bool, List[str]]] = {
    # "text.usetex": True, # Uncomment if you have LaTeX installed
    # "font.family": "serif",
    # "font.serif": ["Computer Modern Roman"],

    "font.family": "serif",
    "font.serif": ["DejaVu Serif"],

    "font.size": 10,
    "axes.labelsize": 10,
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "figure.titlesize": 12,

    "lines.linewidth": 1.5,
    "lines.markersize": 6,

    "figure.autolayout": True,
    "savefig.dpi": 600, # High DPI for raster images
    "savefig.format": "png", # PNG is often fine for quick tests, but PDF is better for papers
    "figure.figsize": (12, 6), # Adjusted for 1x2 subplots

    "axes.grid": False, # Canny plots typically don't need grids
    "grid.linestyle": ':',
    "grid.alpha": 0.6,
}

# Specific DPI for raster images (PNG) when saving
PLOT_DPI = 600


def perform_canny_edge_detection(image_path: Path, low_threshold: int = 100, high_threshold: int = 200) -> Optional[np.ndarray]:
    """
    Performs Canny edge detection on a grayscale image.

    Args:
        image_path: Path to the input image file.
        low_threshold: The first threshold for the hysteresis procedure.
        high_threshold: The second threshold for the hysteresis procedure.

    Returns:
        A NumPy array representing the Canny edge map (binary image),
        or None if the image cannot be loaded.
    """
    if not image_path.exists():
        print(f"Error: Image not found at {image_path}. Please ensure the path is correct.")
        return None

    try:
        img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)

        if img is None:
            print(f"Error: Could not load image from {image_path}. Check file format or corruption.")
            return None

        edges = cv2.Canny(img, low_threshold, high_threshold)

        return edges

    except Exception as e:
        print(f"An error occurred during Canny edge detection: {e}")
        return None

def plot_canny_results(original_image_path: Path, edge_map: np.ndarray, low_threshold: int, high_threshold: int, save_path: Optional[Path] = None) -> None:
    """
    Plots the original image and its Canny edge detection result, and optionally saves it.

    Args:
        original_image_path: Path to the original image (for display purposes).
        edge_map: The binary image containing the detected edges.
        low_threshold: The low threshold used for Canny.
        high_threshold: The high threshold used for Canny.
        save_path: Optional Path object to save the plot. If None, plot is only shown.
    """
    if edge_map is None:
        print("No edge map to plot.")
        return

    try:
        original_img = cv2.imread(str(original_image_path), cv2.IMREAD_COLOR)
        if original_img is None:
            print(f"Warning: Could not load original image for plotting from {original_image_path}. Plotting grayscale.")
            original_img = cv2.imread(str(original_image_path), cv2.IMREAD_GRAYSCALE)
            if original_img is None:
                print("Error: Cannot load any version of the original image for plotting.")
                return

        if len(original_img.shape) == 3:
            original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)

        fig, axes = plt.subplots(1, 2, figsize=(MATPLOTLIB_PARAMS["figure.figsize"][0], MATPLOTLIB_PARAMS["figure.figsize"][1]))

        ax1 = axes[0]
        ax1.imshow(original_img, cmap='gray' if len(original_img.shape) == 2 else None)
        ax1.set_title('Original Image')
        ax1.axis('off')

        ax2 = axes[1]
        ax2.imshow(edge_map, cmap='gray')
        ax2.set_title(f'Canny Edges (Low:{low_threshold}, High:{high_threshold})')
        ax2.axis('off')

        plt.tight_layout()

        if save_path:
            try:
                plt.savefig(save_path, dpi=PLOT_DPI, format=MATPLOTLIB_PARAMS['savefig.format'])
                print(f"Saved Canny results plot to {save_path}")
            except Exception as e:
                print(f"Error saving Canny results plot to {save_path}: {e}")
        else:
            plt.show() # Only show if not saving to file

        plt.close(fig) # Close the figure to free up memory

    except Exception as e:
        print(f"An error occurred during plotting: {e}")

if __name__ == "__main__":
    # Apply global Matplotlib settings
    plt.rcParams.update(MATPLOTLIB_PARAMS)

    PROJECT_ROOT = Path(__file__).resolve().parent.parent

    image_filename = "spiral.png"
    image_path = PROJECT_ROOT / "images" / image_filename

    if not image_path.exists():
        print(f"Image '{image_filename}' not found at '{image_path}'. Creating a dummy image for demonstration.")
        dummy_img_data = np.zeros((256, 256), dtype=np.uint8)
        cv2.rectangle(dummy_img_data, (50, 50), (200, 200), 255, -1)
        cv2.circle(dummy_img_data, (128, 128), 60, 0, -1)
        cv2.line(dummy_img_data, (0, 0), (255, 255), 255, 3)

        os.makedirs(image_path.parent, exist_ok=True)
        cv2.imwrite(str(image_path), dummy_img_data)
        print(f"Dummy image created at: {image_path}")

    canny_low_threshold = 50
    canny_high_threshold = 150

    print(f"Performing Canny edge detection on: {image_path}")
    print(f"Canny thresholds: Low={canny_low_threshold}, High={canny_high_threshold}")

    canny_edges = perform_canny_edge_detection(image_path, canny_low_threshold, canny_high_threshold)

    if canny_edges is not None:
        # Define save path for Canny results
        canny_results_dir = PROJECT_ROOT / "results" / "canny_edge_detection_tests"
        os.makedirs(canny_results_dir, exist_ok=True)
        canny_plot_filename = f"canny_edges_{image_filename.replace('.png', '')}_low{canny_low_threshold}_high{canny_high_threshold}.{MATPLOTLIB_PARAMS['savefig.format']}"
        canny_plot_filepath = canny_results_dir / canny_plot_filename

        plot_canny_results(image_path, canny_edges, canny_low_threshold, canny_high_threshold, save_path=canny_plot_filepath)
        print("\nCanny edge detection and plotting complete.")
    else:
        print("\nCanny edge detection failed. No plot generated.")
