from typing import Optional
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path

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
        # Load the image in grayscale
        # cv2.imread returns None if the image cannot be loaded
        img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)

        if img is None:
            print(f"Error: Could not load image from {image_path}. Check file format or corruption.")
            return None

        # Apply Canny edge detection
        # The Canny function takes a grayscale image and two thresholds.
        # Edges with intensity gradient more than max_val are sure edges.
        # Edges with intensity gradient less than min_val are sure non-edges.
        # Edges with intensity gradient between these two thresholds are classified
        # based on their connectivity. If they are connected to "sure-edge" pixels,
        # they are considered to be part of edges.
        edges = cv2.Canny(img, low_threshold, high_threshold)

        return edges

    except Exception as e:
        print(f"An error occurred during Canny edge detection: {e}")
        return None

def plot_canny_results(original_image_path: Path, edge_map: np.ndarray, low_threshold: int, high_threshold: int) -> None:
    """
    Plots the original image and its Canny edge detection result.

    Args:
        original_image_path: Path to the original image (for display purposes).
        edge_map: The binary image containing the detected edges.
        low_threshold: The low threshold used for Canny.
        high_threshold: The high threshold used for Canny.
    """
    if edge_map is None:
        print("No edge map to plot.")
        return

    try:
        # Load the original image again for plotting (can be color or grayscale)
        original_img = cv2.imread(str(original_image_path), cv2.IMREAD_COLOR)
        if original_img is None:
            print(f"Warning: Could not load original image for plotting from {original_image_path}. Plotting grayscale.")
            original_img = cv2.imread(str(original_image_path), cv2.IMREAD_GRAYSCALE)
            if original_img is None:
                print("Error: Cannot load any version of the original image for plotting.")
                return

        # Convert original image to RGB for consistent display with matplotlib
        if len(original_img.shape) == 3: # If it's a color image (BGR by default in OpenCV)
            original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)

        fig, axes = plt.subplots(1, 2, figsize=(12, 6))

        # Plot original image
        ax1 = axes[0]
        ax1.imshow(original_img, cmap='gray' if len(original_img.shape) == 2 else None)
        ax1.set_title('Original Image')
        ax1.axis('off')

        # Plot Canny edge map
        ax2 = axes[1]
        ax2.imshow(edge_map, cmap='gray') # Canny output is a single-channel binary image
        ax2.set_title(f'Canny Edges (Low:{low_threshold}, High:{high_threshold})')
        ax2.axis('off')

        plt.tight_layout()
        plt.show()

    except Exception as e:
        print(f"An error occurred during plotting: {e}")

if __name__ == "__main__":
    # Define the project root based on the script's location
    PROJECT_ROOT = Path(__file__).resolve().parent.parent # Assuming script is in scripts/edge_detection

    # Define the path to your image
    # You might need to change this if your image is in a different location.
    # For example, if your image is in a folder named 'images' at the project root:
    image_filename = "spiral.png" # Replace with your image filename
    image_path = PROJECT_ROOT / "images" / image_filename

    # --- Create a dummy image if the specified image file does not exist ---
    if not image_path.exists():
        print(f"Image '{image_filename}' not found at '{image_path}'. Creating a dummy image for demonstration.")
        dummy_img_data = np.zeros((256, 256), dtype=np.uint8)
        # Draw a white square
        cv2.rectangle(dummy_img_data, (50, 50), (200, 200), 255, -1)
        # Draw a white circle
        cv2.circle(dummy_img_data, (128, 128), 60, 0, -1) # Black circle in white square
        # Draw a diagonal line
        cv2.line(dummy_img_data, (0, 0), (255, 255), 255, 3)

        os.makedirs(image_path.parent, exist_ok=True)
        cv2.imwrite(str(image_path), dummy_img_data)
        print(f"Dummy image created at: {image_path}")
    # --- End of dummy image creation ---


    # Canny threshold parameters
    # These values often need to be tuned for different images.
    # A common rule of thumb is to set high_threshold to 2-3 times the low_threshold.
    canny_low_threshold = 50
    canny_high_threshold = 150

    print(f"Performing Canny edge detection on: {image_path}")
    print(f"Canny thresholds: Low={canny_low_threshold}, High={canny_high_threshold}")

    # Perform Canny edge detection
    canny_edges = perform_canny_edge_detection(image_path, canny_low_threshold, canny_high_threshold)

    if canny_edges is not None:
        # Plot the results
        plot_canny_results(image_path, canny_edges, canny_low_threshold, canny_high_threshold)
        print("\nCanny edge detection and plotting complete.")
    else:
        print("\nCanny edge detection failed. No plot generated.")

