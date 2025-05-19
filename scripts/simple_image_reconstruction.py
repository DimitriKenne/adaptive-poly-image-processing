import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image # Used for loading/saving images
import os # Import os for path joining
import time # To measure processing time
from typing import Dict, List, Tuple, Union, Optional # Keep necessary typing hints

# Add project root to Python path
# This allows importing modules from the project's root directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Define the base path for the results folder
BASE_RESULTS_DIR = PROJECT_ROOT / "results"

# Define the subfolder for this specific script's results
# We will add an image-specific subfolder inside this
SCRIPT_RESULTS_BASE_DIR = BASE_RESULTS_DIR / "simple_image_reconstruction_tests"


# Assuming your project structure is something like:
# your_project/
# ├── poly_approx/
# │   ├── __init__.py  # This empty file is required to make 'poly_approx' a Python package
# │   ├── interpolation_nodes.py
# │   ├── polynomial_bases.py
# │   ├── poly_projector.py
# │   ├── image_poly_approximation.py # Your updated file
# │   ├── admissible_meshes.py
# │   └── edge_processing.py # Needed for normalize_error_image
# └── scripts/
#     └── simple_image_reconstruction.py # This script (renamed)

# Flag to check if dummy functions are being used
using_dummy_functions = False

try:
    # Import the segment approximation function and save_images
    from poly_approx.image_poly_approximation import image_poly_approximation_segment, save_images
    # Import the error normalization function (still useful for error visualization)
    from poly_approx.edge_processing import normalize_error_image

    # Removed imports for compute_admissible_mesh, extremal_points, gen_vanderm2d
    # as they are not directly called in this script anymore.

except ImportError as e:
    print(f"Could not import modules from poly_approx: {e}")
    print("Please ensure your project structure is correct or adjust the import paths.")
    print("Using dummy functions. Testing and plotting will be skipped.")
    using_dummy_functions = True
    # Provide dummy functions if imports fail
    def image_poly_approximation_segment(image_segment, rectangle, poly_degree, nodes_method, admissible_mesh_type, m_cheb, poly_basis):
        print("Dummy image_poly_approximation_segment called.")
        # Return a dummy result dictionary with empty arrays for raw errors and nodes
        height, width = image_segment.shape
        return {
            'original_segment': image_segment,
            'smoothed_segment': np.zeros_like(image_segment),
            'approx_original': np.zeros_like(image_segment),
            'approx_smoothed': np.zeros_like(image_segment),
            'error_original': np.zeros_like(image_segment), # Return raw errors
            'error_smoothed': np.zeros_like(image_segment), # Return raw errors
            'diff_original_poly_smoothed': np.zeros_like(image_segment), # Return raw errors
            'diff_smoothed_poly_original': np.zeros_like(image_segment), # Return raw errors
            'computation_time': 0.0,
            'nodes': np.array([]) # Return empty array for nodes
        }
    def save_images(image_dict, save_folder):
        print("Dummy save_images called.")
    def normalize_error_image(error_map, epsilon=1e-8):
         print("Dummy normalize_error_image called.")
         if error_map.size == 0:
             return np.zeros_like(error_map, dtype=np.float32)
         min_val = np.min(error_map)
         max_val = np.max(error_map)
         if max_val - min_val < epsilon:
             return np.zeros_like(error_map, dtype=np.float32)
         return ((error_map - min_val) / (max_val - min_val + epsilon)).astype(np.float32)


def load_and_preprocess_image(image_path):
    """Loads a grayscale image and normalizes it to [0, 1]."""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    print(f"Loading image: {image_path}")
    # Open and convert to grayscale ('L') and normalize to [0, 1]
    # Use float32 for consistency with approximation and error maps
    img = np.array(Image.open(image_path).convert('L'), dtype=np.float32) / 255.0
    return img

def segment_image_with_overlap(
    image: np.ndarray,
    segments_x: int,
    segments_y: int,
    overlap_percentage: float = 0.1 # Default overlap of 10%
) -> List[Dict[str, Union[np.ndarray, Tuple[float, float, float, float], Tuple[int, int, int, int], Tuple[int, int, int, int]]]]:
    """
    Divides an image into a grid of segments with overlap.

    Returns a list of dictionaries, each containing:
    - 'segment_data_processed_region': The numpy array for the segment's pixel data
                                       including the overlap (region used for poly approx).
    - 'rectangle_processed_region': The bounds [xmin, ymin, xmax, ymax] of the processed region
                                    relative to the original image scaled to [0,1]x[0,1].
    - 'pixel_coords_processed_region': The pixel coordinates (x_start, y_start, x_end, y_end)
                                       of the processed region in the original image.
    - 'pixel_coords_core_region': The pixel coordinates (x_start, y_start, x_end, y_end)
                                  of the non-overlapping core region in the original image.
    """
    height, width = image.shape
    segment_height_no_overlap = height // segments_y
    segment_width_no_overlap = width // segments_x

    segments = []
    for i in range(segments_y):
        for j in range(segments_x):
            # Define the core (non-overlapping) pixel boundaries for the segment
            core_y_start = i * segment_height_no_overlap
            core_y_end = core_y_start + segment_height_no_overlap
            core_x_start = j * segment_width_no_overlap
            core_x_end = core_x_start + segment_width_no_overlap

            # Adjust the last segment in each dimension to cover the remaining pixels
            if i == segments_y - 1:
                core_y_end = height
            if j == segments_x - 1:
                core_x_end = width

            # Calculate overlap in pixels for this segment's core size
            current_segment_height = core_y_end - core_y_start
            current_segment_width = core_x_end - core_x_start

            overlap_pixels_y = int(current_segment_height * overlap_percentage)
            overlap_pixels_x = int(current_segment_width * overlap_percentage)

            # Define the processed (overlapping) pixel boundaries
            processed_y_start = max(0, core_y_start - overlap_pixels_y)
            processed_y_end = min(height, core_y_end + overlap_pixels_y)
            processed_x_start = max(0, core_x_start - overlap_pixels_x)
            processed_x_end = min(width, core_x_end + overlap_pixels_x)

            # Extract the segment data for the processed region
            segment_data_processed_region = image[processed_y_start:processed_y_end, processed_x_start:processed_x_end]

            # Define the spatial region of the processed segment relative to the whole image [0,1]x[0,1]
            rectangle_processed_region = (
                processed_x_start / width,
                processed_y_start / height,
                processed_x_end / width,
                processed_y_end / height
            )

            segments.append({
                'segment_data_processed_region': segment_data_processed_region,
                'rectangle_processed_region': rectangle_processed_region,
                'pixel_coords_processed_region': (processed_x_start, processed_y_start, processed_x_end, processed_y_end),
                'pixel_coords_core_region': (core_x_start, core_y_start, core_x_end, core_y_end) # Store core region for blending
            })
    return segments


def reassemble_segments_with_blending(
    processed_segments: List[Dict[str, Union[np.ndarray, Tuple[int, int, int, int]]]],
    original_image_shape: Tuple[int, int],
    result_key: str, # The key of the result to reassemble and blend (e.g., 'approx_original')
    overlap_percentage: float = 0.1 # Overlap percentage used during segmentation
) -> np.ndarray:
    """
    Reassembles a single result (e.g., approximation image) from processed segments
    with blending in overlapping regions.

    Parameters:
    -----------
    processed_segments : List[Dict[str, Union[np.ndarray, Tuple[int, int, int, int]]]]
        List of dictionaries, each containing processed data and pixel coordinates
        for the processed (overlapping) and core (non-overlapping) regions.
    original_image_shape : Tuple[int, int]
        The shape of the original image (height, width).
    result_key : str
        The key of the result to reassemble and blend (e.g., 'approx_original').
        The data for this key is expected to be in the processed region's shape.
    overlap_percentage : float, optional
        The overlap percentage used during segmentation. Default is 0.1.

    Returns:
    --------
    np.ndarray
        The reassembled image with blending applied.
    """
    full_height, full_width = original_image_shape
    reassembled_image = np.zeros(original_image_shape, dtype=np.float32)
    # Keep track of the sum of weights for normalization
    weight_sum_map = np.zeros(original_image_shape, dtype=np.float32)

    # Create a list of segment info needed for blending
    segment_info_list = []
    for segment_result in processed_segments:
        if result_key in segment_result and 'pixel_coords_processed_region' in segment_result and 'pixel_coords_core_region' in segment_result:
            segment_info_list.append({
                'data': segment_result[result_key],
                'processed_coords': segment_result['pixel_coords_processed_region'],
                'core_coords': segment_result['pixel_coords_core_region']
            })
        else:
            print(f"Warning: Missing data or coordinates for result key '{result_key}' in a segment. Skipping blending for this segment.")


    # Iterate over each pixel in the final image
    for y in range(full_height):
        for x in range(full_width):
            pixel_value = 0.0
            total_weight = 0.0

            # Find all segments whose processed region covers the current pixel (x, y)
            covering_segments = []
            for segment_info in segment_info_list:
                px_start, py_start, px_end, py_end = segment_info['processed_coords']
                if px_start <= x < px_end and py_start <= y < py_end:
                    covering_segments.append(segment_info)

            # If the pixel is covered by any segment, calculate blended value
            if covering_segments:
                for segment_info in covering_segments:
                    data = segment_info['data']
                    px_start, py_start, px_end, py_end = segment_info['processed_coords']
                    core_x_start, core_y_start, core_x_end, core_y_end = segment_info['core_coords']

                    # Calculate pixel position relative to the processed segment's top-left corner
                    local_x = x - px_start
                    local_y = y - py_start

                    # Ensure local coordinates are within bounds of the segment data
                    if 0 <= local_x < data.shape[1] and 0 <= local_y < data.shape[0]:
                         segment_pixel_value = data[local_y, local_x]

                         # Calculate blending weight based on distance to the core region boundary
                         # Weight is 1 inside the core, and decreases linearly to 0 at the edge of the processed region
                         weight_x = 1.0
                         if x < core_x_start: # Left overlap region
                             # Distance from core_x_start, normalized by overlap width on the left
                             overlap_width_left = core_x_start - px_start
                             if overlap_width_left > 0:
                                 weight_x = (x - px_start) / overlap_width_left
                             else: # Should not happen with proper overlap calculation, but as safeguard
                                 weight_x = 1.0
                         elif x >= core_x_end: # Right overlap region
                             # Distance from core_x_end, normalized by overlap width on the right
                             overlap_width_right = px_end - core_x_end
                             if overlap_width_right > 0:
                                 weight_x = 1.0 - (x - core_x_end) / overlap_width_right
                             else: # Should not happen
                                 weight_x = 1.0

                         weight_y = 1.0
                         if y < core_y_start: # Top overlap region
                             # Distance from core_y_start, normalized by overlap height on the top
                             overlap_height_top = core_y_start - py_start
                             if overlap_height_top > 0:
                                 weight_y = (y - py_start) / overlap_height_top
                             else: # Should not happen
                                 weight_y = 1.0
                         elif y >= core_y_end: # Bottom overlap region
                             # Distance from core_y_end, normalized by overlap height on the bottom
                             overlap_height_bottom = py_end - core_y_end
                             if overlap_height_bottom > 0:
                                 weight_y = 1.0 - (y - core_y_end) / overlap_height_bottom
                             else: # Should not happen
                                 weight_y = 1.0

                         # Total weight is the product of x and y weights
                         blending_weight = weight_x * weight_y

                         pixel_value += segment_pixel_value * blending_weight
                         total_weight += blending_weight
                    else:
                         print(f"Warning: Local coordinates ({local_x}, {local_y}) out of bounds for segment data shape {data.shape}. Skipping pixel.")


                # Normalize the pixel value by the total weight
                if total_weight > 1e-6: # Avoid division by near zero
                    reassembled_image[y, x] = pixel_value / total_weight
                # If total_weight is zero (shouldn't happen if covering_segments is not empty), pixel remains 0.0

    return reassembled_image


if __name__ == "__main__":
    if using_dummy_functions:
        print("Skipping test due to import errors.")
        sys.exit(1) # Exit the script if imports failed

    # --- Configuration ---
    # Provide the path to a sample image
    # You'll need to replace this with a path to an actual image file
    # Example path assuming the image is in a subfolder named 'images' in the project root
    sample_image_path = PROJECT_ROOT / "images" / "spiral.png" # Corrected example path

    # Create a dummy image file if the sample doesn't exist for demonstration
    if not sample_image_path.exists():
        print(f"Sample image not found at {sample_image_path}. Creating a dummy image.")
        dummy_img = np.zeros((200, 200), dtype=np.uint8)
        # Add a white square
        dummy_img[50:150, 50:150] = 255
        # Ensure the 'images' directory exists if saving dummy there
        os.makedirs(PROJECT_ROOT / "images", exist_ok=True)
        Image.fromarray(dummy_img).save(sample_image_path)
        print(f"Dummy image created at {sample_image_path}")

    # Extract the base name of the image file for use in saved filenames and subfolder
    image_base_name = sample_image_path.stem

    # Define the image-specific results subfolder
    IMAGE_RESULTS_DIR = SCRIPT_RESULTS_BASE_DIR / image_base_name

    # Ensure the image-specific results directory exists
    os.makedirs(IMAGE_RESULTS_DIR, exist_ok=True)
    print(f"Saving results to: {IMAGE_RESULTS_DIR}")


    poly_degree = 3       # Degree of polynomial approximation per segment
    # Use 'leja' or 'fekete' which require an admissible mesh. 'padua' does not.
    nodes_method = 'leja' # Node selection method per segment ('full_mesh', 'leja', 'fekete', 'padua')
    # Default admissible mesh type (only relevant for 'leja' and 'fekete' if mesh is not explicitly provided)
    admissible_mesh_type = 'cheb' # Options: 'cheb', 'uni'
    m_cheb = 2 # Parameter 'm' for Chebyshev mesh construction (m > 1)

    segments_x = 20        # Number of segments horizontally
    segments_y = 20        # Number of segments vertically
    overlap_percentage = 0.1 # Percentage of segment size for overlap (0.0 to 0.5)


    # --- 1. Load and Preprocess the Image ---
    try:
        original_image = load_and_preprocess_image(sample_image_path)
        original_image_shape = original_image.shape
        height, width = original_image_shape # Get height and width for plotting grid
        print(f"Original image shape: {original_image_shape}")
    except FileNotFoundError as e:
        print(f"Error loading image: {e}")
        sys.exit(1)


    # --- 2. Segment the Image with Overlap ---
    print(f"\nSegmenting image into {segments_x}x{segments_y} grid with {overlap_percentage*100:.0f}% overlap...")
    segments_with_overlap = segment_image_with_overlap(original_image, segments_x, segments_y, overlap_percentage)
    print(f"Created {len(segments_with_overlap)} segments with overlap.")

    # --- 3. Process Each Segment ---
    print("\nProcessing each segment...")
    processed_segments_results = []
    total_computation_time = 0

    # Define the keys for the approximation and error maps that we want to process
    # For simple reconstruction, we primarily care about the approximation image ('approx_original' or 'approx_smoothed')
    # We'll also keep error_original for plotting the error heatmap.
    result_keys_to_process = [
        'approx_original', # The main reconstruction result
        'error_original' # For error heatmap visualization
    ]

    for i, segment_info in enumerate(segments_with_overlap):
        print(f"\n--- Processing Segment {i+1}/{len(segments_with_overlap)} ---")
        segment_data_processed_region = segment_info['segment_data_processed_region']
        rectangle_processed_region = segment_info['rectangle_processed_region'] # Get the rectangle for this segment's processed region
        pixel_coords_processed_region = segment_info['pixel_coords_processed_region']
        pixel_coords_core_region = segment_info['pixel_coords_core_region']


        # Call the segment approximation function on the processed region data
        segment_results = image_poly_approximation_segment(
            image_segment=segment_data_processed_region, # Pass the data from the processed region
            rectangle=rectangle_processed_region, # Pass the rectangle of the processed region
            poly_degree=poly_degree,
            nodes_method=nodes_method,
            admissible_mesh_type=admissible_mesh_type, # Pass mesh type for internal mesh generation
            m_cheb=m_cheb, # Pass m_cheb for internal mesh generation if needed
            poly_basis=1 # Assuming shifted monomials for now, can be configurable
        )

        if segment_results:
            # Store the relevant results for this segment, along with both sets of coordinates
            segment_processed_data = {key: segment_results[key] for key in result_keys_to_process if key in segment_results}
            segment_processed_data['pixel_coords_processed_region'] = pixel_coords_processed_region
            segment_processed_data['pixel_coords_core_region'] = pixel_coords_core_region
            processed_segments_results.append(segment_processed_data)

            total_computation_time += segment_results.get('computation_time', 0)
        else:
            print(f"Skipping segment {i+1} due to processing error.")

    print(f"\nFinished processing all segments. Total computation time: {total_computation_time:.4f} seconds.")

    # --- 4. Reassemble Results with Blending ---
    final_reassembled_approx_image = np.zeros(original_image_shape, dtype=np.float32)
    final_reassembled_error_image = np.zeros(original_image_shape, dtype=np.float32) # Reassemble error without blending for comparison

    if processed_segments_results:
        print("\nReassembling approximation image with blending...")
        # Reassemble the approximation image with blending
        final_reassembled_approx_image = reassemble_segments_with_blending(
            processed_segments_results,
            original_image_shape,
            result_key='approx_original', # Blend the 'approx_original' result
            overlap_percentage=overlap_percentage # Pass the overlap percentage
        )
        print("Reassembly with blending complete.")

        print("\nReassembling raw error_original map (without blending)...")
        # Reassemble the raw error_original map without blending for a standard comparison heatmap
        # We can use a simplified reassembly logic here or adapt reassemble_segments_with_blending
        # to handle no blending when overlap_percentage is 0 or a flag is set.
        # For simplicity now, let's do a basic reassembly for the error map.
        reassembled_error_map_raw = np.zeros(original_image_shape, dtype=np.float32)
        for segment_result in processed_segments_results:
             if 'error_original' in segment_result and 'pixel_coords_core_region' in segment_result:
                 # Use the core region coordinates for placing the error map data
                 # Note: The error map is calculated over the processed region, but for a simple
                 # non-blended reassembly, we'll place the portion corresponding to the core region.
                 # This might not be ideal, a better approach is to calculate error on the final blended image.
                 # Let's stick to calculating error on the final blended image for the plot.
                 pass # We will calculate the error on the final blended image later for the plot.

        # Calculate the actual error map between the original image and the final blended approximation
        actual_reconstruction_error_map = np.abs(original_image - final_reassembled_approx_image)
        print("Actual reconstruction error map calculated.")

        # Normalize the actual reconstruction error map for visualization
        normalized_reconstruction_error_for_plot = normalize_error_image(actual_reconstruction_error_map)
        print("Actual reconstruction error map normalized for plotting.")


        # --- 5. Save Reassembled Results (PNG only) ---
        print("\nSaving reassembled results...")

        # Create a subfolder for this specific image's results within the simple tests directory
        IMAGE_RESULTS_DIR = SCRIPT_RESULTS_BASE_DIR / image_base_name
        os.makedirs(IMAGE_RESULTS_DIR, exist_ok=True)
        print(f"Saving final reassembled results to: {IMAGE_RESULTS_DIR}")

        # Construct a filename suffix based on parameters including overlap
        params_suffix = (
            f"deg{poly_degree}_{nodes_method}_{segments_x}x{segments_y}_overlap{str(overlap_percentage).replace('.', 'p')}"
        )

        # Dictionary to hold images normalized for saving as PNG
        images_to_save_normalized_png = {}

        # Save the original image (already loaded and normalized)
        images_to_save_normalized_png[f"{image_base_name}_original_image.png"] = original_image

        # Save the final reassembled approximation image (after blending)
        images_to_save_normalized_png[f"{image_base_name}_approx_blended_{params_suffix}.png"] = final_reassembled_approx_image

        # Save the actual reconstruction error map (normalized for visualization)
        images_to_save_normalized_png[f"{image_base_name}_actual_error_heatmap_{params_suffix}.png"] = normalized_reconstruction_error_for_plot


        # Save the normalized images (PNG)
        save_images(images_to_save_normalized_png, str(IMAGE_RESULTS_DIR))

        print("All generated results saved.")


        # --- 6. Visualize Full Image Results ---
        print("\nGenerating plots with original, blended approximation, and actual error heatmap...")

        # Create the plot figure (1 row, 3 columns)
        fig, axes = plt.subplots(1, 3, figsize=(18, 6)) # Adjusted figure size for 3 plots

        # Plot 1: Original Image
        ax1 = axes[0]
        ax1.imshow(original_image, cmap='gray', vmin=0, vmax=1)
        ax1.set_title('Original Image')
        ax1.axis('off')

        # Plot 2: Reassembled Blended Approximation
        ax2 = axes[1]
        ax2.imshow(final_reassembled_approx_image, cmap='gray', vmin=0, vmax=1)
        ax2.set_title(f'Blended Approx (deg={poly_degree}, {segments_x}x{segments_y}, {overlap_percentage*100:.0f}% overlap)')
        ax2.axis('off')

        # Plot 3: Actual Reconstruction Error Heatmap (Normalized)
        ax3 = axes[2]
        im = ax3.imshow(normalized_reconstruction_error_for_plot, cmap='viridis', origin='upper')
        fig.colorbar(im, ax=ax3, label='Actual Reconstruction Error (Normalized)')
        ax3.set_title('Actual Reconstruction Error Heatmap')
        ax3.axis('off')


        plt.tight_layout()

        # --- Save the Plot ---
        # Filename reflects the blending and actual error
        plot_filename = f"{image_base_name}_reconstruction_plot_blended_actual_error_{params_suffix}.png"
        plot_filepath = os.path.join(str(IMAGE_RESULTS_DIR), plot_filename)

        try:
            plt.savefig(plot_filepath)
            print(f"\nSaved visualization plot to {plot_filepath}")
        except Exception as e:
            print(f"\nError saving visualization plot to {plot_filepath}: {e}")


        # Close the plot figure
        plt.close(fig)

    else:
        print("\nNo segments were successfully processed. Cannot reassemble, plot, or save.")

    print("\nSimple image reconstruction script finished.")
