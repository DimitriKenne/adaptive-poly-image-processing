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
    # Import the error normalization function
    from poly_approx.edge_processing import normalize_error_image

    # Removed imports for compute_admissible_mesh, extremal_points, gen_vanderm2d
    # as they are not directly called in this script anymore.

except ImportError as e:
    print(f"Could not import modules from poly_approx: {e}")
    print("Please ensure your project structure is correct or adjust the import paths.")
    print("Using dummy functions. Testing and plotting will be skipped.")
    using_dummy_functions = True
    # Provide dummy functions if imports fail
    def image_poly_approximation_segment(*args, **kwargs):
        print("Dummy image_poly_approximation_segment called.")
        # Return a dummy result dictionary with empty arrays for raw errors and nodes
        dummy_height, dummy_width = 100, 100 # Default dummy size
        return {
            'original_segment': np.zeros((dummy_height, dummy_width)),
            'smoothed_segment': np.zeros((dummy_height, dummy_width)),
            'approx_original': np.zeros((dummy_height, dummy_width)),
            'approx_smoothed': np.zeros((dummy_height, dummy_width)),
            'error_original': np.zeros((dummy_height, dummy_width)), # Return raw errors
            'error_smoothed': np.zeros((dummy_height, dummy_width)), # Return raw errors
            'diff_original_poly_smoothed': np.zeros((dummy_height, dummy_width)), # Return raw errors
            'diff_smoothed_poly_original': np.zeros((dummy_height, dummy_width)), # Return raw errors
            'computation_time': 0.0,
            'nodes': np.array([]) # Return empty array for nodes
        }
    def save_images(*args, **kwargs):
        print("Dummy save_images called.")
    def normalize_error_image(error_map, epsilon=1e-8):
         print("Dummy normalize_error_image called.")
         # Return a dummy normalized image (all zeros or uniform)
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
    # Open, convert to grayscale ('L'), and normalize to [0, 1]
    img = np.array(Image.open(image_path).convert('L'), dtype=np.float32) / 255.0
    return img

def segment_image(image: np.ndarray, segments_x: int, segments_y: int) -> List[Dict[str, Union[np.ndarray, Tuple[float, float, float, float], Tuple[int, int, int, int]]]]:
    """
    Divides an image into a grid of segments.

    Returns a list of dictionaries, each containing:
    - 'segment_data': The numpy array for the segment's pixel data.
    - 'rectangle': The bounds [xmin, ymin, xmax, ymax] of the segment relative
                   to the original image scaled to [0,1]x[0,1].
    - 'pixel_coords': The pixel coordinates (x_start, y_start, x_end, y_end)
                      of the segment in the original image.
    """
    height, width = image.shape
    segment_height = height // segments_y
    segment_width = width // segments_x

    segments = []
    for i in range(segments_y):
        for j in range(segments_x):
            # Define the pixel boundaries for the segment
            y_start = i * segment_height
            y_end = y_start + segment_height
            x_start = j * segment_width
            x_end = x_start + segment_width

            # Extract the segment
            segment = image[y_start:y_end, x_start:x_end]

            # Define the spatial region of this segment relative to the whole image [0,1]x[0,1]
            # This rectangle is passed to image_poly_approximation_segment.
            rectangle = (x_start / width, y_start / height, x_end / width, y_end / height)

            segments.append({
                'segment_data': segment,
                'rectangle': rectangle, # Use 'rectangle' key
                'pixel_coords': (x_start, y_start, x_end, y_end) # Store pixel coordinates for reassembly
            })
    return segments

def reassemble_segments(processed_segments: List[Dict[str, np.ndarray]], original_image_shape: Tuple[int, int], result_keys: List[str]) -> Dict[str, np.ndarray]:
    """Reassembles results for multiple keys from processed segments into full image arrays."""
    full_height, full_width = original_image_shape
    reassembled_results = {}

    for key in result_keys:
        # Initialize an empty array for the reassembled result for this key
        # Use float32 for consistency
        reassembled_image = np.zeros(original_image_shape, dtype=np.float32)

        for segment_result in processed_segments:
            # Get the processed data for the current key
            if key in segment_result:
                processed_data = segment_result[key]
                # Get the original pixel coordinates of this segment
                x_start, y_start, x_end, y_end = segment_result['pixel_coords']

                # Place the processed data into the correct location in the reassembled image
                # Ensure processed_data has the correct shape for the segment slice
                segment_height = y_end - y_start
                segment_width = x_end - x_start
                if processed_data.shape == (segment_height, segment_width):
                    reassembled_image[y_start:y_end, x_start:x_end] = processed_data
                else:
                    print(f"Warning: Shape mismatch for key '{key}' in segment. Expected ({segment_height}, {segment_width}), got {processed_data.shape}. Skipping reassembly for this segment and key.")

            else:
                 # This warning might be noisy if a key is only generated under certain conditions
                 # print(f"Warning: Result key '{key}' not found in one or more segments.")
                 pass # Suppress warning for missing keys, as some might be conditional

        reassembled_results[key] = reassembled_image

    return reassembled_results


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


    poly_degree = 10       # Degree of polynomial approximation per segment
    # Use 'leja' or 'fekete' which require an admissible mesh. 'padua' does not.
    nodes_method = 'leja' # Node selection method per segment ('full_mesh', 'leja', 'fekete', 'padua')
    # Default admissible mesh type (only relevant for 'leja' and 'fekete' if mesh is not explicitly provided)
    # Removed as only Chebyshev mesh is supported now and handled internally by extremal_points
    # admissible_mesh_type = 'cheb' # Options: 'cheb', 'uni'
    m_cheb = 2 # Parameter 'm' for Chebyshev mesh construction (m > 1)

    segments_x = 20        # Number of segments horizontally
    segments_y = 20        # Number of segments vertically


    # --- 1. Load and Preprocess the Image ---
    try:
        original_image = load_and_preprocess_image(sample_image_path)
        original_image_shape = original_image.shape
        height, width = original_image_shape # Get height and width for plotting grid
        print(f"Original image shape: {original_image_shape}")
    except FileNotFoundError as e:
        print(f"Error loading image: {e}")
        sys.exit(1)


    # --- 2. Segment the Image ---
    print(f"\nSegmenting image into {segments_x}x{segments_y} grid...")
    segments = segment_image(original_image, segments_x, segments_y)
    print(f"Created {len(segments)} segments.")

    # --- 3. Process Each Segment ---
    print("\nProcessing each segment...")
    processed_segments = []
    all_nodes_pixel_coords = [] # List to store pixel coordinates of all nodes
    total_computation_time = 0

    # Define the keys for the approximation and error maps
    result_keys_to_reassemble = [
        'original_segment', 'smoothed_segment',
        'approx_original', 'approx_smoothed',
        'error_original', 'error_smoothed',
        'diff_original_poly_smoothed', 'diff_smoothed_poly_original'
    ]

    for i, segment_info in enumerate(segments):
        print(f"\n--- Processing Segment {i+1}/{len(segments)} ---")
        segment_data = segment_info['segment_data']
        segment_rectangle = segment_info['rectangle'] # Get the rectangle for this segment
        pixel_coords = segment_info['pixel_coords'] # Get pixel coordinates of the segment

        # Call the segment approximation function
        segment_results = image_poly_approximation_segment(
            image_segment=segment_data,
            rectangle=segment_rectangle, # Pass the segment's rectangle
            poly_degree=poly_degree,
            nodes_method=nodes_method,
            m_cheb=m_cheb, # Pass m_cheb for internal mesh generation if needed
            poly_basis=1 # Assuming shifted monomials for now, can be configurable
        )

        if segment_results:
            # Store the results for this segment
            segment_processed_data = {key: segment_results[key] for key in result_keys_to_reassemble if key in segment_results}
            segment_processed_data['pixel_coords'] = pixel_coords # Keep pixel coords for reassembly
            processed_segments.append(segment_processed_data)

            # Collect nodes and transform their coordinates to full image pixel space
            nodes_in_segment_rectangle = segment_results.get('nodes', np.array([]))
            if nodes_in_segment_rectangle.size > 0:
                xmin, ymin, xmax, ymax = segment_rectangle
                x_start, y_start, x_end, y_end = pixel_coords
                segment_width_px = x_end - x_start
                segment_height_px = y_end - y_start

                # Avoid division by zero for degenerate segments
                range_x = xmax - xmin
                range_y = ymax - ymin
                epsilon = 1e-9
                if range_x < epsilon: range_x = epsilon
                if range_y < epsilon: range_y = epsilon

                # Scale nodes from [xmin, xmax]x[ymin, ymax] to [x_start, x_end]x[y_start, y_end]
                nodes_x_pixel = (nodes_in_segment_rectangle[:, 0] - xmin) / range_x * segment_width_px + x_start
                nodes_y_pixel = (nodes_in_segment_rectangle[:, 1] - ymin) / range_y * segment_height_px + y_start

                # Append the transformed nodes to the list
                all_nodes_pixel_coords.extend(np.vstack([nodes_x_pixel, nodes_y_pixel]).T.tolist())


            total_computation_time += segment_results.get('computation_time', 0)
        else:
            print(f"Skipping segment {i+1} due to processing error.")

    # Convert collected nodes to a numpy array
    all_nodes_pixel_coords = np.array(all_nodes_pixel_coords)
    num_total_nodes = all_nodes_pixel_coords.shape[0]


    print(f"\nFinished processing all segments. Total computation time: {total_computation_time:.4f} seconds.")

    # --- 4. Reassemble Results ---
    final_results = {}
    if processed_segments:
        print("\nReassembling results from segments...")
        # Reassemble all specified results
        reassembled_data = reassemble_segments(processed_segments, original_image_shape, result_keys_to_reassemble)
        final_results.update(reassembled_data) # Add reassembled data to final results
        print("Reassembly complete.")

        # --- 5. Save Reassembled Results (PNG only) ---
        print("\nSaving reassembled results...")

        # Dictionary to hold images normalized for saving as PNG
        images_to_save_normalized_png = {}

        # Include original, smoothed, and approximation images (assumed to be in [0,1])
        images_to_save_normalized_png.update({
            f"{image_base_name}_original_image.png": final_results.get('original_segment', np.zeros(original_image_shape)),
            f"{image_base_name}_smoothed_image.png": final_results.get('smoothed_segment', np.zeros(original_image_shape)),
            f"{image_base_name}_approx_original_deg{poly_degree}_{nodes_method}_{segments_x}x{segments_y}.png": final_results.get('approx_original', np.zeros(original_image_shape)),
            f"{image_base_name}_approx_smoothed_deg{poly_degree}_{nodes_method}_{segments_x}x{segments_y}.png": final_results.get('approx_smoothed', np.zeros(original_image_shape)),
        })

        # Save reassembled raw error maps as normalized PNGs (normalized over the full image for visualization)
        print("Saving reassembled raw error maps as normalized PNGs...")
        error_keys = ['error_original', 'error_smoothed', 'diff_original_poly_smoothed', 'diff_smoothed_poly_original']
        for error_key in error_keys:
             raw_key = error_key
             if raw_key in final_results:
                 reassembled_error_map = final_results[raw_key]
                 # Normalization for saving is handled within save_images
                 images_to_save_normalized_png[f"{image_base_name}_{error_key}_deg{poly_degree}_{nodes_method}_{segments_x}x{segments_y}_reassembled_raw_full_normalized.png"] = reassembled_error_map

        # Save the normalized images (PNG)
        save_images(images_to_save_normalized_png, str(IMAGE_RESULTS_DIR))

        print("All generated results saved.")


        # --- 6. Visualize Full Image Results ---
        print("\nGenerating plots with original, approximation, segmentation, nodes, and error heatmap...")

        # Create the plot figure (2 rows, 2 columns)
        fig, axes = plt.subplots(2, 2, figsize=(16, 12)) # Adjusted figure size for 4 plots

        # Plot 1: Original Image with Segmentation Grid
        ax1 = axes[0, 0]
        ax1.imshow(final_results.get('original_segment', np.zeros(original_image_shape)), cmap='gray', vmin=0, vmax=1)
        ax1.set_title(f'Original Image with {segments_x}x{segments_y} Segments')
        ax1.axis('off')

        # Draw segmentation lines on the first plot
        segment_height_px = height // segments_y
        segment_width_px = width // segments_x
        for i in range(1, segments_y):
            y_pos = i * segment_height_px - 0.5 # Subtract 0.5 to align with pixel boundaries
            ax1.axhline(y=y_pos, color='red', linestyle='--', linewidth=1)
        for j in range(1, segments_x):
            x_pos = j * segment_width_px - 0.5 # Subtract 0.5 to align with pixel boundaries
            ax1.axvline(x=x_pos, color='red', linestyle='--', linewidth=1)


        # Plot 2: Original Image with Interpolation Nodes
        ax2 = axes[0, 1]
        ax2.imshow(final_results.get('original_segment', np.zeros(original_image_shape)), cmap='gray', vmin=0, vmax=1)
        # Updated title to include number of nodes and polynomial degree
        ax2.set_title(f'Original Image with {num_total_nodes} {nodes_method.capitalize()} Nodes (deg={poly_degree})')
        ax2.axis('off')

        # Plot the transformed interpolation nodes on the second plot
        if all_nodes_pixel_coords.size > 0:
             # Use a small marker size and alpha for better visibility
             ax2.scatter(all_nodes_pixel_coords[:, 0], all_nodes_pixel_coords[:, 1], s=5, color='blue', alpha=0.7, label='Nodes')
             ax2.legend() # Add legend to show node marker meaning


        # Plot 3: Reassembled Polynomial Approximation (Original)
        ax3 = axes[1, 0]
        ax3.imshow(final_results.get('approx_original', np.zeros(original_image_shape)), cmap='gray', vmin=0, vmax=1)
        ax3.set_title('Reassembled Polynomial Approx (Original)')
        ax3.axis('off')


        # Plot 4: Reassembled Raw Error Heatmap (Original)
        ax4 = axes[1, 1]
        error_map_for_plotting = final_results.get('error_original', np.zeros(original_image_shape)) # Default to error_original

        # Normalize error map for plotting (over the full reassembled image)
        if np.max(error_map_for_plotting) - np.min(error_map_for_plotting) < 1e-8:
            error_plot_data = np.zeros_like(error_map_for_plotting)
            print(f"Warning: Reassembled raw error_original is constant. Plotting uniform heatmap.")
        else:
           error_plot_data = normalize_error_image(error_map_for_plotting)


        im = ax4.imshow(error_plot_data, cmap='viridis', origin='upper')
        fig.colorbar(im, ax=ax4, label='Absolute Error (Original, Full Normalized)')
        ax4.set_title('Reassembled Raw Error Heatmap (Original)')
        ax4.axis('off')


        plt.tight_layout()

        # --- Save the Plot ---
        # Filename reflects the enhanced visualization
        plot_filename = f"{image_base_name}_reconstruction_visualization_deg{poly_degree}_{nodes_method}_{segments_x}x{segments_y}.png"
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
