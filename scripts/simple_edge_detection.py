import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image # Used for loading/saving images
import os # Import os for path joining
import time # To measure processing time
from typing import Dict, List, Tuple, Union, Optional # Import typing hints

# Add project root to Python path
# This allows importing modules from the project's root directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Define the base path for the results folder
BASE_RESULTS_DIR = PROJECT_ROOT / "results"

# Define the subfolder where edge detection results will be saved
SCRIPT_RESULTS_BASE_DIR = BASE_RESULTS_DIR / "edge_detection_tests"


# Assuming your project structure is something like:
# your_project/
# ├── poly_approx/
# │   ├── __init__.py  # This empty file is required to make 'poly_approx' a Python package
# │   ├── interpolation_nodes.py
# │   ├── polynomial_bases.py
# │   ├── poly_projector.py
# │   ├── image_poly_approximation.py # Your updated file
# │   ├── admissible_meshes.py # New file
# │   └── edge_processing.py # This is where combination/thresholding functions belong (renamed from error_edge_detection.py)
# └── scripts/
#     └── simple_edge_detection.py # This script (now standalone)

# Flag to check if dummy functions are being used (if imports fail)
using_dummy_functions = False

try:
    # Import necessary functions from poly_approx for segmentation and error processing
    from poly_approx.image_poly_approximation import image_poly_approximation_segment, save_images
    from poly_approx.admissible_meshes import compute_admissible_mesh # Import compute_admissible_mesh function
    # Corrected import: Import from edge_processing instead of error_edge_detection
    from poly_approx.edge_processing import (
        normalize_error_image,
        combine_errors,
        apply_threshold,
        calculate_edge_quality_measure, # Import the quality measure function
        calculate_gradient_magnitude, # Import gradient magnitude for threshold calculation (still needed if measure_type is gradient)
        get_band_around_edges, # Import for band creation
        _scipy_available, # Import the availability flag for scipy
        _skimage_available # Import the availability flag for scikit-image
    )
    # Import other necessary functions if needed (e.g., extremal_points if generating mesh internally)
    from poly_approx.interpolation_nodes import extremal_points # Needed for dummy mesh
    from poly_approx.polynomial_bases import gen_vanderm2d # Needed for dummy poly approx

except ImportError as e:
    print(f"Could not import modules from poly_approx: {e}")
    print("Please ensure your project structure is correct or adjust the import paths in simple_edge_detection.py.")
    print("Expected structure: your_project/poly_approx/ and your_project/scripts/")
    print("Using dummy functions. Edge detection and plotting will be skipped.")
    using_dummy_functions = True
    # Provide dummy functions if imports fail
    def image_poly_approximation_segment(*args, **kwargs):
        print("Dummy image_poly_approximation_segment called.")
        # Return a dummy result dictionary with empty arrays for raw errors
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
            'nodes': np.array([])
        }
    def save_images(*args, **kwargs):
        print("Dummy save_images called.")
    def compute_admissible_mesh(*args, **kwargs):
        print("Dummy compute_admissible_mesh called.")
        return np.array([]) # Return empty array
    def extremal_points(*args, **kwargs):
        print("Dummy extremal_points called.")
        return np.array([])
    def gen_vanderm2d(*args, **kwargs):
        print("Dummy gen_vanderm2d called.")
        return np.eye(1), lambda x: [1] # Return dummy values
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

    def combine_errors(error_dict, strategy='max', weights=None):
        print(f"Dummy combine_errors called with strategy: {strategy}")
        # Return a dummy zero map with the shape of the first input map, if available
        if error_dict:
            first_map = list(error_dict.values())[0]
            return np.zeros_like(first_map, dtype=np.float32)
        else:
            return np.zeros((100, 100), dtype=np.float32) # Default dummy shape

    def apply_threshold(image, threshold_type='fixed', fixed_threshold=0.5): # Renamed from error_image
        print(f"Dummy apply_threshold called with threshold_type: {threshold_type}")
        if image.size == 0: # Check for empty image
             return np.zeros_like(image, dtype=np.uint8)
        return np.zeros_like(image, dtype=np.uint8) # Return all zeros as dummy edge map

    def calculate_edge_quality_measure(*args, **kwargs):
        print("Dummy calculate_edge_quality_measure called.")
        return 0.0 # Return dummy value

    def calculate_gradient_magnitude(*args, **kwargs):
        print("Dummy calculate_gradient_magnitude called.")
        return np.zeros((100, 100), dtype=np.float32) # Return dummy array

    def get_band_around_edges(*args, **kwargs):
        print("Dummy get_band_around_edges called.")
        return np.zeros((100, 100), dtype=np.uint8) # Return dummy array

    _scipy_available = False # Assume not available if import failed
    _skimage_available = False # Assume not available if import failed


def load_image_grayscale(image_path: Path) -> np.ndarray:
    """Loads a grayscale image and normalizes it to [0, 1]."""
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    print(f"Loading image: {image_path}")
    # Open and convert to grayscale ('L') and normalize to [0, 1]
    # Use float32 for consistency with error maps
    img = np.array(Image.open(image_path).convert('L'), dtype=np.float32) / 255.0
    return img


def segment_image(image: np.ndarray, segments_x: int, segments_y: int) -> List[Dict[str, Union[np.ndarray, Tuple[float, float, float, float], Tuple[int, int, int, int]]]]:
    """Divides an image into a grid of segments."""
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
            # This is needed to scale nodes/basis functions correctly within the segment.
            # The image_poly_approximation_segment function now expects this rectangle tuple.
            rectangle = (x_start / width, y_start / height, x_end / width, y_end / height)

            segments.append({
                'segment_data': segment,
                'rectangle': rectangle, # Use 'rectangle' key
                'pixel_coords': (x_start, y_start, x_end, y_end) # Store pixel coordinates for reassembly
            })
    return segments

def process_segment_for_errors(
    segment_data: np.ndarray,
    rectangle: Tuple[float, float, float, float], # Changed from region_min, region_max
    poly_degree: int,
    nodes_method: str,
    admissible_mesh_type: str,
    m_cheb: int,
) -> Dict[str, np.ndarray]:
    """
    Processes a single segment to perform polynomial approximation and generate raw error maps.

    Returns:
    --------
    Dict[str, ndarray]
        A dictionary containing the raw error maps for the segment.
        Returns an empty dictionary if processing fails.
    """
    height, width = segment_data.shape

    if height == 0 or width == 0:
        print("Skipping processing for empty segment.")
        return {}

    # The image_poly_approximation_segment function now handles internal scaling
    # and mesh generation based on the provided 'rectangle'.
    # We no longer need to compute segment_region_center, segment_basis_Zr,
    # or explicitly generate the admissible mesh here before calling it.

    # Perform polynomial approximation on the segment
    # This returns raw error maps
    try:
        segment_approx_results = image_poly_approximation_segment(
            image_segment=segment_data,
            rectangle=rectangle, # Pass the rectangle tuple
            poly_degree=poly_degree,
            nodes_method=nodes_method,
            # Removed basis_Zc, basis_Zr, nodes_region_min, nodes_region_max
            # Removed admissible_mesh as it's generated internally by image_poly_approximation_segment
            m_cheb=m_cheb, # m_cheb is still needed for internal mesh generation if applicable
            poly_basis=1 # Assuming shifted monomials for now, can be configurable
        )

        if not segment_approx_results:
            print("Polynomial approximation failed for segment.")
            return {} # Cannot proceed if approximation fails

        # Extract raw error maps
        raw_error_maps = {
            'error_original': segment_approx_results.get('error_original'),
            'error_smoothed': segment_approx_results.get('error_smoothed'),
            'diff_original_poly_smoothed': segment_approx_results.get('diff_original_poly_smoothed'),
            'diff_smoothed_poly_original': segment_approx_results.get('diff_smoothed_poly_original'),
        }
        # Filter out None values
        raw_error_maps = {k: v for k, v in raw_error_maps.items() if v is not None}

        return raw_error_maps

    except Exception as e:
        print(f"Error during polynomial approximation: {e}")
        return {} # Return empty dict if approximation fails


def reassemble_error_maps(
    image_shape: Tuple[int, int],
    processed_segments_errors: List[Dict[str, np.ndarray]],
    segment_pixel_coords: List[Tuple[int, int, int, int]],
    error_keys: List[str]
) -> Dict[str, np.ndarray]:
    """
    Reassembles raw error maps from processed segments into full image arrays.

    Parameters:
    -----------
    image_shape : Tuple[int, int]
        The shape of the original image (height, width).
    processed_segments_errors : List[Dict[str, np.ndarray]]
        A list where each element is a dictionary containing the raw error maps
        for a single segment.
    segment_pixel_coords : List[Tuple[int, int, int, int]]
        A list of pixel coordinates (x_start, y_start, x_end, y_end) for each segment,
        corresponding to the order in processed_segments_errors.
    error_keys : List[str]
        A list of the keys for the error maps to reassemble.

    Returns:
    --------
    Dict[str, ndarray]
        A dictionary containing the reassembled raw error maps for the full image.
    """
    height, width = image_shape
    reassembled_raw_errors = {key: np.zeros(image_shape, dtype=np.float32) for key in error_keys}

    for i, segment_errors in enumerate(processed_segments_errors):
        x_start, y_start, x_end, y_end = segment_pixel_coords[i]
        seg_height = y_end - y_start
        seg_width = x_end - x_start

        for error_key in error_keys:
            if error_key in segment_errors:
                error_map = segment_errors[error_key]
                if error_map.shape == (seg_height, seg_width):
                    reassembled_raw_errors[error_key][y_start:y_end, x_start:x_end] = error_map
                else:
                    print(f"Warning: Shape mismatch for error map '{error_key}' in segment {i}. Expected {(seg_height, seg_width)}, got {error_map.shape}. Skipping reassembly for this map in this segment.")
            # else:
                 # Warning about missing error key might be noisy if some segments fail
                 # print(f"Warning: Error key '{error_key}' not found in segment {i} results.")


    return reassembled_raw_errors


if __name__ == "__main__":
    if using_dummy_functions:
        print("Skipping edge detection due to import errors.")
        sys.exit(1) # Exit the script if imports failed

    # --- Configuration ---
    # Define the name of the image to process
    image_name = "spiral_and_zigzag" # Replace with the actual base name of your test image eg. spiral_and_zigzag
    image_filename = f"{image_name}.png"
    sample_image_path = PROJECT_ROOT / "images" / image_filename # Assuming images are in a 'images' subfolder

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


    # Define the specific results subfolder for this image within the edge_detection_tests directory
    IMAGE_RESULTS_DIR = SCRIPT_RESULTS_BASE_DIR / image_name

    # Ensure the image-specific results directory for edge detection exists
    os.makedirs(IMAGE_RESULTS_DIR, exist_ok=True)
    print(f"Saving edge detection results to: {IMAGE_RESULTS_DIR}")


    # --- Segmentation Parameters ---
    segments_x = 1        # Number of segments horizontally
    segments_y = 1        # Number of segments vertically

    # --- Polynomial Approximation Parameters (applied per segment) ---
    poly_degree = 3       # Degree of polynomial approximation per segment
    # Use 'leja' or 'fekete' which require an admissible mesh. 'padua' does not.
    nodes_method = 'full_mesh' # Node selection method per segment ('full_mesh', 'leja', 'fekete', 'padua')
    # Default admissible mesh type (only relevant for 'leja' and 'fekete' if mesh is not explicitly provided)
    admissible_mesh_type = 'cheb' # Options: 'cheb', 'uni'
    m_cheb = 2 # Parameter 'm' for Chebyshev mesh construction (m > 1)


    # --- Error Map Combination Strategy ---
    # Define the strategy for combining the error maps.
    # Options: 'max', 'weighted_sum', 'logical_and', 'logical_or'
    # 'max': Take the maximum value across selected error images for each pixel.
    # 'weighted_sum': Combine selected normalized error images using a weighted sum.
    # 'logical_and': Apply threshold to selected normalized maps, then combine binary results with AND.
    #               Requires 'threshold_type' and 'fixed_threshold' to be handled by the caller
    #               before passing binary maps if this function were to handle binary ops.
    # 'logical_or': Apply threshold to selected normalized maps, then combine binary maps with OR.
    combination_strategy = 'logical_and' # Choose your desired strategy

    # --- Parameters for 'weighted_sum' strategy ---
    # Define weights for the 'weighted_sum' strategy (keys must match error map keys being combined)
    # Only used if combination_strategy is 'weighted_sum'
    # The keys here should correspond to the error map keys you intend to combine.
    combination_weights = {
        'error_original': 0.3,
        'error_smoothed': 0.3,
        'diff_original_poly_smoothed': 0.2,
        'diff_smoothed_poly_original': 0.2,
    }
     # Ensure weights sum to 1.0 if using weighted sum and you want the output in [0,1]
    if combination_strategy == 'weighted_sum' and combination_weights is not None:
        total_weight = sum(combination_weights.values())
        if abs(total_weight - 1.0) > 1e-6:
            print(f"Warning: Provided weights for 'weighted_sum' sum to {total_weight}. Normalizing weights.")
            weight_sum = sum(combination_weights.values())
            combination_weights = {k: v / weight_sum for k, v in combination_weights.items()}
            print(f"Normalized weights: {combination_weights}")
        elif total_weight == 0:
             print("Warning: Provided weights for 'weighted_sum' sum to 0. Using equal weights.")
             num_keys = len(combination_weights)
             if num_keys > 0:
                  equal_weight = 1.0 / num_keys
                  combination_weights = {k: equal_weight for k in combination_weights.keys()}
             else:
                  print("Error: No keys provided in combination_weights for 'weighted_sum'. Cannot proceed.")
                  sys.exit(1)


    # --- Parameters for 'logical_and' or 'logical_or' strategies ---
    # Define which raw error maps to use for logical operations.
    # Provide a list of keys from ['error_original', 'error_smoothed', 'diff_original_poly_smoothed', 'diff_smoothed_poly_original']
    # These raw maps will be normalized over the full image before thresholding for logical ops.
    logical_op_error_keys = ['error_original', 'diff_original_poly_smoothed'] # Example: Combine edges from original error and original-poly(smoothed) diff

    # --- Thresholding Parameters ---
    # Define the thresholding type and value.
    # Note: For 'logical_and' or 'logical_or', this threshold is applied to *each* individual map
    # specified in `logical_op_error_keys` after normalization and before the logical combination.
    # For 'max' or 'weighted_sum', this threshold is applied to the final composite map.
    # 'fixed': Use a fixed threshold (0.0 to 1.0)
    # 'otsu': Use Otsu's method (requires scikit-image)
    threshold_type = 'otsu'
    fixed_threshold = 0.5 # Only used if threshold_type is 'fixed'


    # --- 1. Load and Preprocess the Image ---
    try:
        original_image = load_image_grayscale(sample_image_path)
        original_image_shape = original_image.shape
        print(f"Original image shape: {original_image_shape}")
    except FileNotFoundError as e:
        print(f"Error loading image: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred loading the image: {e}")
        sys.exit(1)


    # --- 2. Segment the Image ---
    print(f"\nSegmenting image into {segments_x}x{segments_y} grid...")
    segments = segment_image(original_image, segments_x, segments_y)
    print(f"Created {len(segments)} segments.")

    # --- 3. Process Each Segment to Get Raw Error Maps ---
    print("\nProcessing each segment to get raw error maps...")
    processed_segments_errors = []
    segment_pixel_coords_list = [] # Store pixel coords for reassembly

    # Define the keys for the raw error maps we expect
    raw_error_keys = ['error_original', 'error_smoothed', 'diff_original_poly_smoothed', 'diff_smoothed_poly_original']

    for i, segment_info in enumerate(segments):
        print(f"\n--- Processing Segment {i+1}/{len(segments)} for errors ---")
        segment_data = segment_info['segment_data']
        segment_rectangle = segment_info['rectangle'] # Get the rectangle for this segment
        pixel_coords = segment_info['pixel_coords']

        # Call process_segment_for_errors with the rectangle
        segment_raw_errors = process_segment_for_errors(
            segment_data,
            segment_rectangle, # Pass the rectangle
            poly_degree,
            nodes_method,
            admissible_mesh_type,
            m_cheb,
        )

        if segment_raw_errors:
            processed_segments_errors.append(segment_raw_errors)
            segment_pixel_coords_list.append(pixel_coords)
        else:
            print(f"Skipping segment {i+1} due to processing error.")


    if not processed_segments_errors:
        print("\nNo segments were successfully processed to get error maps. Cannot proceed with edge detection.")
        sys.exit(1)

    # --- 4. Reassemble Raw Error Maps ---
    print("\nReassembling raw error maps from segments...")
    reassembled_raw_error_maps = reassemble_error_maps(
        original_image_shape,
        processed_segments_errors,
        segment_pixel_coords_list,
        raw_error_keys # Reassemble all four raw error maps
    )
    print("Reassembly complete.")

    # Ensure reassembled raw error maps are not empty
    if not reassembled_raw_error_maps or any(img.size == 0 for img in reassembled_raw_error_maps.values()):
         print("\nError: Reassembled raw error maps are empty. Cannot proceed with edge detection.")
         sys.exit(1)


    # --- 5. Process Error Maps based on Combination Strategy ---

    final_binary_edge_map = None
    heatmap_data = None # Data to show in the heatmap plot
    heatmap_title = ""
    plot_filename_suffix = ""

    if combination_strategy in ['max', 'weighted_sum']:
        # These strategies operate on normalized inputs.
        # Normalize the reassembled raw error maps over the full image for combination.
        print("\nNormalizing reassembled raw error maps over the full image for combination.")
        error_maps_for_combination = {
            key: normalize_error_image(err_map)
            for key, err_map in reassembled_raw_error_maps.items()
        }

        # Combine normalized error maps into a single composite map
        print(f"Combining normalized error maps using '{combination_strategy}' strategy...")
        try:
            composite_error_map = combine_errors(
                error_maps_for_combination,
                strategy=combination_strategy, # This will be 'max' or 'weighted_sum'
                weights=combination_weights if combination_strategy == 'weighted_sum' else None
            )
            print("Combination complete.")
            # Apply threshold to the composite map
            print(f"\nApplying {threshold_type} thresholding to composite map...")
            # apply_threshold expects input in [0, 1], which composite_error_map should be.
            final_binary_edge_map = apply_threshold(
                composite_error_map,
                threshold_type=threshold_type,
                fixed_threshold=fixed_threshold
            )
            print("Thresholding complete.")

            # For plotting, the heatmap will be the composite map
            heatmap_data = composite_error_map
            heatmap_title = f'Composite Error Heatmap ({combination_strategy.capitalize()})'
            plot_filename_suffix = f"{combination_strategy}_thresh-{threshold_type}"
            if threshold_type == 'fixed':
                 plot_filename_suffix += f"-{str(fixed_threshold).replace('.', 'p')}"


        except ValueError as e:
            print(f"Error during combination or thresholding: {e}")
            print("Skipping edge detection.")
            sys.exit(1)
        except Exception as e:
            print(f"An unexpected error occurred during combination or thresholding: {e}")
            print("Skipping edge detection.")
            sys.exit(1)


    elif combination_strategy in ['logical_and', 'logical_or']:
        # Apply threshold to individual reassembled raw error maps after normalizing them over the full image
        print(f"\nApplying {threshold_type} thresholding to individual reassembled maps for logical operation...")

        binary_maps_for_logical_op = []
        for key in logical_op_error_keys:
            if key not in reassembled_raw_error_maps:
                 print(f"Error: Reassembled raw error map '{key}' not available for logical operation.")
                 sys.exit(1)

            raw_error_map_to_threshold = reassembled_raw_error_maps[key]

            # Normalize the individual raw map over the FULL IMAGE before thresholding
            print(f"Normalizing reassembled raw map '{key}' over the full image before thresholding.")
            normalized_error_map_full_image = normalize_error_image(raw_error_map_to_threshold)


            try:
                # apply_threshold expects input in [0, 1]
                binary_map = apply_threshold(
                    normalized_error_map_full_image,
                    threshold_type=threshold_type,
                    fixed_threshold=fixed_threshold
                )
                binary_maps_for_logical_op.append(binary_map)
                print(f"Thresholding applied to '{key}'.")
            except ValueError as e:
                print(f"Error applying threshold to '{key}': {e}")
                print("Skipping logical operation.")
                sys.exit(1)
            except Exception as e:
                print(f"An unexpected error occurred during thresholding of '{key}': {e}")
                print("Skipping logical operation.")
                sys.exit(1)


        if not binary_maps_for_logical_op:
             print("No binary maps generated for logical operation.")
             sys.exit(1)

        # Combine binary maps using logical AND or OR
        print(f"\nCombining binary maps using logical '{combination_strategy.split('_')[-1].upper()}'...")
        # Initialize with the first binary map
        final_binary_edge_map = binary_maps_for_logical_op[0]

        for i in range(1, len(binary_maps_for_logical_op)):
            if combination_strategy == 'logical_and':
                final_binary_edge_map = np.logical_and(final_binary_edge_map, binary_maps_for_logical_op[i]).astype(np.uint8)
            elif combination_strategy == 'logical_or':
                 final_binary_edge_map = np.logical_or(final_binary_edge_map, binary_maps_for_logical_op[i]).astype(np.uint8)

        print("Logical operation complete.")

        # For plotting the heatmap in logical operation case, show the first input error map used (full-normalized)
        heatmap_data = normalize_error_image(reassembled_raw_error_maps[logical_op_error_keys[0]])

        heatmap_title = f'Input Error Heatmap ({logical_op_error_keys[0]})'
        plot_filename_suffix = f"{combination_strategy}_keys-{''.join([key[0] for key in logical_op_error_keys])}_thresh-{threshold_type}"
        if threshold_type == 'fixed':
             plot_filename_suffix += f"-{str(fixed_threshold).replace('.', 'p')}"


    else:
        print(f"Error: Invalid combination_strategy '{combination_strategy}'.")
        sys.exit(1)

    # Ensure a binary edge map was generated
    if final_binary_edge_map is None:
        print("Error: Failed to generate a binary edge map.")
        sys.exit(1)


    # --- Load Original Image for Plotting (already loaded at the start) ---
    # We already have the original_image loaded


    # --- Visualize Results ---
    print("\nGenerating plot...")
    # Determine the number of subplots based on whether the original image was loaded
    # We always have the original image loaded now
    num_subplots = 3
    fig, axes = plt.subplots(1, num_subplots, figsize=(6 * num_subplots, 6)) # Adjust figsize based on number of plots

    # Plot Original Image with Segmentation Grid (only on this plot)
    ax1 = axes[0]
    # Added vmin=0, vmax=1 for consistent grayscale plotting
    ax1.imshow(original_image, cmap='gray', vmin=0, vmax=1)
    # Updated title to include segmentation info
    ax1.set_title(f'Original Image with {segments_x}x{segments_y} Segments')
    ax1.axis('off')

    # Draw segmentation lines ONLY on the original image plot
    if segments_x > 1 or segments_y > 1:
        height, width = original_image_shape
        segment_height = height // segments_y
        segment_width = width // segments_x

        for i in range(1, segments_y):
            y_pos = i * segment_height - 0.5 # Subtract 0.5 to align with pixel boundaries
            ax1.axhline(y=y_pos, color='red', linestyle='--', linewidth=1)
        for j in range(1, segments_x):
            x_pos = j * segment_width - 0.5 # Subtract 0.5 to align with pixel boundaries
            ax1.axvline(x=x_pos, color='red', linestyle='--', linewidth=1)


    # Plot Heatmap (Composite or Normalized Individual)
    # Use 'viridis' colormap for errors (blue for low error, yellow for high error)
    # heatmap_data is already full-normalized [0,1]
    ax2 = axes[1]
    im = ax2.imshow(heatmap_data, cmap='viridis', origin='upper')
    fig.colorbar(im, ax=ax2, label=heatmap_title) # Use heatmap_title for colorbar label
    # Updated title to include degree and nodes method
    ax2.set_title(f'{heatmap_title}\n(deg={poly_degree}, nodes={nodes_method})')
    ax2.axis('off')

    # Removed segmentation lines from heatmap plot


    # Plot Binary Edge Map
    ax3 = axes[2]
    ax3.imshow(final_binary_edge_map, cmap='gray') # Binary map is typically grayscale
    # Updated title to include degree and nodes method
    ax3.set_title(f'Detected Edges')
    ax3.axis('off')

    # Removed segmentation lines from binary map plot


    plt.tight_layout()

    # --- Save the Plot ---
    # Include segmentation parameters in the filename
    plot_filename = f"{image_name}_edge_detection_plot_seg{segments_x}x{segments_y}_deg{poly_degree}_{nodes_method}_{plot_filename_suffix}.png"
    plot_filepath = os.path.join(str(IMAGE_RESULTS_DIR), plot_filename)

    try:
        plt.savefig(plot_filepath)
        print(f"\nSaved plot to {plot_filepath}")
    except Exception as e:
        print(f"\nError saving plot to {plot_filepath}: {e}")

    # Close the plot figure
    plt.close(fig)


    # --- Save the final Binary Edge Map ---
    print("\nSaving final binary edge map...")
    # Use the same filename suffix as the plot for consistency
    edge_map_filename = f"{image_name}_binary_edge_map_seg{segments_x}x{segments_y}_deg{poly_degree}_{nodes_method}_{plot_filename_suffix}.png"
    edge_map_filepath = os.path.join(str(IMAGE_RESULTS_DIR), edge_map_filename)

    try:
        # binary_edge_map is already uint8 (0 or 1)
        # Scale to 0-255 for standard grayscale PNG
        binary_edge_map_uint8 = (final_binary_edge_map * 255).astype(np.uint8)
        Image.fromarray(binary_edge_map_uint8, 'L').save(edge_map_filepath)
        print(f"Saved binary edge map to {edge_map_filepath}")
    except Exception as e:
        print(f"Error saving binary edge map to {edge_map_filepath}: {e}")


    print("\nEdge detection script finished.")
