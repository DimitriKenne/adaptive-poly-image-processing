import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
import time
from typing import Dict, List, Tuple, Union, Optional
import math # Import math for comb

# Add project root to Python path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Define the base path for the results folder
BASE_RESULTS_DIR = PROJECT_ROOT / "results"

# Define the subfolder for adaptive image reconstruction results (renamed)
ADAPTIVE_RECONSTRUCTION_RESULTS_BASE_DIR = BASE_RESULTS_DIR / "adaptive_image_reconstruction_tests"

# Ensure the base results directory for adaptive reconstruction tests exists
os.makedirs(ADAPTIVE_RECONSTRUCTION_RESULTS_BASE_DIR, exist_ok=True)

# Import necessary functions from poly_approx
try:
    # Corrected import: import save_images (plural)
    from poly_approx.image_poly_approximation import image_poly_approximation_segment, save_images
    from poly_approx.edge_processing import combine_errors, apply_threshold, normalize_error_image
    # Corrected import: Import calculate_error_measure from image_reconstruction_metrics
    from poly_approx.image_reconstruction_metrics import calculate_error_measure

    # Import admissible_mesh function (as it was in your provided code)
    from poly_approx.admissible_meshes import compute_admissible_mesh

    _poly_approx_available = True
except ImportError as e:
    print(f"Could not import modules from poly_approx: {e}")
    print("Please ensure your project structure is correct.")
    print("Using dummy functions. Adaptive image reconstruction will be skipped.")
    _poly_approx_available = False

    # Define dummy functions if imports fail
    # Updated dummy function signature to match the expected one in process_segment
    def image_poly_approximation_segment(image_segment, rectangle, poly_degree, nodes_method, admissible_mesh_type, m_cheb, poly_basis):
        print("Dummy image_poly_approximation_segment called.")
        # Return dummy results: original, smoothed, zero approx, zero errors
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
            'nodes_method': nodes_method,
            'poly_degree': poly_degree
        }

    # Dummy save_images function
    def save_images(image_dict, save_folder):
        print("Dummy save_images called.")
        # Dummy implementation: just print what would be saved
        for filename in image_dict.keys():
            print(f"  Dummy saving: {filename} to {save_folder}")


    def combine_errors(error_maps, strategy='max', weights=None):
        print("Dummy combine_errors called.")
        if error_maps:
            first_map_shape = None
            for img in error_maps.values():
                if img.size > 0:
                    first_map_shape = img.shape
                    break
            if first_map_shape is not None:
                 return np.zeros(first_map_shape, dtype=np.float32)
            else:
                 return np.zeros((0, 0), dtype=np.float32)
        else:
            return np.zeros((0, 0), dtype=np.float32)

    def apply_threshold(error_image, threshold_type='fixed', fixed_threshold=0.5):
        print("Dummy apply_threshold called.")
        if error_image.size == 0:
             return np.zeros_like(error_image, dtype=np.uint8)
        return np.zeros_like(error_image, dtype=np.uint8)

    # Dummy calculate_error_measure function (returns a value > threshold to force subdivision in dummy mode)
    def calculate_error_measure(error_map, measure_type='mse'):
        print(f"Dummy calculate_error_measure called with measure: {measure_type}")
        return 1.0 # Always return a value > threshold to force subdivision (in dummy mode)

    def normalize_error_image(error_map, epsilon=1e-8):
         print("Dummy normalize_error_image called.")
         if error_map.size == 0:
             return np.zeros_like(error_map, dtype=np.float32)
         min_val = np.min(error_map)
         max_val = np.max(error_map)
         if max_val - min_val < epsilon:
             return np.zeros_like(error_map, dtype=np.float32)
         return ((error_map - min_val) / (max_val - min_val + epsilon)).astype(np.float32)

    # Dummy compute_admissible_mesh function (as it was in your provided code)
    def compute_admissible_mesh(*args, **kwargs):
        print("Dummy compute_admissible_mesh called.")
        return np.array([]) # Return empty array


def load_image_grayscale(image_path: Path) -> np.ndarray:
    """Loads a grayscale image and normalizes it to [0, 1]."""
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    print(f"Loading image: {image_path}")
    # Open and convert to grayscale ('L') and normalize to [0, 1]
    # Use float32 for consistency with approximation and error maps
    img = np.array(Image.open(image_path).convert('L'), dtype=np.float32) / 255.0
    return img


def process_segment(
    original_image: np.ndarray,
    segment_bbox: Tuple[int, int, int, int], # (row_start, row_end, col_start, col_end)
    poly_degree: int,
    nodes_method: str,
    admissible_mesh_type: str,
    m_cheb: int,
    sigma: float,
    error_measure_type: str,
    error_threshold: float,
    max_depth: int,
    current_depth: int,
    min_segment_size: int, # Added min_segment_size parameter
    results_dict: Dict, # Dictionary to store results from final segments
    segment_id: str
):
    """
    Recursively processes an image segment for polynomial approximation and adaptive refinement.
    Raw error maps and approximation image are stored for final segments.

    Parameters:
    -----------
    original_image : ndarray
        The full original image data.
    segment_bbox : Tuple[int, int, int, int]
        Bounding box of the current segment (row_start, row_end, col_start, col_end).
    poly_degree : int
        Degree of the polynomial approximation for this segment.
    nodes_method : str
        Method for selecting interpolation/approximation nodes.
    admissible_mesh_type : str
        Type of admissible mesh to generate.
    m_cheb : int
        Parameter 'm' for Chebyshev mesh construction.
    sigma : float
        Standard deviation for Gaussian smoothing.
    error_measure_type : str
        Type of error measure to use for adaptive criterion ('mse', 'mae', 'rmse').
    error_threshold : float
        Threshold for the error measure M(S).
    max_depth : int
        Maximum recursive segmentation depth.
    current_depth : int
        Current recursion depth.
    min_segment_size : int
        Minimum dimension (width or height) of a segment to stop subdivision.
    results_dict : Dict
        Dictionary to store results from terminal segments.
    segment_id : str
        Unique identifier for the current segment.
    """
    row_start, row_end, col_start, col_end = segment_bbox
    segment_image = original_image[row_start:row_end, col_start:col_end]

    height, width = segment_image.shape

    print(f"\n--- Processing segment {segment_id} at depth {current_depth} with shape {segment_image.shape} ---")

    if height == 0 or width == 0:
        print(f"Skipping empty segment {segment_id}.")
        return # Skip empty segments

    # --- Base Case 1: Maximum depth reached ---
    if current_depth >= max_depth:
        print(f"Max depth ({max_depth}) reached for segment {segment_id}. Stopping recursion.")
        # Process this segment as a terminal segment
        return process_terminal_reconstruction_segment(
            original_image,
            segment_bbox,
            poly_degree,
            nodes_method,
            admissible_mesh_type,
            m_cheb,
            sigma,
            current_depth,
            results_dict,
            segment_id
        )

    # --- Base Case 2: Segment is too small ---
    if height <= min_segment_size or width <= min_segment_size:
         print(f"Segment size ({width}x{height}) below minimum ({min_segment_size}) for segment {segment_id}. Stopping recursion.")
         # Process this segment as a terminal segment
         return process_terminal_reconstruction_segment(
            original_image,
            segment_bbox,
            poly_degree,
            nodes_method,
            admissible_mesh_type,
            m_cheb,
            sigma,
            current_depth,
            results_dict,
            segment_id
         )


    # Define the rectangle for this segment relative to the original image [0,1]x[0,1]
    original_height, original_width = original_image.shape
    segment_rectangle = (
        col_start / original_width,
        row_start / original_height,
        col_end / original_width,
        row_end / original_height
    )

    # --- Step 1 & 2: Polynomial Approximation and Raw Error Map Calculation ---
    try:
        # Call image_poly_approximation_segment with the rectangle and relevant parameters
        approximation_results = image_poly_approximation_segment(
            image_segment=segment_image,
            rectangle=segment_rectangle, # Pass the rectangle tuple
            poly_degree=poly_degree,
            nodes_method=nodes_method,
            admissible_mesh_type=admissible_mesh_type, # Pass mesh type for internal mesh generation
            m_cheb=m_cheb, # m_cheb is still needed for internal mesh generation if applicable
            poly_basis=1 # Assuming shifted monomials for now, can be configurable
        )
        # Extract raw error maps and approximation image
        raw_error_maps = {
            'error_original': approximation_results.get('error_original', np.zeros_like(segment_image)),
            'error_smoothed': approximation_results.get('error_smoothed', np.zeros_like(segment_image)),
            'diff_original_poly_smoothed': approximation_results.get('diff_original_poly_smoothed', np.zeros_like(segment_image)),
            'diff_smoothed_poly_original': approximation_results.get('diff_smoothed_poly_original', np.zeros_like(segment_image))
        }
        segment_approx_image = approximation_results.get('approx_smoothed', np.zeros_like(segment_image)) # Or approx_original, based on preference
        print(f"Approximation complete for segment {segment_id}.")

    except Exception as e:
        print(f"Error during polynomial approximation for segment {segment_id}: {e}")
        print(f"Stopping processing for segment {segment_id}.")
        # Store zero maps for this segment if approximation fails
        results_dict[segment_id] = {
            'bbox': segment_bbox,
            'approx_image': np.zeros_like(segment_image),
            'raw_error_maps': { # Store zero raw error maps
                'error_original': np.zeros_like(segment_image),
                'error_smoothed': np.zeros_like(segment_image),
                'diff_original_poly_smoothed': np.zeros_like(segment_image),
                'diff_smoothed_poly_original': np.zeros_like(segment_image)
            },
            'depth': current_depth,
            'status': 'approximation_failed'
        }
        return


    # --- Step 4: Evaluate Reconstruction Quality (Compute M(S)) ---
    # Calculate the error measure M(S) on a chosen RAW error map to guide subdivision.
    # Let's use the raw 'error_original' map for M(S) to be more sensitive to original detail.
    error_map_for_measure = raw_error_maps.get('error_original')

    if error_map_for_measure is None:
         print(f"Error: Could not get 'error_original' map for error measure in segment {segment_id}.")
         print(f"Stopping processing for segment {segment_id}.")
         results_dict[segment_id] = {
             'bbox': segment_bbox,
             'approx_image': segment_approx_image,
             'raw_error_maps': raw_error_maps, # Store the available raw error maps
             'depth': current_depth,
             'status': 'error_measure_input_missing'
         }
         return

    try:
        # Calculate error measure on the RAW error map
        segment_error_measure = calculate_error_measure(error_map_for_measure, measure_type=error_measure_type)
        print(f"Error measure ({error_measure_type}) for segment {segment_id}: {segment_error_measure:.6f}")
    except ValueError as e:
        print(f"Error calculating error measure for segment {segment_id}: {e}")
        print(f"Stopping processing for segment {segment_id}.")
        results_dict[segment_id] = {
            'bbox': segment_bbox,
            'approx_image': segment_approx_image,
            'raw_error_maps': raw_error_maps, # Store the problematic error maps
            'depth': current_depth,
            'status': 'error_measure_failed'
        }
        return
    except Exception as e:
        print(f"An unexpected error occurred calculating error measure for segment {segment_id}: {e}")
        print(f"Stopping processing for segment {segment_id}.")
        results_dict[segment_id] = {
            'bbox': segment_bbox,
            'approx_image': segment_approx_image,
            'raw_error_maps': raw_error_maps, # Store the problematic error maps
            'depth': current_depth,
            'status': 'error_measure_failed'
        }
        return


    # --- Step 5: Decision and Refinement ---
    # Stopping criterion met: Either error is low enough, max depth reached, or segment is too small.
    if segment_error_measure <= error_threshold:
        print(f"Stopping for segment {segment_id}: Error measure {segment_error_measure:.6f} <= {error_threshold}.")

        # Store the raw error maps and approximation image for this final segment
        results_dict[segment_id] = {
            'bbox': segment_bbox,
            'approx_image': segment_approx_image,
            'raw_error_maps': raw_error_maps, # Store all raw error maps
            'depth': current_depth,
            'status': 'terminated_by_error' # Indicate termination by error
        }

    else:
        # Error is too high and stopping criteria not met: Subdivide and recurse.
        print(f"Subdividing segment {segment_id}: Error measure {segment_error_measure:.6f} > {error_threshold}.")

        # Subdivide the segment (e.g., into 2x2 sub-segments)
        mid_row = row_start + height // 2
        mid_col = col_start + width // 2

        sub_segments_bbox = []
        # Top-left
        if mid_row > row_start and mid_col > col_start:
            sub_segments_bbox.append((row_start, mid_row, col_start, mid_col))
        # Top-right
        if mid_row > row_start and col_end > mid_col:
             sub_segments_bbox.append((row_start, mid_row, mid_col, col_end))
        # Bottom-left
        if row_end > mid_row and mid_col > col_start:
            sub_segments_bbox.append((mid_row, row_end, col_start, mid_col))
        # Bottom-right
        if row_end > mid_row and col_end > mid_col: # Corrected condition here
            sub_segments_bbox.append((mid_row, row_end, mid_col, col_end))

        # Recursively process sub-segments
        for i, sub_bbox in enumerate(sub_segments_bbox):
            sub_segment_id = f"{segment_id}_{i}"
            process_segment(
                original_image,
                sub_bbox,
                poly_degree,
                nodes_method,
                admissible_mesh_type,
                m_cheb, # Pass the configured m_cheb (which is 2) to sub-segments
                sigma,
                error_measure_type,
                error_threshold,
                max_depth,
                current_depth + 1,
                min_segment_size, # Pass min_segment_size to sub-segments
                results_dict,
                sub_segment_id
            )

def process_terminal_reconstruction_segment(
    original_image: np.ndarray,
    segment_bbox: Tuple[int, int, int, int],
    poly_degree: int,
    nodes_method: str,
    admissible_mesh_type: str,
    m_cheb: int,
    sigma: float,
    current_depth: int,
    results_dict: Dict,
    segment_id: str
):
    """
    Processes a segment that has reached a termination condition (max depth or min size).
    Performs polynomial approximation and stores results.
    """
    row_start, row_end, col_start, col_end = segment_bbox
    segment_image = original_image[row_start:row_end, col_start:col_end]
    height, width = segment_image.shape

    print(f"\n--- Processing terminal segment {segment_id} at depth {current_depth} with shape {segment_image.shape} ---")

    if height == 0 or width == 0:
        print(f"Skipping empty terminal segment {segment_id}.")
        return # Skip empty segments


    # Define the rectangle for this segment relative to the original image [0,1]x[0,1]
    original_height, original_width = original_image.shape
    segment_rectangle = (
        col_start / original_width,
        row_start / original_height,
        col_end / original_width,
        row_end / original_height
    )


    # --- Perform Polynomial Approximation and Raw Error Map Calculation ---
    try:
        # Call image_poly_approximation_segment with the rectangle and relevant parameters
        approximation_results = image_poly_approximation_segment(
            image_segment=segment_image,
            rectangle=segment_rectangle, # Pass the rectangle tuple
            poly_degree=poly_degree,
            nodes_method=nodes_method,
            admissible_mesh_type=admissible_mesh_type, # Pass mesh type for internal mesh generation
            m_cheb=m_cheb, # m_cheb is still needed for internal mesh generation if applicable
            poly_basis=1 # Assuming shifted monomials for now, can be configurable
        )
        raw_error_maps = {
            'error_original': approximation_results.get('error_original', np.zeros_like(segment_image)),
            'error_smoothed': approximation_results.get('error_smoothed', np.zeros_like(segment_image)),
            'diff_original_poly_smoothed': approximation_results.get('diff_original_poly_smoothed', np.zeros_like(segment_image)),
            'diff_smoothed_poly_original': approximation_results.get('diff_smoothed_poly_original', np.zeros_like(segment_image))
        }
        segment_approx_image = approximation_results.get('approx_smoothed', np.zeros_like(segment_image)) # Or approx_original
        print(f"Approximation complete for terminal segment {segment_id}.")

        # Store the raw error maps and approximation image for this terminal segment
        results_dict[segment_id] = {
            'bbox': segment_bbox,
            'approx_image': segment_approx_image,
            'raw_error_maps': raw_error_maps, # Store all raw error maps
            'depth': current_depth,
            'status': 'terminated_by_size_or_depth' # Indicate termination reason
        }

    except Exception as e:
        print(f"Error during polynomial approximation for terminal segment {segment_id}: {e}")
        print(f"Stopping processing for terminal segment {segment_id}.")
        # Store zero maps for this segment if approximation fails
        results_dict[segment_id] = {
            'bbox': segment_bbox,
            'approx_image': np.zeros_like(segment_image),
            'raw_error_maps': { # Store zero raw error maps
                'error_original': np.zeros_like(segment_image),
                'error_smoothed': np.zeros_like(segment_image),
                'diff_original_poly_smoothed': np.zeros_like(segment_image),
                'diff_smoothed_poly_original': np.zeros_like(segment_image)
            },
            'depth': current_depth,
            'status': 'approximation_failed' # Indicate approximation failure
        }
        return


def reassemble_results(image_shape: Tuple[int, int], segment_results: Dict) -> Dict[str, np.ndarray]:
    """
    Reassemble the results (approximation image and raw error maps)
    from processed segments into full image arrays.

    Parameters:
    -----------
    image_shape : Tuple[int, int]
        The shape of the original image (height, width).
    segment_results : Dict
        Dictionary containing results from each final segment, as stored by process_segment.

    Returns:
    --------
    Dict[str, ndarray]
        A dictionary containing the reassembled 'approx_image' and all reassembled
        raw error maps ('error_original', 'error_smoothed', etc.) for the full image.
    """
    height, width = image_shape
    reassembled_approx = np.zeros(image_shape, dtype=np.float32)
    # Initialize reassembled raw error maps
    reassembled_raw_errors = {
        'error_original': np.zeros(image_shape, dtype=np.float32),
        'error_smoothed': np.zeros(image_shape, dtype=np.float32),
        'diff_original_poly_smoothed': np.zeros(image_shape, dtype=np.float32),
        'diff_smoothed_poly_original': np.zeros(image_shape, dtype=np.float32)
    }


    for segment_id, results in segment_results.items():
        # Only reassemble results from segments that terminated successfully or with a known issue
        # Include all terminal statuses for reassembly
        if results['status'] in ['terminated_by_error', 'terminated_by_size_or_depth', 'approximation_failed', 'error_measure_input_missing', 'error_measure_failed', 'mesh_generation_failed']:
             row_start, row_end, col_start, col_end = results['bbox']
             seg_height = row_end - row_start
             seg_width = col_end - col_start
             # Ensure bbox coordinates are within image bounds
             if row_start < 0 or row_end > height or col_start < 0 or col_end > width:
                  print(f"Warning: Segment {segment_id} bbox [{row_start}:{row_end}, {col_start}:{col_end}] is out of original image bounds {image_shape}. Skipping reassembly for this segment.")
                  continue # Skip this segment if bbox is invalid


             # Reassemble approximation image
             if 'approx_image' in results and results['approx_image'].shape == (seg_height, seg_width):
                 reassembled_approx[row_start:row_end, col_start:col_end] = results['approx_image']
             else:
                  print(f"Warning: Approx image data missing or shape mismatch for segment {segment_id}. Expected {(seg_height, seg_width)}, got {results.get('approx_image', np.array([])).shape}. Skipping reassembly for approx_image.")

             # Reassemble raw error maps
             if 'raw_error_maps' in results:
                 for error_key, error_map in results['raw_error_maps'].items():
                     if error_key in reassembled_raw_errors and error_map.shape == (seg_height, seg_width):
                         reassembled_raw_errors[error_key][row_start:row_end, col_start:col_end] = error_map
                     else:
                         print(f"Warning: Raw error map '{error_key}' data missing or shape mismatch for segment {segment_id} or invalid key. Skipping reassembly for this map. Expected {(seg_height, seg_width)}, got {error_map.shape if isinstance(error_map, np.ndarray) else 'N/A'}.")
             else:
                  print(f"Warning: 'raw_error_maps' key missing for segment {segment_id}. Skipping raw error map reassembly for this segment.")


    return {
        'approx_image': reassembled_approx,
        'raw_error_maps': reassembled_raw_errors # Return the dictionary of raw error maps
    }


if __name__ == "__main__":
    if not _poly_approx_available:
        print("Skipping adaptive image reconstruction due to missing poly_approx modules.")
        sys.exit(1)

    # --- Configuration ---
    image_name = "spiral_and_zigzag" # Base name of the image (e.g., "spiral.png")
    image_filename = f"{image_name}.png"
    image_path = PROJECT_ROOT / "images" / image_filename # Assuming images are in a 'images' subfolder

    # Parameters for polynomial approximation within segments
    poly_degree = 3
    nodes_method = 'leja' # or 'fekete', 'padua', 'full_mesh'
    # Note: admissible_mesh_type and m_cheb are used here to generate the mesh
    # passed to image_poly_approximation_segment.
    admissible_mesh_type = 'cheb' # Options: 'cheb', 'uni'
    m_cheb = 2 # Parameter 'm' for Chebyshev mesh construction (m > 1). Reverted to 2 as requested.
    sigma = 1.0 # Standard deviation for Gaussian smoothing

    # Parameters for error map combination and thresholding (applied AFTER reassembly)
    # These are still calculated and saved, but the plot will focus on reconstruction.
    combination_strategy = 'logical_and' # 'max', 'weighted_sum', 'logical_and', 'logical_or'
    combination_weights = None # Define weights if using 'weighted_sum'
    threshold_type = 'otsu' # 'fixed', 'otsu'
    fixed_threshold = 0.5 # Threshold for binary edge map (if threshold_type is 'fixed')

    # Parameters for adaptive segmentation control
    error_measure_type = 'mse' # 'mse', 'mae', 'rmse'
    error_threshold = 0.0001 # Threshold for the error measure M(S)
    max_depth = 15 # Maximum recursive segmentation depth (0 is the whole image). Reverted to 3.
    min_segment_size = 2 # Added minimum segment dimension

    # --- Load the original image ---
    try:
        original_image = load_image_grayscale(image_path)
        original_image_shape = original_image.shape
    except FileNotFoundError as e:
        print(f"Error loading image: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred loading the image: {e}")
        sys.exit(1)

    # Define the initial segment (the whole image)
    initial_bbox = (0, original_image_shape[0], 0, original_image_shape[1])
    initial_segment_id = "root"

    # Dictionary to store results from final segments
    final_segment_results: Dict[str, Dict] = {}

    print(f"\nStarting adaptive image reconstruction for {image_filename}...")
    start_time = time.time()

    # Start the recursive processing from the root segment
    process_segment(
        original_image,
        initial_bbox,
        poly_degree,
        nodes_method,
        admissible_mesh_type,
        m_cheb, # Pass the configured m_cheb (which is 2)
        sigma,
        error_measure_type, # Pass error measure params
        error_threshold,
        max_depth,
        0, # Start at depth 0
        min_segment_size, # Pass min_segment_size
        final_segment_results,
        initial_segment_id
    )

    end_time = time.time()
    print(f"\nAdaptive image reconstruction finished in {end_time - start_time:.4f} seconds.")
    print(f"Processed {len(final_segment_results)} final segments.")

    # --- Reassemble the final results ---
    print("\nReassembling final results...")
    reassembled_final_results = reassemble_results(original_image_shape, final_segment_results)
    print("Reassembly complete.")

    # --- Prepare Data for Plotting the Actual Reconstruction Error ---
    # Get the reassembled raw error_original map
    actual_reconstruction_error_map = reassembled_final_results.get('raw_error_maps', {}).get('error_original', np.zeros(original_image_shape))

    # Normalize this error map for visualization purposes (scale to [0, 1])
    # This ensures the heatmap colormap is applied consistently regardless of the raw error range.
    normalized_reconstruction_error_for_plot = normalize_error_image(actual_reconstruction_error_map)

    # --- Perform Final Error Map Combination and Thresholding (for saving, not primary plot) ---
    # This part is kept to generate the error and binary maps for saving,
    # even though the main plot will focus on reconstruction.
    print("\nPerforming final error map combination and thresholding (for saving)...")

    final_binary_edge_map = np.zeros(original_image_shape, dtype=np.uint8) # Default to zero map
    plot_filename_suffix = "" # Suffix for the edge map filename

    try:
        # Get the required raw error maps for combination/logical ops
        error_maps_for_final_processing = {}
        if combination_strategy in ['max', 'weighted_sum']:
            # Need all four for these strategies
            keys_to_use = [
                'error_original',
                'error_smoothed',
                'diff_original_poly_smoothed',
                'diff_smoothed_poly_original'
            ]
            for key in keys_to_use:
                 if key in reassembled_final_results['raw_error_maps']:
                     error_maps_for_final_processing[key] = reassembled_final_results['raw_error_maps'][key]
                 else:
                      print(f"Warning: Reassembled raw error map '{key}' not found. Using zero map for this key.")
                      error_maps_for_final_processing[key] = np.zeros(original_image_shape, dtype=np.float32)


            # Normalize the selected raw error maps over the FULL IMAGE
            normalized_error_maps_full_image = {
                key: normalize_error_image(err_map)
                for key, err_map in error_maps_for_final_processing.items()
            }

            # Combine normalized error maps into a single composite map (full image)
            composite_error_map_full_image = combine_errors(
                normalized_error_maps_full_image, # Use normalized maps for combination
                strategy=combination_strategy,
                weights=combination_weights
            )

            # Apply threshold to the composite map (full image)
            final_binary_edge_map = apply_threshold(
                composite_error_map_full_image,
                threshold_type=threshold_type,
                fixed_threshold=fixed_threshold
            )
            plot_filename_suffix = f"{combination_strategy}_thresh-{threshold_type}"
            if threshold_type == 'fixed':
                 plot_filename_suffix += f"-{str(fixed_threshold).replace('.', 'p')}"


        elif combination_strategy in ['logical_and', 'logical_or']:
            # Apply threshold to the relevant individual raw error maps and combine
            binary_maps_for_logical_op = []
            # Default keys for logical ops, based on common error types
            logical_op_keys = ['error_original', 'diff_original_poly_smoothed']
            # If combination_weights are provided, use those keys for logical ops
            if combination_weights:
                 logical_op_keys = list(combination_weights.keys())

            if not logical_op_keys:
                 print("Warning: No keys specified for logical operation. Using default ['error_original', 'diff_original_poly_smoothed'].")
                 logical_op_keys = ['error_original', 'diff_original_poly_smoothed']


            for key in logical_op_keys:
                if key in reassembled_final_results['raw_error_maps']:
                    raw_error_map_to_threshold = reassembled_final_results['raw_error_maps'][key]

                    # Normalize the individual raw map over the FULL IMAGE before thresholding
                    normalized_error_map_full_image = normalize_error_image(raw_error_map_to_threshold)

                    try:
                        # Apply threshold to the individual normalized map (full image)
                        binary_map = apply_threshold(
                            normalized_error_map_full_image,
                            threshold_type=threshold_type,
                            fixed_threshold=fixed_threshold
                        )
                        binary_maps_for_logical_op.append(binary_map)
                    except Exception as e:
                        print(f"Error applying threshold to reassembled raw map '{key}' for logical op: {e}")
                        # Append a zero map if thresholding fails for one input
                        binary_maps_for_logical_op.append(np.zeros(original_image_shape, dtype=np.uint8))
                else:
                    print(f"Warning: Reassembled raw error map '{key}' not found for logical operation. Skipping.")
                    # Append a zero map if the input map is missing
                    binary_maps_for_logical_op.append(np.zeros(original_image_shape, dtype=np.uint8))


            if binary_maps_for_logical_op:
                # Combine binary maps using logical AND or OR
                final_binary_edge_map = binary_maps_for_logical_op[0]
                for i in range(1, len(binary_maps_for_logical_op)):
                    if combination_strategy == 'logical_and':
                        final_binary_edge_map = np.logical_and(final_binary_edge_map, binary_maps_for_logical_op[i]).astype(np.uint8)
                    elif combination_strategy == 'logical_or':
                         final_binary_edge_map = np.logical_or(final_binary_edge_map, binary_maps_for_logical_op[i]).astype(np.uint8)
            else:
                print(f"No binary maps generated for logical operation. Final edge map is all zeros.")
                final_binary_edge_map = np.zeros(original_image_shape, dtype=np.uint8)


        else:
            print(f"Error: Invalid combination_strategy '{combination_strategy}' for final edge detection. Final edge map is all zeros.")
            final_binary_edge_map = np.zeros(original_image_shape, dtype=np.uint8)


    except Exception as e:
        print(f"An unexpected error occurred during final error map processing: {e}")
        print(f"Final binary edge map is all zeros.")
        final_binary_edge_map = np.zeros(original_image_shape, dtype=np.uint8)


    # --- Save the final reassembled results ---
    # Create a subfolder for this specific image's results within the adaptive tests directory
    IMAGE_ADAPTIVE_RECONSTRUCTION_RESULTS_DIR = ADAPTIVE_RECONSTRUCTION_RESULTS_BASE_DIR / image_name
    os.makedirs(IMAGE_ADAPTIVE_RECONSTRUCTION_RESULTS_DIR, exist_ok=True)
    print(f"Saving final reassembled results to: {IMAGE_ADAPTIVE_RECONSTRUCTION_RESULTS_DIR}")

    # Construct a filename suffix based on parameters
    params_suffix = (
        f"deg{poly_degree}_{nodes_method}"
        f"_sigma{sigma:.1f}_combine-{combination_strategy}_thresh-{threshold_type}"
        f"{'-' + str(fixed_threshold).replace('.', 'p') if threshold_type == 'fixed' else ''}"
        f"_measure-{error_measure_type}_errthresh{str(error_threshold).replace('.', 'p')}_depth{max_depth}_min{min_segment_size}" # Added min_segment_size to filename
    )

    # Save the reassembled approximation image
    try:
        # Corrected function call: save_images (plural)
        # Save the approximate image (approx_image is already in [0,1] range from reassembly)
        save_images({f"{image_name}_approx_{params_suffix}.png": reassembled_final_results.get('approx_image', np.zeros(original_image_shape))}, str(IMAGE_ADAPTIVE_RECONSTRUCTION_RESULTS_DIR))
    except Exception as e:
        print(f"Error saving reassembled approximation image: {e}")

    # Save a normalized version of the *actual reconstruction error* for visualization
    try:
        # Normalize the actual reconstruction error map for saving as PNG
        normalized_reconstruction_error_for_save = normalize_error_image(actual_reconstruction_error_map)
        save_images({f"{image_name}_actual_error_viz_{params_suffix}.png": normalized_reconstruction_error_for_save}, str(IMAGE_ADAPTIVE_RECONSTRUCTION_RESULTS_DIR))
    except Exception as e:
        print(f"Error saving actual error visualization image: {e}")


    # --- Save the final Binary Edge Map (still useful for analysis) ---
    print("\nSaving final binary edge map...")
    # Use the same suffix as the main results for consistency
    edge_map_filename = f"{image_name}_binary_edge_map_{params_suffix}.png"
    edge_map_filepath = os.path.join(str(IMAGE_ADAPTIVE_RECONSTRUCTION_RESULTS_DIR), edge_map_filename)

    try:
        # binary_edge_map is already uint8 (0 or 1)
        # Scale to 0-255 for standard grayscale PNG
        binary_edge_map_uint8 = (final_binary_edge_map * 255).astype(np.uint8)
        Image.fromarray(binary_edge_map_uint8, 'L').save(edge_map_filepath)
        print(f"Saved binary edge map to {edge_map_filepath}")
    except Exception as e:
        print(f"Error saving final binary edge map to {edge_map_filepath}: {e}")


    # --- Visualize Results (Updated Plot) ---
    print("\nGenerating plot...")
    # Determine the number of subplots (Original, Approximate Image, Actual Error Heatmap)
    num_subplots = 3
    fig, axes = plt.subplots(1, num_subplots, figsize=(6 * num_subplots, 6)) # Adjust figsize

    # Plot Original Image
    # Explicitly set vmin/vmax for grayscale images
    axes[0].imshow(original_image, cmap='gray', vmin=0, vmax=1)
    axes[0].set_title('Original Image')
    axes[0].axis('off')

    # Plot Approximate Image (Reconstruction)
    # Explicitly set vmin/vmax for grayscale images
    axes[1].imshow(reassembled_final_results.get('approx_image', np.zeros(original_image_shape)), cmap='gray', vmin=0, vmax=1)
    axes[1].set_title('Approximate Image (Reconstruction)')
    axes[1].axis('off')

    # Plot Actual Reconstruction Error Heatmap
    # normalized_reconstruction_error_for_plot is already normalized to [0, 1]
    # Use 'viridis' colormap for errors (blue for low error, yellow for high error)
    im = axes[2].imshow(normalized_reconstruction_error_for_plot, cmap='viridis', origin='upper')
    fig.colorbar(im, ax=axes[2], label='Actual Reconstruction Error (Normalized)') # Updated colorbar label
    axes[2].set_title('Actual Reconstruction Error Heatmap') # Updated plot title
    axes[2].axis('off')

    plt.tight_layout()

    # --- Save the Plot ---\
    # Ensure the plot filename reflects it's a reconstruction plot and shows actual error
    plot_filename = f"{image_name}_adaptive_image_reconstruction_plot_actual_error_{params_suffix}.png"
    plot_filepath = os.path.join(str(IMAGE_ADAPTIVE_RECONSTRUCTION_RESULTS_DIR), plot_filename)

    try:
        plt.savefig(plot_filepath)
        print(f"\nSaved adaptive image reconstruction plot to {plot_filepath}")
    except Exception as e:
        print(f"\nError saving adaptive image reconstruction plot to {plot_filepath}: {e}")

    # Close the plot figure
    plt.close(fig)


    print("\nAdaptive image reconstruction script finished.")
