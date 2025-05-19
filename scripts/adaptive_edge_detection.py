import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
import time
from typing import Dict, List, Tuple, Union, Optional

# Add project root to Python path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Define the base path for the results folder
BASE_RESULTS_DIR = PROJECT_ROOT / "results"

# Define the subfolder for adaptive edge detection results
SCRIPT_RESULTS_BASE_DIR = BASE_RESULTS_DIR / "adaptive_edge_detection_tests"

# --- Import core poly_approx modules ---
# These are modules expected to be present for the script's core functionality
try:
    # Removed compute_admissible_mesh import as it's now handled internally by image_poly_approximation_segment
    from poly_approx.image_poly_approximation import image_poly_approximation_segment, save_images
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
    # Removed extremal_points import as it's now handled internally by image_poly_approximation_segment
    # from poly_approx.interpolation_nodes import extremal_points

    _imports_successful = True
except ImportError as e:
    print(f"Error: Could not import core modules from poly_approx: {e}")
    print("Please ensure your project structure is correct or adjust the import paths.")
    print("Expected structure: your_project/poly_approx/ and your_project/scripts/")
    _imports_successful = False

# Removed the separate SciPy and scikit-image import blocks here,
# as the flags and functions are now imported directly from error_edge_detection.py

# Define a structure to hold results from each processed segment
# This will be used to collect edge maps from terminal segments
SegmentResult = Dict[str, Union[np.ndarray, Tuple[int, int, int, int]]]


def load_and_preprocess_image(image_path: Path) -> np.ndarray:
    """Loads a grayscale image and normalizes it to [0, 1]."""
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    print(f"Loading image: {image_path}")
    img = np.array(Image.open(image_path).convert('L'), dtype=np.float32) / 255.0
    return img


def process_segment_recursively(
    segment_data: np.ndarray,
    segment_pixel_coords: Tuple[int, int, int, int], # (x_start, y_start, x_end, y_end)
    segment_region_min: np.ndarray, # relative to original image [0,1]x[0,1]
    segment_region_max: np.ndarray, # relative to original image [0,1]x[0,1]
    original_image_shape: Tuple[int, int],
    poly_degree: int,
    nodes_method: str,
    admissible_mesh_type: str, # Still needed for internal node generation
    m_cheb: int, # Still needed for internal node generation
    error_combination_strategy: str,
    error_combination_weights: Optional[Dict[str, float]],
    edge_threshold_type: str,
    fixed_edge_threshold: float,
    edge_quality_measure_type: str,
    edge_quality_threshold: float, # T_threshold for M(S) - Now calculated dynamically
    edge_quality_band_width: int,
    current_depth: int,
    max_depth: int,
    min_segment_size: int # Minimum dimension (width or height) to stop subdivision
) -> List[SegmentResult]:
    """
    Recursively processes an image segment, performing polynomial approximation,
    edge detection, evaluating edge quality, and subdividing if necessary.

    Parameters:
    -----------
    segment_data : ndarray
        The 2D numpy array representing the current image segment (grayscale, normalized).
    segment_pixel_coords : Tuple[int, int, int, int]
        The pixel coordinates (x_start, y_start, x_end, y_end) of this segment
        relative to the original full image.
    segment_region_min : ndarray
        Minimum coordinates [x_min, y_min] of the segment's region relative to the
        original image scaled to [0,1]x[0,1].
    segment_region_max : ndarray
        Maximum coordinates [x_max, y_max] of the segment's region relative to the
        original image scaled to [0,1]x[0,1].
    original_image_shape : Tuple[int, int]
        The shape (height, width) of the original full image.
    poly_degree : int
        The degree of the polynomial approximation for this segment.
    nodes_method : str
        The method for selecting interpolation/approximation nodes ('full_mesh', 'leja', 'fekete', 'padua').
    admissible_mesh_type : str
        The type of admissible mesh to generate if needed ('cheb', 'uni').
    m_cheb : int
        Parameter 'm' for Chebyshev mesh construction.
    error_combination_strategy : str
        Strategy for combining error maps ('max', 'weighted_sum', 'logical_and', 'logical_or').
    error_combination_weights : Optional[Dict[str, float]], optional
        Weights for 'weighted_sum'.
    edge_threshold_type : str
        Type of thresholding for edge detection ('fixed', 'otsu').
    fixed_edge_threshold : float
        Fixed threshold value if edge_threshold_type is 'fixed'.
    edge_quality_measure_type : str
        Type of measure for evaluating edge quality M(S).
    edge_quality_threshold : float
        Threshold for the edge quality measure (T_threshold).
    edge_quality_band_width : int
        Width of the band in pixels.
    current_depth : int
        The current recursion depth.
    max_depth : int
        The maximum allowed recursion depth.
    min_segment_size: int # Minimum dimension (width or height) to stop subdivision

    Returns:
    --------
    List[SegmentResult]
        A list of dictionaries, each containing the binary edge map and pixel
        coordinates for segments where the recursion terminated.
    """
    height, width = segment_data.shape
    x_start, y_start, x_end, y_end = segment_pixel_coords

    print(f"Processing segment at depth {current_depth} / {max_depth} [{x_start}:{x_end}, {y_start}:{y_end}] (Shape: {height}x{width})")

    # --- Base Case 1: Maximum depth reached ---
    if current_depth >= max_depth:
        print(f"Max depth ({max_depth}) reached. Stopping recursion for this segment.")
        # Process this segment and return its edge map
        return process_terminal_segment(
            segment_data,
            segment_pixel_coords,
            segment_region_min,
            segment_region_max,
            original_image_shape,
            poly_degree,
            nodes_method,
            admissible_mesh_type, # Pass to terminal segment processing
            m_cheb, # Pass to terminal segment processing
            error_combination_strategy,
            error_combination_weights,
            edge_threshold_type,
            fixed_edge_threshold
        )

    # --- Base Case 2: Segment is too small ---
    if height <= min_segment_size or width <= min_segment_size:
         print(f"Segment size ({width}x{height}) below minimum ({min_segment_size}). Stopping recursion.")
         # Process this segment and return its edge map
         return process_terminal_segment(
            segment_data,
            segment_pixel_coords,
            segment_region_min,
            segment_region_max,
            original_image_shape,
            poly_degree,
            nodes_method,
            admissible_mesh_type, # Pass to terminal segment processing
            m_cheb, # Pass to terminal segment processing
            error_combination_strategy,
            error_combination_weights,
            edge_threshold_type,
            fixed_edge_threshold
        )


    # --- Step 1-3: Polynomial Approximation, Error Maps, and Initial Error Calculation for Quality Evaluation ---
    # We need an initial error map for the segment to calculate M(S).
    # This error map is based on the current polynomial approximation.
    print("Performing polynomial approximation and initial error calculation for quality evaluation...")

    # Removed compute_admissible_mesh call here, it's handled internally by image_poly_approximation_segment

    # Perform polynomial approximation on the segment
    # This returns raw error maps
    segment_approx_results = image_poly_approximation_segment(
        image_segment=segment_data,
        poly_degree=poly_degree,
        nodes_method=nodes_method,
        # Pass the segment's region bounds as the rectangle
        rectangle=(segment_region_min[0], segment_region_min[1], segment_region_max[0], segment_region_max[1]),
        admissible_mesh_type=admissible_mesh_type, # Pass mesh type for internal node generation
        m_cheb=m_cheb # Pass m_cheb for internal node generation
        # Removed basis_Zc, basis_Zr, nodes_region_min, nodes_region_max, admissible_mesh
    )

    if not segment_approx_results:
        print(f"Polynomial approximation failed for segment at depth {current_depth}. Returning empty list.")
        return [] # Cannot proceed if approximation fails

    # Extract raw error maps needed for combination and quality measure
    raw_error_maps = {
        'error_original': segment_approx_results.get('error_original'),
        'error_smoothed': segment_approx_results.get('error_smoothed'),
        'diff_original_poly_smoothed': segment_approx_results.get('diff_original_poly_smoothed'),
        'diff_smoothed_poly_original': segment_approx_results.get('diff_smoothed_poly_original'),
    }
    # Filter out None values if any error map was not generated
    raw_error_maps = {k: v for k, v in raw_error_maps.items() if v is not None}

    if not raw_error_maps:
         print(f"No error maps generated for segment at depth {current_depth}. Cannot calculate quality measure. Returning empty list.")
         return []


    # --- Combine Error Maps for Quality Measure Calculation ---
    # We need a single error map to calculate the quality measure on.
    # Using the 'max' strategy for combination for the quality measure calculation
    # seems reasonable, as it highlights areas with high error in *any* of the maps.
    # Normalize raw error maps over the segment for combination for quality measure
    error_maps_for_quality_combination = {}
    print("Normalizing error maps over the segment for quality measure combination.")
    for key, error_map in raw_error_maps.items():
         error_maps_for_quality_combination[key] = normalize_error_image(error_map)

    # Combine normalized error maps for the quality measure
    composite_error_map_for_quality = combine_errors(
        error_maps_for_quality_combination,
        strategy='max' # Using 'max' strategy for combining errors for quality measure
    )
    print("Combined error maps for quality measure.")


    # --- Step 4: Evaluate Edge Quality M(S) ---
    # The quality measure is calculated on the composite error map for the segment.
    # The band around edges for the quality measure calculation will be based on
    # an initial edge detection on the *composite error map itself* within the segment.
    print(f"Calculating edge quality measure M(S) using '{edge_quality_measure_type}'...")

    # Generate a binary map from the composite error map to define the band for M(S)
    # Use Otsu's thresholding on the composite error map for the band definition
    # Ensure skimage is available before attempting to use apply_threshold with 'otsu'
    if not _skimage_available:
         print("Error: scikit-image is required for Otsu thresholding for quality band definition. Exiting.")
         return [] # Cannot proceed if skimage is required but not available

    initial_binary_edge_map_for_quality_band = apply_threshold(
        composite_error_map_for_quality,
        threshold_type='otsu' # Using Otsu on the composite error map for band definition
    )
    print(f"Generated initial binary edge map for quality band (Sum: {np.sum(initial_binary_edge_map_for_quality_band)}).")

    # Ensure get_band_around_edges is available (requires skimage)
    if not _skimage_available:
        print("Error: scikit-image is required for get_band_around_edges for quality measure. Exiting.")
        return [] # Cannot proceed if skimage is required but not available

    # Get the band around edges in the composite error map for this segment
    segment_band_mask = get_band_around_edges(initial_binary_edge_map_for_quality_band, edge_quality_band_width)

    # Pass the composite error map and the generated band mask to calculate_edge_quality_measure
    segment_edge_quality = calculate_edge_quality_measure(
        image_for_quality=composite_error_map_for_quality, # Pass the composite error map as the image_for_quality
        image_source_for_band=composite_error_map_for_quality, # Use composite error map to define band
        measure_type=edge_quality_measure_type,
        band_width=edge_quality_band_width, # Pass the band width
        quantile=None # Calculate mean for quality measure (not quantile)
    )
    print(f"Segment edge quality M(S) = {segment_edge_quality:.4f}")

    # --- Step 5: Decision and Refinement ---
    if segment_edge_quality <= edge_quality_threshold:
        print(f"Edge quality M(S) ({segment_edge_quality:.4f}) is below threshold ({edge_quality_threshold:.4f}). Stopping recursion.")
        # Quality is satisfactory, process this segment as a terminal segment
        return process_terminal_segment(
            segment_data,
            segment_pixel_coords,
            segment_region_min,
            segment_region_max,
            original_image_shape,
            poly_degree,
            nodes_method,
            admissible_mesh_type, # Pass to terminal segment processing
            m_cheb, # Pass to terminal segment processing
            error_combination_strategy,
            error_combination_weights,
            edge_threshold_type,
            fixed_edge_threshold
        )
    else:
        print(f"Edge quality M(S) ({segment_edge_quality:.4f}) is above threshold ({edge_quality_threshold:.4f}). Subdividing segment.")
        # Quality is not satisfactory, subdivide and recurse

        # Subdivide the segment (e.g., into 2x2 grid)
        sub_segments = []
        sub_height = height // 2
        sub_width = width // 2

        # Handle cases where segment dimension is 1 (cannot subdivide further)
        if sub_height == 0 or sub_width == 0:
             print("Segment dimension is 1, cannot subdivide further despite quality below threshold. Processing as terminal.")
             return process_terminal_segment(
                segment_data,
                segment_pixel_coords,
                segment_region_min,
                segment_region_max,
                original_image_shape,
                poly_degree,
                nodes_method,
                admissible_mesh_type, # Pass to terminal segment processing
                m_cheb, # Pass to terminal segment processing
                error_combination_strategy,
                error_combination_weights,
                edge_threshold_type,
                fixed_edge_threshold
            )


        for i in range(2):
            for j in range(2):
                y_start_sub = y_start + i * sub_height
                y_end_sub = y_start_sub + sub_height
                x_start_sub = x_start + j * sub_width
                x_end_sub = x_start_sub + sub_width

                # Ensure the last sub-segment covers the remaining pixels if height/width is odd
                if i == 1 and height % 2 != 0:
                    y_end_sub = y_end
                if j == 1 and width % 2 != 0:
                    x_end_sub = x_end

                sub_segment_data = segment_data[i * sub_height : i * sub_height + sub_height + (height % 2 if i == 1 else 0),
                                                j * sub_width : j * sub_width + sub_width + (width % 2 if j == 1 else 0)]

                sub_segment_pixel_coords = (x_start_sub, y_start_sub, x_end_sub, y_end_sub)

                # Calculate the region min/max for the sub-segment relative to the original image [0,1]x[0,1]
                original_height, original_width = original_image_shape
                sub_segment_region_min = np.array([x_start_sub / original_width, y_start_sub / original_height], dtype=np.float32)
                sub_segment_region_max = np.array([x_end_sub / original_width, y_end_sub / original_image_shape[0]], dtype=np.float32)


                # Recursive call for the sub-segment
                sub_segments.extend(process_segment_recursively(
                    sub_segment_data,
                    sub_segment_pixel_coords,
                    sub_segment_region_min,
                    sub_segment_region_max,
                    original_image_shape,
                    poly_degree,
                    nodes_method,
                    admissible_mesh_type,
                    m_cheb,
                    error_combination_strategy,
                    error_combination_weights,
                    edge_threshold_type,
                    fixed_edge_threshold,
                    edge_quality_measure_type, # Pass the determined quality measure type
                    edge_quality_threshold, # Pass the calculated threshold
                    edge_quality_band_width,
                    current_depth + 1,
                    max_depth,
                    min_segment_size
                ))
        return sub_segments


def process_terminal_segment(
    segment_data: np.ndarray,
    segment_pixel_coords: Tuple[int, int, int, int],
    segment_region_min: np.ndarray,
    segment_region_max: np.ndarray,
    original_image_shape: Tuple[int, int],
    poly_degree: int,
    nodes_method: str,
    admissible_mesh_type: str, # Still needed for internal node generation
    m_cheb: int, # Still needed for internal node generation
    error_combination_strategy: str,
    error_combination_weights: Optional[Dict[str, float]],
    edge_threshold_type: str,
    fixed_edge_threshold: float
) -> List[SegmentResult]:
    """
    Processes a segment that has reached a termination condition (max depth or quality threshold met).
    Performs polynomial approximation, combines errors, applies final thresholding,
    and returns the binary edge map for this segment.

    Returns:
    --------
    List[SegmentResult]
        A list containing a single dictionary with the binary edge map and pixel
        coordinates for this terminal segment.
    """
    height, width = segment_data.shape
    x_start, y_start, x_end, y_end = segment_pixel_coords

    print(f"Processing terminal segment [{x_start}:{x_end}, {y_start}:{y_end}] (Shape: {height}x{width})")

    # Removed compute_admissible_mesh call here, it's handled internally by image_poly_approximation_segment

    # Perform polynomial approximation on the segment
    # This returns raw error maps
    segment_approx_results = image_poly_approximation_segment(
        image_segment=segment_data,
        poly_degree=poly_degree,
        nodes_method=nodes_method,
        # Pass the segment's region bounds as the rectangle
        rectangle=(segment_region_min[0], segment_region_min[1], segment_region_max[0], segment_region_max[1]),
        admissible_mesh_type=admissible_mesh_type, # Pass mesh type for internal node generation
        m_cheb=m_cheb # Pass m_cheb for internal node generation
        # Removed basis_Zc, basis_Zr, nodes_region_min, nodes_region_max, admissible_mesh
    )

    if not segment_approx_results:
        print("Polynomial approximation failed for terminal segment. Returning empty list.")
        return []

    # Extract raw error maps needed for final edge detection
    raw_error_maps = {
        'error_original': segment_approx_results.get('error_original'),
        'error_smoothed': segment_approx_results.get('error_smoothed'),
        'diff_original_poly_smoothed': segment_approx_results.get('diff_original_poly_smoothed'),
        'diff_smoothed_poly_original': segment_approx_results.get('diff_smoothed_poly_original'),
    }
    # Filter out None values
    raw_error_maps = {k: v for k, v in raw_error_maps.items() if v is not None}

    if not raw_error_maps:
         print("No error maps generated for terminal segment. Cannot perform edge detection. Returning empty list.")
         return []


    # --- Combine and Threshold for Final Edge Map ---

    final_binary_edge_map = None

    if error_combination_strategy in ['max', 'weighted_sum']:
        # Normalize raw error maps over the segment for combination
        print("Normalizing error maps over the segment for final combination.")
        error_maps_for_combination = {}
        for key, error_map in raw_error_maps.items():
             error_maps_for_combination[key] = normalize_error_image(error_map)

        # Combine normalized error maps
        print(f"Combining normalized error maps using '{error_combination_strategy}' strategy for final edge map...")
        composite_error_map = combine_errors(
            error_maps_for_combination,
            strategy=error_combination_strategy,
            weights=error_combination_weights
        )
        # Apply final threshold to the composite map
        print(f"Applying {edge_threshold_type} thresholding to final composite map...")
        # Ensure skimage is available if using Otsu
        if edge_threshold_type == 'otsu' and not _skimage_available:
             print("Error: scikit-image is required for Otsu thresholding for final edge map. Exiting.")
             return [] # Cannot proceed if skimage is required but not available

        final_binary_edge_map = apply_threshold(
            composite_error_map,
            threshold_type=edge_threshold_type,
            fixed_threshold=fixed_edge_threshold
        )
        print("Final thresholding complete.")

    elif error_combination_strategy in ['logical_and', 'logical_or']:
        # Apply threshold to individual raw error maps first
        print(f"Applying {edge_threshold_type} thresholding to individual raw maps for final logical operation...")
        binary_maps_for_logical_op = []
        # Ensure skimage is available if using Otsu
        if edge_threshold_type == 'otsu' and not _skimage_available:
             print("Error: scikit-image is required for Otsu thresholding for final logical operation. Exiting.")
             return [] # Cannot proceed if skimage is required but not available

        for key in raw_error_maps.keys(): # Iterate through all raw error keys
             error_map_to_threshold = raw_error_maps[key]

             # --- Added check for constant image before applying Otsu ---
             if edge_threshold_type == 'otsu' and np.max(error_map_to_threshold) - np.min(error_map_to_threshold) < 1e-8:
                 print(f"Warning: Reassembled raw error map '{key}' is constant. Returning zero binary map for logical operation.")
                 binary_map = np.zeros_like(error_map_to_threshold, dtype=np.uint8)
             else:
                 # apply_threshold will handle normalization internally for raw input
                 binary_map = apply_threshold(
                     error_map_to_threshold,
                     threshold_type=edge_threshold_type,
                     fixed_threshold=fixed_edge_threshold
                 )
             # --- End of added check ---

             binary_maps_for_logical_op.append(binary_map)

        if not binary_maps_for_logical_op:
             print("No binary maps generated for final logical operation. Cannot generate final edge map. Returning empty list.")
             return []

        # Combine binary maps using logical AND or OR
        print(f"Combining binary maps using final logical '{error_combination_strategy.split('_')[-1].upper()}'...")
        final_binary_edge_map = binary_maps_for_logical_op[0]
        for i in range(1, len(binary_maps_for_logical_op)):
            if error_combination_strategy == 'logical_and':
                final_binary_edge_map = np.logical_and(final_binary_edge_map, binary_maps_for_logical_op[i]).astype(np.uint8)
            elif error_combination_strategy == 'logical_or':
                 final_binary_edge_map = np.logical_or(final_binary_edge_map, binary_maps_for_logical_op[i]).astype(np.uint8)
        print("Final logical operation complete.")

    else:
         print(f"Error: Unhandled error combination strategy '{error_combination_strategy}'. Returning empty list.")
         return []


    if final_binary_edge_map is None:
         print("Error: Failed to generate final binary edge map for terminal segment. Returning empty list.")
         return []

    # Return the binary edge map and its original pixel coordinates
    return [{
        'binary_edge_map': final_binary_edge_map,
        'pixel_coords': segment_pixel_coords
    }]


def reassemble_binary_edge_map(
    segment_results: List[SegmentResult],
    original_image_shape: Tuple[int, int]
) -> np.ndarray:
    """
    Reassembles binary edge maps from processed segments into a single full image edge map.
    """
    full_height, full_width = original_image_shape
    reassembled_edge_map = np.zeros(original_image_shape, dtype=np.uint8)

    for segment_result in segment_results:
        if 'binary_edge_map' in segment_result and 'pixel_coords' in segment_result:
            binary_edge_map = segment_result['binary_edge_map']
            x_start, y_start, x_end, y_end = segment_result['pixel_coords']

            # Ensure the segment edge map has the correct shape for the slice
            segment_height = y_end - y_start
            segment_width = x_end - x_start
            if binary_edge_map.shape == (segment_height, segment_width):
                reassembled_edge_map[y_start:y_end, x_start:x_end] = binary_edge_map
            else:
                print(f"Warning: Shape mismatch during reassembly. Expected ({segment_height}, {segment_width}), got {binary_edge_map.shape}. Skipping segment.")

    return reassembled_edge_map


if __name__ == "__main__":
    # The runtime SciPy check is no longer needed here due to the restructured imports
    # try:
    #     import scipy
    #     print(f"SciPy successfully imported at runtime. Version: {scipy.__version__}")
    # except ImportError:
    #     print("Error: SciPy failed to import at runtime.")
    #     print("Please ensure SciPy is installed in the Python environment used to run this script.")
    #     sys.exit(1)


    if not _imports_successful:
        print("Skipping adaptive edge detection due to core module import errors.")
        sys.exit(1)

    # Initialize variables used in dynamic threshold calculation to None or empty
    full_image_approx_results = None
    full_image_raw_error_maps = {}
    full_image_normalized_error_maps = {}
    full_image_composite_error_map = None # Initialized here
    full_image_binary_edge_map_for_band = None
    full_image_band_mask = None
    measure_values_on_full_image_in_band = None
    calculated_edge_quality_threshold = None # Also initialize the threshold


    # --- Configuration ---
    sample_image_path = PROJECT_ROOT / "images" / "spiral_and_zizag.png" # Replace with your image path

    # Create a dummy image file if the sample doesn't exist for demonstration
    if not sample_image_path.exists():
        print(f"Sample image not found at {sample_image_path}. Creating a dummy image.")
        dummy_img = np.zeros((256, 256), dtype=np.uint8) # Increased dummy size
        # Add a white square and a diagonal line
        dummy_img[64:192, 64:192] = 255
        for i in range(100):
            dummy_img[100 + i, 100 + i] = 128 # Gray diagonal line
        os.makedirs(PROJECT_ROOT / "images", exist_ok=True)
        Image.fromarray(dummy_img).save(sample_image_path)
        print(f"Dummy image created at {sample_image_path}")


    image_base_name = sample_image_path.stem
    IMAGE_RESULTS_DIR = SCRIPT_RESULTS_BASE_DIR / image_base_name
    os.makedirs(IMAGE_RESULTS_DIR, exist_ok=True)
    print(f"Saving results to: {IMAGE_RESULTS_DIR}")

    # Adaptive Parameters
    max_depth = 2             # Maximum recursion depth (0 means no subdivision)
    min_segment_size = 32     # Minimum segment dimension (width or height) to stop subdivision

    # Polynomial Approximation Parameters (applied per segment)
    poly_degree = 10
    nodes_method = 'leja'     # 'full_mesh', 'leja', 'fekete', 'padua'
    admissible_mesh_type = 'cheb' # 'cheb', 'uni' (only for 'leja', 'fekete', 'full_mesh')
    m_cheb = 2                # Parameter 'm' for Chebyshev mesh

    # Error Combination and Edge Thresholding Parameters (applied per segment)
    error_combination_strategy = 'max' # 'max', 'weighted_sum', 'logical_and', 'logical_or'
    # Weights for 'weighted_sum' (keys must match error map keys)
    error_combination_weights = {
        'error_original': 0.4,
        'error_smoothed': 0.4,
        'diff_original_poly_smoothed': 0.1,
        'diff_smoothed_poly_original': 0.1,
    }
     # Ensure weights sum to 1.0 if using weighted sum and you want the output in [0,1]
    if error_combination_strategy == 'weighted_sum':
        total_weight = sum(error_combination_weights.values())
        if abs(total_weight - 1.0) > 1e-6:
            print(f"Warning: Provided weights for 'weighted_sum' sum to {total_weight}. Normalizing weights.")
            weight_sum = sum(error_combination_weights.values())
            error_combination_weights = {k: v / weight_sum for k, v in error_combination_weights.items()}
            print(f"Normalized weights: {error_combination_weights}")


    edge_threshold_type = 'otsu' # 'fixed', 'otsu'
    fixed_edge_threshold = 0.5    # Only for 'fixed' threshold_type

    # Edge Quality Measure Parameters (M(S))
    # User can now choose the measure type
    edge_quality_measure_type = 'gradient_magnitude_near_edges' # Options: 'gradient_magnitude_near_edges', 'variance_near_edges', 'mean_abs_error_near_edges'
    # NOTE: 'gradient_magnitude_near_edges' requires SciPy.
    # If SciPy is not installed and this measure is chosen, the script will exit.

    # edge_quality_threshold = 0.01 # T_threshold for M(S) - This will now be calculated dynamically
    edge_quality_band_width = 5   # Width of the band in pixels for M(S) calculation
    quality_threshold_quantile = 0.6 # Quantile of the quality measure on the full image to use as threshold

    # --- Check if chosen edge quality measure is available ---
    if edge_quality_measure_type == 'gradient_magnitude_near_edges' and not _scipy_available:
        print(f"Error: Chosen edge quality measure '{edge_quality_measure_type}' requires SciPy, but SciPy is not available.")
        print("Please install SciPy (`pip install scipy`) or choose a different measure type ('variance_near_edges' or 'mean_abs_error_near_edges').")
        sys.exit(1)
    if edge_quality_measure_type in ['variance_near_edges', 'mean_abs_error_near_edges'] and not _skimage_available:
         print(f"Error: Chosen edge quality measure '{edge_quality_measure_type}' requires scikit-image, but scikit-image is not available.")
         print("Please install scikit-image (`pip install scikit-image`) or choose a different measure type.")
         sys.exit(1)

    print(f"Using '{edge_quality_measure_type}' as the edge quality measure.")


    # --- Load and Preprocess the Image ---
    try:
        original_image = load_and_preprocess_image(sample_image_path)
        original_image_shape = original_image.shape
        print(f"Original image shape: {original_image_shape}")
    except FileNotFoundError as e:
        print(f"Error loading image: {e}")
        sys.exit(1)

    # --- Calculate Dynamic Edge Quality Threshold ---
    print("\nCalculating dynamic edge quality threshold from initial full image analysis...")
    start_time_threshold = time.time()

    # Perform initial polynomial approximation on the full image
    full_image_region_min = np.array([0.0, 0.0], dtype=np.float32)
    full_image_region_max = np.array([1.0, 1.0], dtype=np.float32)
    # Removed compute_admissible_mesh call here, it's handled internally by image_poly_approximation_segment

    full_image_approx_results = image_poly_approximation_segment(
        image_segment=original_image,
        poly_degree=poly_degree,
        nodes_method=nodes_method,
        # Pass the full image region bounds as the rectangle
        rectangle=(full_image_region_min[0], full_image_region_min[1], full_image_region_max[0], full_image_region_max[1]),
        admissible_mesh_type=admissible_mesh_type, # Pass mesh type for internal node generation
        m_cheb=m_cheb # Pass m_cheb for internal node generation
        # Removed basis_Zc, basis_Zr, nodes_region_min, nodes_region_max, admissible_mesh
    )

    if not full_image_approx_results:
        print("Full image polynomial approximation failed. Cannot calculate dynamic threshold. Exiting.")
        sys.exit(1)

    # Extract raw error maps for the full image
    full_image_raw_error_maps = {
        'error_original': full_image_approx_results.get('error_original'),
        'error_smoothed': full_image_approx_results.get('error_smoothed'),
        'diff_original_poly_smoothed': full_image_approx_results.get('diff_original_poly_smoothed'),
        'diff_smoothed_poly_original': full_image_approx_results.get('diff_smoothed_poly_original'),
    }
    full_image_raw_error_maps = {k: v for k, v in full_image_raw_error_maps.items() if v is not None}

    if not full_image_raw_error_maps:
         print("No error maps generated for full image. Cannot calculate dynamic threshold. Exiting.")
         sys.exit(1)

    # Combine error maps for the full image using 'max' strategy
    full_image_normalized_error_maps = {}
    for key, error_map in full_image_raw_error_maps.items():
         full_image_normalized_error_maps[key] = normalize_error_image(error_map)

    full_image_composite_error_map = combine_errors(
        full_image_normalized_error_maps,
        strategy='max' # Using 'max' strategy for combining errors for threshold calculation
    )

    # Generate a binary map from the full image composite error map for the band
    # Ensure skimage is available before attempting to use apply_threshold with 'otsu' or get_band_around_edges
    if not _skimage_available:
        print("Error: scikit-image is required for dynamic threshold calculation (Otsu and band creation). Exiting.")
        sys.exit(1)

    full_image_binary_edge_map_for_band = apply_threshold(
        full_image_composite_error_map,
        threshold_type='otsu' # Using Otsu on the composite error map for band definition
    )

    # Get the band around edges in the full image composite error map
    # Ensure get_band_around_edges is available (requires skimage)
    if not _skimage_available:
        print("Error: scikit-image is required for get_band_around_edges for dynamic threshold. Exiting.")
        sys.exit(1)
    full_image_band_mask = get_band_around_edges(full_image_binary_edge_map_for_band, edge_quality_band_width)

    # Calculate the chosen quality measure values within the band on the full image composite error map
    # We need the *values* of the measure, not the single M(S) value yet.
    measure_values_on_full_image_in_band = None

    # --- Calculate local measure values within the band for quantile thresholding ---
    # This part needs to extract the relevant local values based on the chosen measure type
    # from the composite error map within the band mask.

    if edge_quality_measure_type == 'gradient_magnitude_near_edges':
        # Check if scipy is available before calling the function that uses it
        if not _scipy_available:
             # This case should ideally be handled by the initial check, but as a safeguard
             print(f"Error: scipy not available for '{edge_quality_measure_type}'. Cannot calculate dynamic threshold. Exiting.")
             sys.exit(1)
        # Calculate gradient magnitude of the full image composite error map
        full_image_gradient_magnitude = calculate_gradient_magnitude(full_image_composite_error_map)
        # Ensure gradient magnitude was calculated successfully (e.g., not None or empty due to internal scipy error)
        if full_image_gradient_magnitude is None or full_image_gradient_magnitude.size == 0:
             print(f"Error: Gradient magnitude calculation failed or returned empty result for dynamic threshold. Ensure scipy is installed correctly. Exiting.")
             sys.exit(1)
        # Get gradient magnitude values within the band mask
        measure_values_on_full_image_in_band = full_image_gradient_magnitude[full_image_band_mask > 0]

    elif edge_quality_measure_type == 'variance_near_edges':
        if not _skimage_available:
            print(f"Error: scikit-image not available for '{edge_quality_measure_type}'. Cannot calculate dynamic threshold. Exiting.")
            sys.exit(1)
        # Get error values within the band mask
        # Note: Calculating local variance for quantile thresholding is complex.
        # This implementation uses the raw error values within the band.
        measure_values_on_full_image_in_band = full_image_composite_error_map[full_image_band_mask > 0]


    elif edge_quality_measure_type == 'mean_abs_error_near_edges':
        if not _skimage_available:
            print(f"Error: scikit-image not available for '{edge_quality_measure_type}'. Cannot calculate dynamic threshold. Exiting.")
            sys.exit(1)
        # Get absolute error values within the band mask
        measure_values_on_full_image_in_band = np.abs(full_image_composite_error_map[full_image_band_mask > 0])

    else:
        print(f"Error: Invalid edge quality measure type '{edge_quality_measure_type}'. Cannot calculate dynamic threshold. Exiting.")
        sys.exit(1)

    # Calculate the specified quantile of these measure values to get the threshold
    if measure_values_on_full_image_in_band is None or measure_values_on_full_image_in_band.size == 0:
         print(f"No measure values found within the band on the full image for threshold calculation. Cannot calculate dynamic threshold. Exiting.")
         sys.exit(1)

    calculated_edge_quality_threshold = np.quantile(measure_values_on_full_image_in_band, quality_threshold_quantile)

    elapsed_time_threshold = time.time() - start_time_threshold
    print(f"Dynamic edge quality threshold calculated: {calculated_edge_quality_threshold:.4f} (using {quality_threshold_quantile*100:.0f}th percentile of local '{edge_quality_measure_type}' values in band on full image composite error map).")
    print(f"Threshold calculation completed in {elapsed_time_threshold:.4f} seconds.")


    # --- Start Adaptive Processing ---
    print("\nStarting adaptive edge detection with dynamic threshold...")
    start_time_adaptive = time.time()

    # Initial segment is the whole image
    initial_segment_data = original_image
    initial_segment_pixel_coords = (0, 0, original_image_shape[1], original_image_shape[0]) # (x_start, y_start, x_end, y_end)
    initial_segment_region_min = np.array([0.0, 0.0], dtype=np.float32)
    initial_segment_region_max = np.array([1.0, 1.0], dtype=np.float32)


    terminal_segment_results = process_segment_recursively(
        initial_segment_data,
        initial_segment_pixel_coords,
        initial_segment_region_min,
        initial_segment_region_max,
        original_image_shape,
        poly_degree,
        nodes_method,
        admissible_mesh_type,
        m_cheb,
        error_combination_strategy,
        error_combination_weights,
        edge_threshold_type,
        fixed_edge_threshold,
        edge_quality_measure_type, # Pass the determined quality measure type
        calculated_edge_quality_threshold, # Pass the calculated threshold
        edge_quality_band_width,
        current_depth=0,
        max_depth=max_depth,
        min_segment_size=min_segment_size
    )

    elapsed_time_adaptive = time.time() - start_time_adaptive
    print(f"\nAdaptive edge detection finished in {elapsed_time_adaptive:.4f} seconds.")
    print(f"Processed {len(terminal_segment_results)} terminal segments.")

    # --- Reassemble Final Edge Map ---
    if terminal_segment_results:
        print("\nReassembling final binary edge map...")
        final_edge_map = reassemble_binary_edge_map(terminal_segment_results, original_image_shape)
        print("Reassembly complete.")

        # --- Save Final Binary Edge Map ---
        print("\nSaving final binary edge map...")
        # Construct a descriptive filename
        filename_suffix = f"d{max_depth}_min{min_segment_size}_p{poly_degree}_{nodes_method}_err-{error_combination_strategy}_thresh-{edge_threshold_type}"
        if edge_threshold_type == 'fixed':
             filename_suffix += f"-{str(fixed_edge_threshold).replace('.', 'p')}"
        filename_suffix += f"_qual-{edge_quality_measure_type}_qthresh-auto{quality_threshold_quantile}_band{edge_quality_band_width}" # Indicate auto threshold


        edge_map_filename = f"{image_base_name}_adaptive_edge_map_{filename_suffix}.png"
        edge_map_filepath = os.path.join(str(IMAGE_RESULTS_DIR), edge_map_filename)

        try:
            # binary_edge_map is already uint8 (0 or 1)
            # Scale to 0-255 for standard grayscale PNG
            binary_edge_map_uint8 = (final_edge_map * 255).astype(np.uint8)
            Image.fromarray(binary_edge_map_uint8).save(edge_map_filepath)
            print(f"Saved final binary edge map to {edge_map_filepath}")
        except Exception as e:
            print(f"Error saving final binary edge map to {edge_map_filepath}: {e}")

        # --- Visualize Final Result ---
        print("\nGenerating final plot...")
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))

        # Plot Original Image
        axes[0].imshow(original_image, cmap='gray')
        axes[0].set_title('Original Image')
        axes[0].axis('off')

        # Plot Final Binary Edge Map
        axes[1].imshow(final_edge_map, cmap='gray')
        # Include calculated threshold in title, formatted to 4 decimal places
        axes[1].set_title(f'Adaptive Edge Map (Depth {max_depth}, Auto Quality Threshold {calculated_edge_quality_threshold:.4f})')
        axes[1].axis('off')

        plt.tight_layout()

        # --- Save the Plot ---\
        plot_filename = f"{image_base_name}_adaptive_edge_plot_{filename_suffix}.png"
        plot_filepath = os.path.join(str(IMAGE_RESULTS_DIR), plot_filename)

        try:
            plt.savefig(plot_filepath)
            print(f"\nSaved final plot to {plot_filepath}")
        except Exception as e:
            print(f"\nError saving final plot to {plot_filepath}: {e}")

        plt.close(fig)

    else:
        print("\nNo terminal segments processed. No final edge map to reassemble or save.")

    print("\nAdaptive edge detection script finished.")
