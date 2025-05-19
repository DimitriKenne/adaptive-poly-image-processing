import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
import time
from typing import Dict, List, Tuple, Union, Optional, Callable
import math # Import math for comb
import pickle # To save the results dictionary
import matplotlib.patches as patches # Import patches for drawing rectangles
import matplotlib.cm as cm # Import colormap functionality

# Add project root to Python path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Define the base path for the results folder
BASE_RESULTS_DIR = PROJECT_ROOT / "results"

# Define the subfolder for adaptive image reconstruction results
ADAPTIVE_RECONSTRUCTION_RESULTS_BASE_DIR = BASE_RESULTS_DIR / "adaptive_image_reconstruction_tests"

# Ensure the base results directory for adaptive reconstruction tests exists
os.makedirs(ADAPTIVE_RECONSTRUCTION_RESULTS_BASE_DIR, exist_ok=True)

# Import necessary functions from poly_approx
try:
    # image_poly_approximation_segment is now expected to return coefficients
    from poly_approx.image_poly_approximation import image_poly_approximation_segment, save_images
    # Import calculate_error_measure and normalize_error_image from image_reconstruction_metrics
    from poly_approx.image_reconstruction_metrics import calculate_error_measure, normalize_error_image
    # Import evaluate_polynomial_from_coeffs and gen_vanderm2d, graded_lexicographic_multi_indices
    from poly_approx.poly_projector import evaluate_polynomial_from_coeffs
    from poly_approx.polynomial_bases import gen_vanderm2d, graded_lexicographic_multi_indices


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
        height, width = image_segment.shape
        num_coeffs = int((poly_degree + 1) * (poly_degree + 2) / 2)
        dummy_coeffs = np.zeros(num_coeffs)
        return {
            'original_segment': np.zeros_like(image_segment),
            'smoothed_segment': np.zeros_like(image_segment),
            'approx_original': np.zeros_like(image_segment),
            'approx_smoothed': np.zeros_like(image_segment),
            'error_original': np.zeros_like(image_segment),
            'error_smoothed': np.zeros_like(image_segment),
            'diff_original_poly_smoothed': np.zeros_like(image_segment),
            'diff_smoothed_poly_original': np.zeros_like(image_segment),
            'computation_time': 0.0,
            'nodes_method': nodes_method,
            'poly_degree': poly_degree,
            'coefficients_original': dummy_coeffs,
            'coefficients_smoothed': dummy_coeffs
        }

    # Dummy save_images function
    def save_images(image_dict, save_folder):
        print("Dummy save_images called.")
        for filename in image_dict.keys():
            print(f"  Dummy saving: {filename} to {save_folder}")

    # Dummy calculate_error_measure function (returns a value > threshold to force subdivision in dummy mode)
    def calculate_error_measure(error_map, measure_type='mse'):
        print(f"Dummy calculate_error_measure called with measure: {measure_type}")
        return 1.0

    # Dummy normalize_error_image function for fallback
    def normalize_error_image(error_map, epsilon=1e-8):
         print("Dummy normalize_error_image called.")
         if error_map.size == 0:
             return np.zeros_like(error_map, dtype=np.float32)
         min_val = np.min(error_map)
         max_val = np.max(error_map)
         if max_val - min_val < epsilon:
             return np.zeros_like(error_map, dtype=np.float32)
         return ((error_map - min_val) / (max_val - min_val + epsilon)).astype(np.float32)

    # Dummy evaluate_polynomial_from_coeffs and a dummy basis function generator
    # We need a dummy gen_vanderm2d to provide a dummy basis function generator
    def dummy_basis_func_generator(point):
         # A dummy basis function that always returns an array of ones
         # The size should match the expected number of coefficients for a dummy poly_degree
         dummy_poly_degree = 5 # Assume a default degree for dummy
         num_coeffs = int((dummy_poly_degree + 1) * (dummy_poly_degree + 2) / 2)
         return np.ones(num_coeffs)

    def gen_vanderm2d(X, col=None, poly_basis=1, rectangle=None):
         print("Dummy gen_vanderm2d called to get basis function generator.")
         # Return a dummy Vandermonde matrix and a dummy basis function generator
         num_points = len(X) if X is not None else 1
         num_coeffs = col if col is not None else int((5 + 1) * (5 + 2) / 2) # Assume degree 5 for dummy
         dummy_V = np.zeros((num_points, num_coeffs))
         return dummy_V, dummy_basis_func_generator


    def evaluate_polynomial_from_coeffs(coeffs, basis_func_generator, eval_points):
        print("Dummy evaluate_polynomial_from_coeffs called.")
        eval_points = np.atleast_2d(eval_points)
        return np.zeros(eval_points.shape[0])

    def graded_lexicographic_multi_indices(total_terms):
        print("Dummy graded_lexicographic_multi_indices called.")
        return [(0,0)] * total_terms # Return dummy indices


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
    segment_id: str,
    poly_basis: int # Pass poly_basis to process_segment
):
    """
    Recursively processes an image segment for polynomial approximation and adaptive refinement.
    Stores polynomial coefficients, basis type, segment bounding boxes, and final error measure
    for terminal segments.

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
    poly_basis : int
        Polynomial basis used for approximation.
    """
    row_start, row_end, col_start, col_end = segment_bbox
    segment_image = original_image[row_start:row_end, col_start:col_end]

    height, width = segment_image.shape

    print(f"\n--- Processing segment {segment_id} at depth {current_depth} with shape {segment_image.shape} ---")

    if height == 0 or width == 0:
        print(f"Skipping empty segment {segment_id}.")
        return # Skip empty segments

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
        # This function is now expected to return coefficients as well
        approximation_results = image_poly_approximation_segment(
            image_segment=segment_image,
            rectangle=segment_rectangle, # Pass the rectangle tuple
            poly_degree=poly_degree,
            nodes_method=nodes_method,
            admissible_mesh_type=admissible_mesh_type, # Pass mesh type for internal mesh generation
            m_cheb=m_cheb, # m_cheb is still needed for internal mesh generation if applicable
            poly_basis=poly_basis # Pass poly_basis
        )
        # Extract raw error maps and polynomial coefficients
        raw_error_maps = {
            'error_original': approximation_results.get('error_original', np.zeros_like(segment_image)),
            'error_smoothed': approximation_results.get('error_smoothed', np.zeros_like(segment_image)),
            'diff_original_poly_smoothed': approximation_results.get('diff_original_poly_smoothed', np.zeros_like(segment_image)),
            'diff_smoothed_poly_original': approximation_results.get('diff_smoothed_poly_original', np.zeros_like(segment_image))
        }
        # We will use the coefficients from the smoothed approximation for reconstruction and error evaluation
        polynomial_coefficients = approximation_results.get('coefficients_smoothed', np.array([])) # Get the smoothed coefficients
        # segment_approx_image = approximation_results.get('approx_smoothed', np.zeros_like(segment_image)) # Not needed for recursive step

        print(f"Approximation complete for segment {segment_id}.")

    except Exception as e:
        print(f"Error during polynomial approximation for segment {segment_id}: {e}")
        print(f"Stopping processing for segment {segment_id}.")
        # Store empty coefficients and status for this segment if approximation fails
        results_dict[segment_id] = {
            'bbox': segment_bbox,
            'coefficients': np.array([]), # Store empty coefficients
            'depth': current_depth,
            'poly_degree': poly_degree, # Store poly degree used
            'poly_basis': poly_basis, # Store poly basis used
            'rectangle': segment_rectangle, # Store the segment rectangle
            'final_error_measure': -1.0, # Indicate error during approximation
            'status': 'approximation_failed'
        }
        return


    # --- Step 3: Evaluate Reconstruction Quality (Compute M(S)) ---
    # Calculate the error measure M(S) on a chosen RAW error map to guide subdivision.
    # Let's use the raw 'error_original' map for M(S) to be more sensitive to original detail.
    error_map_for_measure = raw_error_maps.get('error_original')

    if error_map_for_measure is None:
         print(f"Error: Could not get 'error_original' map for error measure in segment {segment_id}.")
         print(f"Stopping processing for segment {segment_id}.")
         results_dict[segment_id] = {
             'bbox': segment_bbox,
             'coefficients': polynomial_coefficients, # Store the coefficients
             'depth': current_depth,
             'poly_degree': poly_degree, # Store poly degree used
             'poly_basis': poly_basis, # Store poly basis used
             'rectangle': segment_rectangle, # Store the segment rectangle
             'final_error_measure': -1.0, # Indicate error during measure calculation
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
            'coefficients': polynomial_coefficients, # Store the coefficients
            'depth': current_depth,
            'poly_degree': poly_degree, # Store poly degree used
            'poly_basis': poly_basis, # Store poly basis used
            'rectangle': segment_rectangle, # Store the segment rectangle
            'final_error_measure': -1.0, # Indicate error during measure calculation
            'status': 'error_measure_failed'
        }
        return
    except Exception as e:
        print(f"An unexpected error occurred calculating error measure for segment {segment_id}: {e}")
        print(f"Stopping processing for segment {segment_id}.")
        results_dict[segment_id] = {
            'bbox': segment_bbox,
            'coefficients': polynomial_coefficients, # Store the coefficients
            'depth': current_depth,
            'poly_degree': poly_degree, # Store poly degree used
            'poly_basis': poly_basis, # Store poly basis used
            'rectangle': segment_rectangle, # Store the segment rectangle
            'final_error_measure': -1.0, # Indicate error during measure calculation
            'status': 'error_measure_failed'
        }
        return


    # --- Step 4: Decision and Refinement ---
    # Stopping criterion met: Either error is low enough, max depth reached, or segment is too small.
    if segment_error_measure <= error_threshold:
        print(f"Stopping for segment {segment_id}: Error measure {segment_error_measure:.6f} <= {error_threshold}.")

        # Store the segment bbox, coefficients, and final error measure for this final segment
        results_dict[segment_id] = {
            'bbox': segment_bbox,
            'coefficients': polynomial_coefficients, # Store the coefficients
            'depth': current_depth,
            'poly_degree': poly_degree, # Store poly degree used
            'poly_basis': poly_basis, # Store poly basis used
            'rectangle': segment_rectangle, # Store the segment rectangle
            'final_error_measure': segment_error_measure, # Store the final error measure
            'status': 'terminated_by_error' # Indicate termination by error
        }

    elif current_depth >= max_depth:
         print(f"Max depth ({max_depth}) reached for segment {segment_id}. Stopping recursion.")
         # Store the segment bbox, coefficients, and final error measure for this final segment
         results_dict[segment_id] = {
             'bbox': segment_bbox,
             'coefficients': polynomial_coefficients, # Store the coefficients
             'depth': current_depth,
             'poly_degree': poly_degree, # Store poly degree used
             'poly_basis': poly_basis, # Store poly basis used
             'rectangle': segment_rectangle, # Store the segment rectangle
             'final_error_measure': segment_error_measure, # Store the final error measure
             'status': 'terminated_by_depth' # Indicate termination by depth
         }

    elif height <= min_segment_size or width <= min_segment_size:
         print(f"Segment size ({width}x{height}) below minimum ({min_segment_size}) for segment {segment_id}. Stopping recursion.")
         # Store the segment bbox, coefficients, and final error measure for this final segment
         results_dict[segment_id] = {
             'bbox': segment_bbox,
             'coefficients': polynomial_coefficients, # Store the coefficients
             'depth': current_depth,
             'poly_degree': poly_degree, # Store poly degree used
             'poly_basis': poly_basis, # Store poly basis used
             'rectangle': segment_rectangle, # Store the segment rectangle
             'final_error_measure': segment_error_measure, # Store the final error measure
             'status': 'terminated_by_size' # Indicate termination by size
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
        if row_end > mid_row and col_end > mid_col:
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
                m_cheb,
                sigma,
                error_measure_type,
                error_threshold,
                max_depth,
                current_depth + 1,
                min_segment_size,
                results_dict,
                sub_segment_id,
                poly_basis # Pass poly_basis to recursive call
            )


def reassemble_results_from_coefficients(
    image_shape: Tuple[int, int],
    segment_results: Dict
) -> np.ndarray:
    """
    Reassembles the approximation image from stored segment bounding boxes,
    polynomial coefficients, degree, basis type, and rectangle.

    Parameters:
    -----------
    image_shape : Tuple[int, int]
        The shape of the original image (height, width).
    segment_results : Dict
        Dictionary containing results from each final segment, including bbox,
        coefficients, poly_degree, poly_basis, and rectangle.

    Returns:
    --------
    np.ndarray
        The reassembled approximation image.
    """
    height, width = image_shape
    reassembled_approx = np.zeros(image_shape, dtype=np.float32)

    original_height, original_width = image_shape

    for segment_id, results in segment_results.items():
        if 'bbox' in results and 'coefficients' in results and 'poly_degree' in results and 'poly_basis' in results and 'rectangle' in results:
            row_start, row_end, col_start, col_end = results['bbox']
            coefficients = results['coefficients']
            poly_degree = results['poly_degree']
            poly_basis = results['poly_basis'] # Get the poly_basis used for this segment
            segment_rectangle = results['rectangle'] # Get the segment rectangle

            seg_height = row_end - row_start
            seg_width = col_end - col_start

            # Ensure bbox coordinates are within image bounds
            if row_start < 0 or row_end > height or col_start < 0 or col_end > width:
                 print(f"Warning: Segment {segment_id} bbox [{row_start}:{row_end}, {col_start}:{col_end}] is out of original image bounds {image_shape}. Skipping reassembly for this segment.")
                 continue # Skip this segment if bbox is invalid

            if coefficients.size == 0:
                 # print(f"Warning: No coefficients found for segment {segment_id}. Filling with zeros.")
                 # Segment was likely terminated due to approximation failure, it's already zero in reassembled_approx
                 continue # Skip to next segment if no coefficients


            # Define the rectangle for this segment relative to the original image [0,1]x[0,1]
            xmin, ymin, xmax, ymax = segment_rectangle

            # Create a grid of points over the segment's pixel coordinates [0, width-1] x [0, height-1]
            # scaled to the rectangle [xmin, ymin] to [xmax, ymax] for evaluation.
            # The evaluation grid points should be in the same domain as the input 'points' for the polynomial function.
            x_eval_scaled = np.linspace(xmin, xmax, seg_width)
            y_eval_scaled = np.linspace(ymin, ymax, seg_height)
            XX_eval_scaled, YY_eval_scaled = np.meshgrid(x_eval_scaled, y_eval_scaled)
            evaluation_points_scaled = np.vstack([XX_eval_scaled.ravel(), YY_eval_scaled.ravel()]).T

            # Recreate the basis function generator 'p' using the stored poly_basis, poly_degree, and rectangle.
            # This uses the logic from gen_vanderm2d without needing actual nodes or func_values.
            try:
                # Determine the dimension based on the polynomial degree
                poly_dimension = int((poly_degree + 1) * (poly_degree + 2) / 2)
                # Get the multi-indices for this degree
                multi_indices = graded_lexicographic_multi_indices(poly_dimension)

                # Recreate the basis function generator based on the stored poly_basis
                # We need a dummy X for gen_vanderm2d, its value doesn't matter for getting 'p'
                dummy_X = np.zeros((1, 2)) # Just needs to be a 2D array
                _, basis_func_generator = gen_vanderm2d(
                    X=dummy_X, # Dummy nodes
                    col=poly_dimension, # Use the dimension corresponding to the degree
                    poly_basis=poly_basis, # Use the stored poly_basis
                    rectangle=segment_rectangle # Use the segment's rectangle
                )

                # Now use the imported evaluate_polynomial_from_coeffs with the recreated basis function generator
                approx_flat = evaluate_polynomial_from_coeffs(
                    coefficients,
                    basis_func_generator, # Use the recreated basis function generator
                    evaluation_points_scaled
                ).real # Take real part just in case


                # Reshape and clip
                reassembled_approx[row_start:row_end, col_start:col_end] = np.clip(approx_flat.reshape(seg_height, seg_width), 0, 1)

            except Exception as e:
                 print(f"Error evaluating polynomial for segment {segment_id}: {e}")
                 # Fill this segment region with zeros in the reassembled image
                 reassembled_approx[row_start:row_end, col_start:col_end] = np.zeros((seg_height, seg_width), dtype=np.float32)


    return reassembled_approx

def draw_segmentation_boundaries(image: np.ndarray, segment_results: Dict, ax: plt.Axes, color='red', linewidth=1):
    """
    Draws the bounding boxes of the terminal segments on an image plot.

    Parameters:
    -----------
    image : np.ndarray
        The image to draw on (e.g., the original or reconstructed image).
    segment_results : Dict
        Dictionary containing results from each final segment, including bbox.
    ax : plt.Axes
        The matplotlib Axes object to draw the rectangles on.
    color : str, optional
        The color of the boundary lines. Default is 'red'.
    linewidth : int, optional
        The width of the boundary lines. Default is 1.
    """
    # Display the image first
    ax.imshow(image, cmap='gray', vmin=0, vmax=1)
    ax.set_title("Reconstructed Image with Segmentation Boundaries")
    ax.axis('off')

    for segment_id, results in segment_results.items():
        if 'bbox' in results:
            row_start, row_end, col_start, col_end = results['bbox']

            # Create a Rectangle patch
            # Rectangle takes (x, y) as the lower left corner, width, height
            # The bbox is (row_start, row_end, col_start, col_end)
            # So, x = col_start, y = row_start, width = col_end - col_start, height = row_end - row_start
            rect = patches.Rectangle(
                (col_start, row_start), # (x, y) of lower left corner
                col_end - col_start,    # width
                row_end - row_start,    # height
                linewidth=linewidth,
                edgecolor=color,
                facecolor='none' # No fill
            )

            # Add the patch to the Axes
            ax.add_patch(rect)

def draw_segment_error_heatmap(image_shape: Tuple[int, int], segment_results: Dict, ax: plt.Axes, colormap='viridis'):
    """
    Draws a color-coded heatmap of the segmentation, where each segment's color
    intensity reflects its final error measure.

    Parameters:
    -----------
    image_shape : Tuple[int, int]
        The shape of the original image (height, width).
    segment_results : Dict
        Dictionary containing results from each final segment, including bbox and final_error_measure.
    ax : plt.Axes
        The matplotlib Axes object to draw the heatmap on.
    colormap : str, optional
        The colormap to use for visualizing error. Default is 'viridis'.
    """
    height, width = image_shape
    error_heatmap_image = np.zeros(image_shape, dtype=np.float32)

    # Collect all final error measures from segments that terminated by error or depth/size
    # Exclude segments that failed approximation or had missing error measure input
    final_errors = [
        results['final_error_measure'] for results in segment_results.values()
        if 'final_error_measure' in results and results['final_error_measure'] >= 0
    ]

    if not final_errors:
        print("No valid final error measures found in segment results. Cannot generate segment error heatmap.")
        ax.set_title("Segment Error Heatmap (No data)")
        ax.axis('off')
        return

    # Normalize the final error measures to the range [0, 1]
    min_error = np.min(final_errors)
    max_error = np.max(final_errors)
    epsilon = 1e-8 # For stability
    normalized_errors = (final_errors - min_error) / (max_error - min_error + epsilon)

    # Create a mapping from segment_id to its normalized error
    segment_normalized_error = {
        segment_id: (results['final_error_measure'] - min_error) / (max_error - min_error + epsilon)
        for segment_id, results in segment_results.items()
        if 'final_error_measure' in results and results['final_error_measure'] >= 0
    }

    # Fill the heatmap image with normalized error values for each segment
    for segment_id, results in segment_results.items():
        if 'bbox' in results and segment_id in segment_normalized_error:
            row_start, row_end, col_start, col_end = results['bbox']
            normalized_err = segment_normalized_error[segment_id]

            # Fill the segment area with the normalized error value
            error_heatmap_image[row_start:row_end, col_start:col_end] = normalized_err

    # Display the heatmap image
    im = ax.imshow(error_heatmap_image, cmap=colormap, origin='upper')
    fig = ax.get_figure() # Get the figure to add a colorbar
    fig.colorbar(im, ax=ax, label=f'Final Segment Error ({error_measure_type}, Normalized)')
    ax.set_title(f'Segment Error Heatmap ({error_measure_type})')
    ax.axis('off')


if __name__ == "__main__":
    if not _poly_approx_available:
        print("Skipping adaptive image reconstruction due to missing poly_approx modules.")
        sys.exit(1)

    # --- Configuration ---
    image_name = "spiral_and_zigzag" # Base name of the image (e.g., "spiral.png")
    image_filename = f"{image_name}.png"
    image_path = PROJECT_ROOT / "images" / image_filename # Assuming images are in a 'images' subfolder

    # Create a dummy image file if the sample doesn't exist for demonstration
    if not image_path.exists():
        print(f"Sample image not found at {image_path}. Creating a dummy image.")
        dummy_img = np.zeros((256, 256), dtype=np.uint8)
        # Add a white square
        dummy_img[50:150, 50:150] = 255
        # Add a gradient
        for i in range(256):
            dummy_img[i, :] = i
        # Ensure the 'images' directory exists if saving dummy there
        os.makedirs(PROJECT_ROOT / "images", exist_ok=True)
        Image.fromarray(dummy_img).save(image_path)
        print(f"Dummy image created at {image_path}")


    # Parameters for polynomial approximation within segments
    poly_degree = 5 # Increased default poly degree slightly
    nodes_method = 'leja' # or 'fekete', 'padua', 'full_mesh'
    admissible_mesh_type = 'cheb' # Options: 'cheb', 'uni'
    m_cheb = 2 # Parameter 'm' for Chebyshev mesh construction (m > 1).
    sigma = 1.0 # Standard deviation for Gaussian smoothing
    poly_basis_used = 1 # Assuming Shifted Monomials (basis 1) were used in approximation

    # Parameters for adaptive segmentation control
    error_measure_type = 'mse' # 'mse', 'mae', 'rmse'
    error_threshold = 0.001 # Threshold for the error measure M(S) - Adjusted for potentially higher degree
    max_depth = 5 # Maximum recursive segmentation depth (0 is the whole image). Increased max depth
    min_segment_size = 8 # Increased minimum segment dimension to avoid very small segments

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

    # Dictionary to store results (bbox, coefficients, poly_degree, poly_basis, rectangle, final_error_measure) from final segments
    # This dictionary structure is designed to be saved for compression analysis and visualization
    final_segment_data_for_compression: Dict[str, Dict] = {}

    print(f"\nStarting adaptive image reconstruction for {image_filename}...")
    start_time = time.time()

    # Start the recursive processing from the root segment
    process_segment(
        original_image,
        initial_bbox,
        poly_degree,
        nodes_method,
        admissible_mesh_type,
        m_cheb,
        sigma,
        error_measure_type,
        error_threshold,
        max_depth,
        0, # Start at depth 0
        min_segment_size,
        final_segment_data_for_compression, # Pass the dictionary to store results
        initial_segment_id,
        poly_basis_used # Pass poly_basis to the recursive function
    )

    end_time = time.time()
    print(f"\nAdaptive image reconstruction finished in {end_time - start_time:.4f} seconds.")
    print(f"Processed {len(final_segment_data_for_compression)} final segments.")

    # --- Save the segment data and coefficients for compression analysis ---
    # Create a subfolder for this specific image's results within the adaptive tests directory
    IMAGE_ADAPTIVE_RECONSTRUCTION_RESULTS_DIR = ADAPTIVE_RECONSTRUCTION_RESULTS_BASE_DIR / image_name
    os.makedirs(IMAGE_ADAPTIVE_RECONSTRUCTION_RESULTS_DIR, exist_ok=True)
    print(f"Saving segment data and coefficients to: {IMAGE_ADAPTIVE_RECONSTRUCTION_RESULTS_DIR}")

    # Construct a filename suffix based on parameters
    params_suffix = (
        f"deg{poly_degree}_{nodes_method}"
        f"_sigma{sigma:.1f}"
        f"_measure-{error_measure_type}_errthresh{str(error_threshold).replace('.', 'p')}_depth{max_depth}_min{min_segment_size}"
        f"_basis{poly_basis_used}" # Add basis to filename
    )

    # Filename for the saved data
    data_filename = f"{image_name}_segment_data_{params_suffix}.pkl" # Using pickle for simplicity
    data_filepath = os.path.join(str(IMAGE_ADAPTIVE_RECONSTRUCTION_RESULTS_DIR), data_filename)

    try:
        with open(data_filepath, 'wb') as f:
            pickle.dump(final_segment_data_for_compression, f)
        print(f"Saved segment data and coefficients to {data_filepath}")
        print("This file contains the bounding boxes, polynomial degree, basis type, and coefficients for each terminal segment.")
        print("It can be used as the 'compressed' representation for compression analysis.")
    except Exception as e:
        print(f"Error saving segment data and coefficients to {data_filepath}: {e}")


    # --- Reassemble the final results from the stored coefficients for visualization ---
    # This step acts as a 'decoding' process to reconstruct the image from the saved data.
    print("\nReassembling final approximation image from coefficients...")
    if final_segment_data_for_compression:
        # reassemble_results_from_coefficients now uses the basis stored per segment
        reassembled_approx_image = reassemble_results_from_coefficients(
            original_image_shape,
            final_segment_data_for_compression
        )
        print("Reassembly from coefficients complete.")

        # --- Calculate the actual reconstruction error on the reassembled image ---
        actual_reconstruction_error_map = np.abs(original_image - reassembled_approx_image)

        # Normalize the actual reconstruction error for visualization
        normalized_reconstruction_error_for_plot = normalize_error_image(actual_reconstruction_error_map)
        print("Actual reconstruction error map calculated and normalized.")

        # --- Save the reassembled approximation image and error map for visualization ---
        print("\nSaving reassembled approximation image and actual error map...")

        # Save the reassembled approximation image (already in [0,1] range)
        try:
            save_images({f"{image_name}_approx_from_coeffs_{params_suffix}.png": reassembled_approx_image}, str(IMAGE_ADAPTIVE_RECONSTRUCTION_RESULTS_DIR))
        except Exception as e:
            print(f"Error saving reassembled approximation image: {e}")

        # Save a normalized version of the actual reconstruction error for visualization
        try:
            save_images({f"{image_name}_actual_error_viz_from_coeffs_{params_suffix}.png": normalized_reconstruction_error_for_plot}, str(IMAGE_ADAPTIVE_RECONSTRUCTION_RESULTS_DIR))
        except Exception as e:
            print(f"Error saving actual error visualization image: {e}")

        # --- Visualize Main Results (Original, Reconstructed, Error, Segmentation) ---
        print("\nGenerating main plot...")
        num_subplots = 4
        fig1, axes1 = plt.subplots(1, num_subplots, figsize=(6 * num_subplots, 6))

        # Plot 1: Original Image
        axes1[0].imshow(original_image, cmap='gray', vmin=0, vmax=1)
        axes1[0].set_title('Original Image')
        axes1[0].axis('off')

        # Plot 2: Approximate Image (Reconstruction from Coefficients)
        axes1[1].imshow(reassembled_approx_image, cmap='gray', vmin=0, vmax=1)
        axes1[1].set_title('Approximate Image (Reconstructed from Coeffs)')
        axes1[1].axis('off')

        # Plot 3: Actual Reconstruction Error Heatmap (Normalized)
        im1 = axes1[2].imshow(normalized_reconstruction_error_for_plot, cmap='viridis', origin='upper')
        fig1.colorbar(im1, ax=axes1[2], label='Actual Reconstruction Error (Normalized)')
        axes1[2].set_title('Actual Reconstruction Error Heatmap')
        axes1[2].axis('off')

        # Plot 4: Reconstructed Image with Segmentation Boundaries
        draw_segmentation_boundaries(reassembled_approx_image, final_segment_data_for_compression, axes1[3])

        plt.tight_layout()

        # --- Save the Main Plot ---
        plot_filename_main = f"{image_name}_adaptive_reconstruction_plot_with_segmentation_{params_suffix}.png"
        plot_filepath_main = os.path.join(str(IMAGE_ADAPTIVE_RECONSTRUCTION_RESULTS_DIR), plot_filename_main)

        try:
            plt.savefig(plot_filepath_main)
            print(f"\nSaved main adaptive image reconstruction plot to {plot_filepath_main}")
        except Exception as e:
            print(f"\nError saving main adaptive image reconstruction plot to {plot_filepath_main}: {e}")

        # Close the main plot figure
        plt.close(fig1)


        # --- Visualize Segment Error Heatmap ---
        print("\nGenerating segment error heatmap plot...")
        fig2, ax2 = plt.subplots(1, 1, figsize=(8, 8)) # Create a new figure for this plot

        draw_segment_error_heatmap(original_image_shape, final_segment_data_for_compression, ax2, colormap='hot') # Use 'hot' colormap for errors

        plt.tight_layout()

        # --- Save the Segment Error Heatmap Plot ---
        plot_filename_segment_error = f"{image_name}_segment_error_heatmap_{params_suffix}.png"
        plot_filepath_segment_error = os.path.join(str(IMAGE_ADAPTIVE_RECONSTRUCTION_RESULTS_DIR), plot_filename_segment_error)

        try:
            plt.savefig(plot_filepath_segment_error)
            print(f"\nSaved segment error heatmap plot to {plot_filepath_segment_error}")
        except Exception as e:
            print(f"\nError saving segment error heatmap plot to {plot_filepath_segment_error}: {e}")

        # Close the segment error heatmap figure
        plt.close(fig2)


    else:
        print("\nNo segments were successfully processed. Cannot reassemble, plot, or save.")


    print("\nAdaptive image reconstruction script finished.")

