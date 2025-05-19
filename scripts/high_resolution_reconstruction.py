import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
import time
from typing import Dict, List, Tuple, Union, Optional, Callable
import pickle # To load the results dictionary
# import matplotlib.patches as patches # No longer needed as we are not plotting boundaries
# import matplotlib.cm as cm # No longer needed for segment error heatmap

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
    # Import evaluate_polynomial_from_coeffs and gen_vanderm2d, graded_lexicographic_multi_indices
    from poly_approx.poly_projector import evaluate_polynomial_from_coeffs
    from poly_approx.polynomial_bases import gen_vanderm2d, graded_lexicographic_multi_indices

    _poly_approx_available = True
except ImportError as e:
    print(f"Could not import necessary modules from poly_approx for high-resolution reconstruction: {e}")
    print("Please ensure your project structure is correct.")
    print("Using dummy functions. High-resolution reconstruction will be skipped.")
    _poly_approx_available = False

    # Define dummy functions if imports fail
    def dummy_basis_func_generator(point):
         dummy_poly_degree = 5 # Assume a default degree for dummy
         num_coeffs = int((dummy_poly_degree + 1) * (dummy_poly_degree + 2) / 2)
         return np.ones(num_coeffs)

    def gen_vanderm2d(X, col=None, poly_basis=1, rectangle=None):
         print("Dummy gen_vanderm2d called.")
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


def evaluate_polynomial_for_segment(
    segment_data: Dict,
    evaluation_points_scaled: np.ndarray
) -> np.ndarray:
    """
    Evaluates the polynomial for a single segment at given points.

    Parameters:
    -----------
    segment_data : Dict
        Dictionary containing segment data: coefficients, poly_degree, poly_basis, rectangle.
    evaluation_points_scaled : np.ndarray
        Points to evaluate the polynomial at, scaled to the segment's rectangle domain.

    Returns:
    --------
    np.ndarray
        The polynomial values evaluated at the given points.
    """
    coefficients = segment_data['coefficients']
    poly_degree = segment_data['poly_degree']
    poly_basis = segment_data['poly_basis']
    segment_rectangle = segment_data['rectangle']

    # Recreate the basis function generator 'p' using the stored poly_basis, poly_degree, and rectangle.
    try:
        poly_dimension = int((poly_degree + 1) * (poly_degree + 2) / 2)
        multi_indices = graded_lexicographic_multi_indices(poly_dimension)
        dummy_X = np.zeros((1, 2)) # Dummy nodes
        _, basis_func_generator = gen_vanderm2d(
            X=dummy_X,
            col=poly_dimension,
            poly_basis=poly_basis,
            rectangle=segment_rectangle
        )

        # Use the imported evaluate_polynomial_from_coeffs
        # evaluate_polynomial_from_coeffs is designed to handle both single points and arrays of points
        approx_flat = evaluate_polynomial_from_coeffs(
            coefficients,
            basis_func_generator,
            evaluation_points_scaled
        ).real # Take real part just in case

        return approx_flat

    except Exception as e:
        print(f"Error evaluating polynomial for segment: {e}")
        # Return zeros with the expected shape based on evaluation_points_scaled
        return np.zeros(evaluation_points_scaled.shape[0])


def reassemble_high_resolution(
    original_image_shape: Tuple[int, int],
    segment_results: Dict,
    high_res_shape: Tuple[int, int],
    enable_blending: bool = False,
    blend_width: int = 4 # Blend width in terms of ORIGINAL image pixels
) -> np.ndarray:
    """
    Reassembles the approximation image at a higher resolution from stored segment data.

    Optimized version: Evaluates each segment's polynomial only over the relevant
    high-resolution pixels and uses vectorized weight calculation for blending.

    Parameters:
    -----------
    original_image_shape : Tuple[int, int]
        The shape of the original image (height, width).
    segment_results : Dict
        Dictionary containing results from each final segment, including bbox,
        coefficients, poly_degree, poly_basis, and rectangle.
    high_res_shape : Tuple[int, int]
        The desired shape of the high-resolution reconstructed image (height, width).
    enable_blending : bool, optional
        Whether to enable blending between segments. Default is False.
    blend_width : int, optional
        The width of the blending zone in pixels, interpreted relative to the
        ORIGINAL image resolution. Default is 4.

    Returns:
    --------
    np.ndarray
        The reassembled approximation image at the specified high resolution.
    """
    original_height, original_width = original_image_shape
    high_res_height, high_res_width = high_res_shape

    high_res_approx = np.zeros(high_res_shape, dtype=np.float32)
    total_weight_map = np.zeros(high_res_shape, dtype=np.float32) # For blending weights

    print(f"\nReassembling at high resolution: {high_res_shape}...")

    # Calculate the blend width in terms of high-resolution pixels
    # Ensure blend width is at least 1 pixel in high-res if blending is enabled
    high_res_blend_width_h = max(1, int(blend_width * high_res_width / original_width)) if enable_blending else 0
    high_res_blend_width_v = max(1, int(blend_width * high_res_height / original_height)) if enable_blending else 0


    # Iterate through each segment
    print("Processing segments for high-resolution reassembly...")
    for segment_id, results in segment_results.items():
        if 'bbox' in results and 'coefficients' in results and 'poly_degree' in results and 'poly_basis' in results and 'rectangle' in results:
            row_start, row_end, col_start, col_end = results['bbox']
            segment_rectangle = results['rectangle'] # Rectangle in [0,1]x[0,1] based on ORIGINAL image

            if results['coefficients'].size == 0:
                continue # Skip segments with no coefficients

            # Calculate the pixel coordinates of the segment's bounding box in the HIGH-RES image
            high_res_col_start = int(col_start / original_width * high_res_width)
            high_res_col_end = int(col_end / original_width * high_res_width)
            high_res_row_start = int(row_start / original_height * high_res_height)
            high_res_row_end = int(row_end / original_height * high_res_height)

            # Define the region of high-res pixels to evaluate this segment's polynomial over.
            # If blending is enabled, expand the region by the high-res blend width.
            if enable_blending:
                eval_row_start = max(0, high_res_row_start - high_res_blend_width_v)
                eval_row_end = min(high_res_height, high_res_row_end + high_res_blend_width_v)
                eval_col_start = max(0, high_res_col_start - high_res_blend_width_h)
                eval_col_end = min(high_res_width, high_res_col_end + high_res_blend_width_h)
            else:
                # No blending, evaluate only within the segment's high-res bbox
                eval_row_start = high_res_row_start
                eval_row_end = high_res_row_end
                eval_col_start = high_res_col_start
                eval_col_end = high_res_col_end

            # Ensure the evaluation region is valid
            if eval_row_end <= eval_row_start or eval_col_end <= eval_col_start:
                 # print(f"Segment {segment_id} has invalid evaluation region. Skipping.")
                 continue # Skip if region is invalid

            # Create a grid of high-res points within the evaluation region
            eval_cols = np.arange(eval_col_start, eval_col_end)
            eval_rows = np.arange(eval_row_start, eval_row_end)
            XX_eval, YY_eval = np.meshgrid(eval_cols, eval_rows)

            # Map these high-res pixel coordinates back to the segment's original [0,1]x[0,1] rectangle domain
            # The mapping should be based on the segment's original pixel dimensions and its rectangle.
            xmin, ymin, xmax, ymax = segment_rectangle
            seg_original_width = col_end - col_start
            seg_original_height = row_end - row_start
            epsilon = 1e-9
            if seg_original_width < epsilon: seg_original_width = epsilon
            if seg_original_height < epsilon: seg_original_height = epsilon

            # Map high-res pixel coords (XX_eval, YY_eval) to segment's [0,1]x[0,1] domain
            # This mapping needs to consider the segment's original pixel bounds and its rectangle in [0,1]x[0,1]
            # A point (r, c) in the high-res grid, within the segment's high-res bbox (high_res_row_start, high_res_col_start)
            # corresponds to a point in the original image at approximately
            # (r / high_res_height * original_height, c / high_res_width * original_width).
            # We then map this original image coordinate to the segment's [0,1]x[0,1] rectangle.

            # Map high-res pixel coordinates (XX_eval, YY_eval) to the original image pixel coordinates
            original_pixel_c = XX_eval.ravel() / high_res_width * original_width
            original_pixel_r = YY_eval.ravel() / high_res_height * original_height

            # Map original image pixel coordinates to the segment's [0,1]x[0,1] rectangle domain
            mapped_eval_points = np.vstack([
                xmin + (original_pixel_c - col_start) / seg_original_width * (xmax - xmin),
                ymin + (original_pixel_r - row_start) / seg_original_height * (ymax - ymin)
            ]).T


            # Evaluate the polynomial for the current segment over the evaluation region
            segment_eval_flat = evaluate_polynomial_for_segment(results, mapped_eval_points)
            segment_high_res_eval = segment_eval_flat.reshape(eval_row_end - eval_row_start, eval_col_end - eval_col_start)

            if enable_blending:
                # Calculate the weight mask for this segment over the evaluation region (vectorized)
                weight_mask = np.zeros_like(segment_high_res_eval, dtype=np.float32)

                # Calculate distances to the nearest edge of the segment's core (in HIGH-RES pixels)
                # within the evaluation region
                dist_h = np.minimum(XX_eval - high_res_col_start, high_res_col_end - 1 - XX_eval)
                dist_v = np.minimum(YY_eval - high_res_row_start, high_res_row_end - 1 - YY_eval)

                # Weight is 1 in the core, linearly decreases to 0 in the blend zone
                # Use high_res_blend_width_h/v for division
                weight_h = np.clip(dist_h / high_res_blend_width_h, 0.0, 1.0) if high_res_blend_width_h > 0 else (dist_h >= 0).astype(np.float32) # Handle blend_width = 0 case
                weight_v = np.clip(dist_v / high_res_blend_width_v, 0.0, 1.0) if high_res_blend_width_v > 0 else (dist_v >= 0).astype(np.float32) # Handle blend_width = 0 case


                # Combine horizontal and vertical weights (e.g., multiply)
                weights = weight_h * weight_v

                # Add weighted contribution to the high-res approximation and total weight map
                high_res_approx[eval_row_start:eval_row_end, eval_col_start:eval_col_end] += segment_high_res_eval * weights
                total_weight_map[eval_row_start:eval_row_end, eval_col_start:eval_col_end] += weights

            else:
                # No blending: assign the polynomial values directly within the segment's high-res bbox
                # Ensure we only assign within the actual segment bbox, not the expanded eval region
                # Need to be careful with slicing based on eval_row/col_start
                row_slice_start = high_res_row_start - eval_row_start
                row_slice_end = high_res_row_end - eval_row_start
                col_slice_start = high_res_col_start - eval_col_start
                col_slice_end = high_res_col_end - eval_col_start

                # Ensure slices are valid
                if row_slice_start >= 0 and row_slice_end <= segment_high_res_eval.shape[0] and \
                   col_slice_start >= 0 and col_slice_end <= segment_high_res_eval.shape[1]:

                    high_res_approx[high_res_row_start:high_res_row_end, high_res_col_start:high_res_col_end] = \
                        segment_high_res_eval[row_slice_start:row_slice_end, col_slice_start:col_slice_end]
                else:
                    print(f"Warning: Invalid slice calculated for segment {segment_id} during no-blending reassembly. Skipping assignment.")


    # Finalize blending by dividing by the total weight
    if enable_blending:
        # Avoid division by zero where no segments contributed
        total_weight_map[total_weight_map < 1e-9] = 1.0 # Set very small weights to 1 to avoid NaN/Inf
        high_res_approx /= total_weight_map


    # Clip the final high-res image to [0, 1]
    high_res_approx = np.clip(high_res_approx, 0, 1)

    return high_res_approx


if __name__ == "__main__":
    if not _poly_approx_available:
        print("Skipping high-resolution reconstruction due to missing poly_approx modules.")
        sys.exit(1)

    # --- Configuration ---
    image_name = "spiral_and_zigzag" # Base name of the image (e.g., "spiral.png")

    # Specify the parameters used for the adaptive reconstruction
    # These must match the suffix of the .pkl file you want to load
    poly_degree = 5
    nodes_method = 'leja'
    sigma = 1.0
    error_measure_type = 'mse'
    error_threshold = 0.001
    max_depth = 5
    min_segment_size = 8
    poly_basis_used = 1
    enable_blending_adaptive = False # Blending state during adaptive segmentation
    blend_width_adaptive = 4 # Blend width during adaptive segmentation

    # Construct the filename suffix for the data file to load
    params_suffix_load = (
        f"deg{poly_degree}_{nodes_method}"
        f"_sigma{sigma:.1f}"
        f"_measure-{error_measure_type}_errthresh{str(error_threshold).replace('.', 'p')}_depth{max_depth}_min{min_segment_size}"
        f"_basis{poly_basis_used}"
        f"_blend{blend_width_adaptive if enable_blending_adaptive else 'off'}"
    )

    data_filename = f"{image_name}_segment_data_{params_suffix_load}.pkl"
    data_filepath = os.path.join(str(ADAPTIVE_RECONSTRUCTION_RESULTS_BASE_DIR / image_name), data_filename)

    # High-Resolution Reconstruction Parameters
    high_res_factor = 2 # Factor to increase resolution (e.g., 2 means 2x width and 2x height)
    # Or specify exact high resolution:
    # high_res_shape = (512, 512) # Example: (height, width)

    # Blending parameters for HIGH-RESOLUTION reassembly
    enable_blending_high_res = True # Enable blending during high-res reassembly
    blend_width_high_res = blend_width_adaptive # Use the same blend width as adaptive (relative to original image)

    # --- Load the segment data ---
    print(f"Loading segment data from: {data_filepath}")
    if not os.path.exists(data_filepath):
        print(f"Error: Segment data file not found at {data_filepath}")
        sys.exit(1)

    try:
        with open(data_filepath, 'rb') as f:
            loaded_segment_data = pickle.load(f)
        print("Segment data loaded successfully.")
    except Exception as e:
        print(f"Error loading segment data from {data_filepath}: {e}")
        sys.exit(1)

    # Get the original image shape from the loaded data (assuming at least one segment exists)
    if not loaded_segment_data:
        print("Error: Loaded segment data is empty.")
        sys.exit(1)

    # Find the overall bounding box to determine original image shape
    min_row, min_col = float('inf'), float('inf')
    max_row, max_col = float('-inf'), float('-inf')
    for segment_id, results in loaded_segment_data.items():
        if 'bbox' in results:
            row_start, row_end, col_start, col_end = results['bbox']
            min_row = min(min_row, row_start)
            min_col = min(min_col, col_start)
            max_row = max(max_row, row_end)
            max_col = max(max_col, col_end)

    original_image_shape_from_data = (max_row - min_row, max_col - min_col)
    print(f"Inferred original image shape from segment data: {original_image_shape_from_data}")

    # Determine the desired high resolution shape
    # Using high_res_factor:
    high_res_shape = (
        int(original_image_shape_from_data[0] * high_res_factor),
        int(original_image_shape_from_data[1] * high_res_factor)
    )
    # If you prefer to specify exact resolution, uncomment the line above and comment the factor lines

    # --- Perform high-resolution reassembly ---
    start_time_high_res = time.time()
    high_res_approx_image = reassemble_high_resolution(
        original_image_shape_from_data,
        loaded_segment_data,
        high_res_shape,
        enable_blending=enable_blending_high_res,
        blend_width=blend_width_high_res # Blend width is relative to original image scale
    )
    end_time_high_res = time.time()
    print(f"\nHigh-resolution reassembly finished in {end_time_high_res - start_time_high_res:.4f} seconds.")

    # --- Save the high-resolution image ---
    # Create a subfolder for high-resolution results
    HIGH_RES_RESULTS_DIR = ADAPTIVE_RECONSTRUCTION_RESULTS_BASE_DIR / image_name / "high_resolution"
    os.makedirs(HIGH_RES_RESULTS_DIR, exist_ok=True)
    print(f"Saving high-resolution image to: {HIGH_RES_RESULTS_DIR}")

    # Construct a filename suffix for the high-res image
    high_res_params_suffix = (
        f"orig_deg{poly_degree}_{nodes_method}"
        f"_sigma{sigma:.1f}"
        f"_measure-{error_measure_type}_errthresh{str(error_threshold).replace('.', 'p')}_depth{max_depth}_min{min_segment_size}"
        f"_basis{poly_basis_used}"
        f"_orig_blend{blend_width_adaptive if enable_blending_adaptive else 'off'}"
        f"_highres{high_res_shape[0]}x{high_res_shape[1]}"
        f"_blend{blend_width_high_res if enable_blending_high_res else 'off'}" # Blending for high-res reassembly
    )

    high_res_filename = f"{image_name}_high_res_approx_{high_res_params_suffix}.png"
    high_res_filepath = os.path.join(str(HIGH_RES_RESULTS_DIR), high_res_filename)

    try:
        # Convert back to uint8 for saving as a standard image file
        Image.fromarray((high_res_approx_image * 255).astype(np.uint8)).save(high_res_filepath)
        print(f"Saved high-resolution approximation image to {high_res_filepath}")
    except Exception as e:
        print(f"Error saving high-resolution approximation image to {high_res_filepath}: {e}")

    # --- Removed Plotting Code ---
    # The original code included plotting the high-resolution image with segmentation boundaries.
    # This section has been removed as requested to only focus on saving the image.
    # If you need to visualize, you can add plotting code back here, but the request was to remove it.
    # Example of removed plotting code:
    # print("\nGenerating plot of high-resolution image with segmentation boundaries...")
    # fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    # # Need a function similar to draw_segmentation_boundaries_on_image but for high-res
    # # This would involve scaling the original segment bboxes to the high-res dimensions
    # # For now, we just show the image
    # ax.imshow(high_res_approx_image, cmap='gray', vmin=0, vmax=1)
    # ax.set_title(f'High-Resolution Approximate Image ({high_res_shape[0]}x{high_res_shape[1]})')
    # ax.axis('off')
    # plt.tight_layout()
    # # Save the plot if needed
    # # plot_filename_high_res = f"{image_name}_high_res_approx_plot_{high_res_params_suffix}.png"
    # # plot_filepath_high_res = os.path.join(str(HIGH_RES_RESULTS_DIR), plot_filename_high_res)
    # # try:
    # #     plt.savefig(plot_filepath_high_res)
    # #     print(f"Saved high-resolution approximation plot to {plot_filepath_high_res}")
    # # except Exception as e:
    # #     print(f"Error saving high-resolution approximation plot to {plot_filepath_high_res}: {e}")
    # plt.show() # Or plt.close(fig) if you don't want to display

    print("\nHigh-resolution reconstruction script finished.")

