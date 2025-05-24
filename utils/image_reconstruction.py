# image_reconstruction.py

import sys
import os
import time
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Union, Optional, Callable
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.cm as cm
from PIL import Image # Still needed for loading/saving images
import multiprocessing
from queue import Queue # Using queue for managing segments to process

# Determine the project root dynamically based on the location of this file
# If this file is in 'utils/', the project root is two levels up.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
# Add project root to Python path to allow importing 'config' and 'poly_approx'
sys.path.insert(0, str(PROJECT_ROOT))

# Import configuration from the 'config' directory
try:
    import config.config as config # Import config.py as config
except ImportError as e:
    print(f"Error importing config file: {e}. Please ensure your project structure is correct.")
    sys.exit(1)


# Import necessary functions from poly_approx package
# These imports assume 'poly_approx' is a package directly under PROJECT_ROOT
try:
    # image_poly_approximation_segment is now expected to return coefficients and all raw error maps
    from poly_approx.image_poly_approximation import image_poly_approximation_segment, save_images
    # Import calculate_error_measure and normalize_error_image from image_reconstruction_metrics
    from poly_approx.image_reconstruction_metrics import calculate_error_measure, normalize_error_image
    # Import evaluate_polynomial_from_coeffs and gen_vanderm2d from poly_projector
    from poly_approx.poly_projector import evaluate_polynomial_from_coeffs, gen_vanderm2d
    # Import graded_lexicographic_multi_indices from polynomial_bases (assuming it's in poly_approx)
    from poly_approx.polynomial_bases import graded_lexicographic_multi_indices
    # Import edge processing functions
    from poly_approx.edge_processing import combine_errors, apply_threshold


    _poly_approx_available = True
except ImportError as e:
    print(f"Could not import modules from poly_approx: {e}")
    print("Please ensure your project structure is correct and poly_approx is in your Python path.")
    print("Adaptive image reconstruction will be skipped.")
    _poly_approx_available = False

    # Define dummy functions if imports fail (keep these for robustness in case of import issues)
    def image_poly_approximation_segment(image_segment, rectangle, poly_degree, nodes_method, admissible_mesh_type, m_cheb, poly_basis):
        print("Dummy image_poly_approximation_segment called.")
        height, width = image_segment.shape
        num_coeffs = int((poly_degree + 1) * (poly_degree + 2) / 2)
        dummy_coeffs = np.zeros(num_coeffs)
        # Return dummy raw error maps as well
        dummy_error_map = np.zeros_like(image_segment)
        return {
            'original_segment': np.zeros_like(image_segment),
            'smoothed_segment': np.zeros_like(image_segment),
            'approx_original': np.zeros_like(image_segment),
            'approx_smoothed': np.zeros_like(image_segment),
            'error_original': dummy_error_map,
            'error_smoothed': dummy_error_map,
            'diff_original_poly_smoothed': dummy_error_map,
            'diff_smoothed_poly_original': dummy_error_map,
            'computation_time': 0.0,
            'nodes_method': nodes_method,
            'poly_degree': poly_degree,
            'coefficients_original': dummy_coeffs,
            'coefficients_smoothed': dummy_coeffs
        }

    def save_images(image_dict, save_folder):
        print("Dummy save_images called.")
        for filename in image_dict.keys():
            print(f"  Dummy saving: {filename} to {save_folder}")

    def calculate_error_measure(error_map, measure_type='mse'):
        print(f"Dummy calculate_error_measure called with measure: {measure_type}")
        return 1.0 # Return > threshold to force subdivision in dummy mode

    def normalize_error_image(error_map, epsilon=1e-8):
         print("Dummy normalize_error_image called.")
         if error_map.size == 0:
             return np.zeros_like(error_map, dtype=np.float32)
         min_val = np.min(error_map)
         max_val = np.max(error_map)
         if max_val - min_val < epsilon:
             return np.zeros_like(error_map, dtype=np.float32)
         return ((error_map - min_val) / (max_val - min_val + epsilon)).astype(np.float32)

    def dummy_basis_func_generator(point):
         dummy_poly_degree = 5
         num_coeffs = int((dummy_poly_degree + 1) * (dummy_poly_degree + 2) / 2)
         return np.ones(num_coeffs)

    def gen_vanderm2d(X, col=None, poly_basis=1, rectangle=None):
         print("Dummy gen_vanderm2d called.")
         num_points = len(X) if X is not None else 1
         num_coeffs = col if col is not None else int((5 + 1) * (5 + 2) / 2)
         dummy_V = np.zeros((num_points, num_coeffs))
         return dummy_V, dummy_basis_func_generator

    def evaluate_polynomial_from_coeffs(coeffs, basis_func_generator, eval_points):
        print("Dummy evaluate_polynomial_from_coeffs called.")
        eval_points = np.atleast_2d(eval_points)
        return np.zeros(eval_points.shape[0])

    def graded_lexicographic_multi_indices(total_terms):
        print("Dummy graded_lexicographic_multi_indices called.")
        return [(0,0)] * total_terms

    def combine_errors(error_dict, strategy='max', weights=None): # Removed logical_op_error_keys
        print(f"Dummy combine_errors called with strategy: {strategy}")
        if error_dict: return np.zeros_like(list(error_dict.values())[0], dtype=np.float32)
        return np.zeros((100, 100), dtype=np.float32)

    def apply_threshold(image, threshold_type='fixed', fixed_threshold=0.5):
        print(f"Dummy apply_threshold called with threshold_type: {threshold_type}")
        if image.size == 0: return np.zeros_like(image, dtype=np.uint8)
        return np.zeros_like(image, dtype=np.uint8)


# --- Helper function for multiprocessing pool initialization ---
# This initializer now does nothing as we pass data directly to the worker.
def _worker_init():
    """Initializer for worker processes (does nothing now)."""
    pass


# --- Worker function for multiprocessing ---
# This function runs in a separate process and processes a single segment task.
# It now accepts the image segment and rectangle directly.
def process_segment_worker(
    task: Tuple,
    image_segment: np.ndarray, # Pass the segment image directly
    segment_rectangle: Tuple[float, float, float, float], # Pass the rectangle directly
    config_params: Dict
) -> Tuple[str, Dict, Optional[List[Tuple]]]:
    """
    Worker function to process a single image segment.

    Args:
        task: A tuple containing (segment_bbox, current_depth, segment_id).
        image_segment: The image data for the current segment.
        segment_rectangle: The rectangle tuple for the current segment.
        config_params: Dictionary containing necessary configuration parameters.

    Returns:
        A tuple containing:
        - segment_id: The ID of the processed segment.
        - results_data: Dictionary containing results for this segment if it's terminal.
                        Empty dictionary if subdivision occurs.
        - sub_segment_tasks: List of new tasks (bbox, depth, id) if subdivision occurs,
                             otherwise None.
    """
    segment_bbox, current_depth, segment_id = task
    # We no longer need to slice the original image here, as image_segment is passed directly.
    # segment_image = original_image[row_start:row_end, col_start:col_end]

    height, width = image_segment.shape

    # print(f"\n--- Worker processing segment {segment_id} at depth {current_depth} with shape {segment_image.shape} ---") # Verbose worker output

    if height == 0 or width == 0:
        # print(f"Worker skipping empty segment {segment_id}.")
        return segment_id, {}, None # Skip empty segments

    # We no longer need to calculate segment_rectangle here, as it's passed directly.
    # original_height, original_width = original_image.shape
    # segment_rectangle = (
    #     col_start / original_width,
    #     row_start / original_height,
    #     col_end / original_width,
    #     row_end / original_height
    # )

    # --- Step 1 & 2: Polynomial Approximation and Raw Error Map Calculation ---
    try:
        # Use config_params dictionary to access configuration values
        approximation_results = image_poly_approximation_segment(
            image_segment=image_segment, # Use the passed segment image
            rectangle=segment_rectangle, # Use the passed rectangle
            poly_degree=config_params['POLY_DEGREE'],
            nodes_method=config_params['NODES_METHOD'],
            admissible_mesh_type=config_params['ADMISSIBLE_MESH_TYPE'],
            m_cheb=config_params['M_CHEB'],
            poly_basis=config_params['POLY_BASIS_USED']
        )
        # Collect all raw error maps
        raw_error_maps = {
            'error_original': approximation_results.get('error_original', np.zeros_like(image_segment)),
            'error_smoothed': approximation_results.get('error_smoothed', np.zeros_like(image_segment)),
            'diff_original_poly_smoothed': approximation_results.get('diff_original_poly_smoothed', np.zeros_like(image_segment)),
            'diff_smoothed_poly_original': approximation_results.get('diff_smoothed_poly_original', np.zeros_like(image_segment)),
        }
        # IMPORTANT CHANGE: Return coefficients_original for the main reconstruction
        polynomial_coefficients_for_reconstruction = approximation_results.get('coefficients_original', np.array([]))

        # print(f"Worker approximation complete for segment {segment_id}.") # Verbose worker output

    except Exception as e:
        print(f"Worker Error during polynomial approximation for segment {segment_id}: {e}")
        # Return results indicating failure
        results_data = {
            'bbox': segment_bbox,
            'coefficients': np.array([]), # Use empty array on failure
            'depth': current_depth,
            'poly_degree': config_params['POLY_DEGREE'],
            'poly_basis': config_params['POLY_BASIS_USED'],
            'rectangle': segment_rectangle,
            'final_error_measure': -1.0,
            'status': 'approximation_failed',
            'raw_error_maps': {key: np.zeros_like(image_segment) for key in ['error_original', 'error_smoothed', 'diff_original_poly_smoothed', 'diff_smoothed_poly_original']} # Return empty error maps on failure
        }
        return segment_id, results_data, None


    # --- Step 3: Evaluate Reconstruction Quality (Compute M(S)) ---
    # The error measure for adaptive criterion is still based on 'error_original'
    error_map_for_measure = raw_error_maps.get('error_original')

    if error_map_for_measure is None:
         print(f"Worker Error: Could not get 'error_original' map for error measure in segment {segment_id}.")
         results_data = {
             'bbox': segment_bbox,
             'coefficients': polynomial_coefficients_for_reconstruction, # Pass the coefficients
             'depth': current_depth,
             'poly_degree': config_params['POLY_DEGREE'],
             'poly_basis': config_params['POLY_BASIS_USED'],
             'rectangle': segment_rectangle,
             'final_error_measure': -1.0,
             'status': 'error_measure_input_missing',
             'raw_error_maps': raw_error_maps # Still pass raw error maps
         }
         return segment_id, results_data, None

    try:
        # Use config_params dictionary to access configuration values
        segment_error_measure = calculate_error_measure(error_map_for_measure, measure_type=config_params['ERROR_MEASURE_TYPE'])
        # print(f"Worker Error measure ({config_params['ERROR_MEASURE_TYPE']}) for segment {segment_id}: {segment_error_measure:.6f}") # Verbose worker output
    except Exception as e:
        print(f"Worker Error calculating error measure for segment {segment_id}: {e}")
        results_data = {
            'bbox': segment_bbox,
            'coefficients': polynomial_coefficients_for_reconstruction, # Pass the coefficients
            'depth': current_depth,
            'poly_degree': config_params['POLY_DEGREE'],
            'poly_basis': config_params['POLY_BASIS_USED'],
            'rectangle': segment_rectangle,
            'final_error_measure': -1.0,
            'status': 'error_measure_failed',
            'raw_error_maps': raw_error_maps # Still pass raw error maps
        }
        return segment_id, results_data, None


    # --- Step 4: Decision and Refinement ---
    # Stopping criterion met: Either error is low enough, max depth reached, or segment is too small.
    # Use config_params dictionary to access configuration values
    if segment_error_measure <= config_params['ERROR_THRESHOLD']:
        # print(f"Worker Stopping for segment {segment_id}: Error measure {segment_error_measure:.6f} <= {config_params['ERROR_THRESHOLD']}.") # Verbose worker output
        results_data = {
            'bbox': segment_bbox,
            'coefficients': polynomial_coefficients_for_reconstruction, # Pass the coefficients
            'depth': current_depth,
            'poly_degree': config_params['POLY_DEGREE'],
            'poly_basis': config_params['POLY_BASIS_USED'],
            'rectangle': segment_rectangle,
            'final_error_measure': segment_error_measure,
            'status': 'terminated_by_error',
            'raw_error_maps': raw_error_maps # Pass raw error maps
        }
        return segment_id, results_data, None

    elif current_depth >= config_params['MAX_DEPTH']:
         # print(f"Worker Max depth ({config_params['MAX_DEPTH']}) reached for segment {segment_id}. Stopping recursion.") # Verbose worker output
         results_data = {
             'bbox': segment_bbox,
             'coefficients': polynomial_coefficients_for_reconstruction, # Pass the coefficients
             'depth': current_depth,
             'poly_degree': config_params['POLY_DEGREE'],
             'poly_basis': config_params['POLY_BASIS_USED'],
             'rectangle': segment_rectangle,
             'final_error_measure': segment_error_measure,
             'status': 'terminated_by_depth',
             'raw_error_maps': raw_error_maps # Pass raw error maps
         }
         return segment_id, results_data, None

    elif height <= config_params['MIN_SEGMENT_SIZE'] or width <= config_params['MIN_SEGMENT_SIZE']:
         # print(f"Worker Segment size ({width}x{height}) below minimum ({config_params['MIN_SEGMENT_SIZE']}) for segment {segment_id}. Stopping recursion.") # Verbose worker output
         results_data = {
             'bbox': segment_bbox,
             'coefficients': polynomial_coefficients_for_reconstruction, # Pass the coefficients
             'depth': current_depth,
             'poly_degree': config_params['POLY_DEGREE'],
             'poly_basis': config_params['POLY_BASIS_USED'],
             'rectangle': segment_rectangle,
             'final_error_measure': segment_error_measure,
             'status': 'terminated_by_size',
             'raw_error_maps': raw_error_maps # Pass raw error maps
         }
         return segment_id, results_data, None

    else:
        # Error is too high and stopping criteria not met: Subdivide and return sub-segment tasks.
        # print(f"Worker Subdividing segment {segment_id}: Error measure {segment_error_measure:.6f} > {config_params['ERROR_THRESHOLD']}.") # Verbose worker output

        mid_row = segment_bbox[0] + height // 2 # Use segment_bbox to calculate midpoints
        mid_col = segment_bbox[2] + width // 2

        sub_segments_bbox = []
        # Top-left
        if mid_row > segment_bbox[0] and mid_col > segment_bbox[2]:
            sub_segments_bbox.append((segment_bbox[0], mid_row, segment_bbox[2], mid_col))
        # Top-right
        if mid_row > segment_bbox[0] and segment_bbox[3] > mid_col:
             sub_segments_bbox.append((segment_bbox[0], mid_row, mid_col, segment_bbox[3]))
        # Bottom-left
        if segment_bbox[1] > mid_row and mid_col > segment_bbox[2]:
            sub_segments_bbox.append((mid_row, segment_bbox[1], segment_bbox[2], mid_col))
        # Bottom-right
        if segment_bbox[1] > mid_row and segment_bbox[3] > mid_col:
            sub_segments_bbox.append((mid_row, segment_bbox[1], mid_col, segment_bbox[3]))

        # Return the list of new sub-segment tasks
        sub_segment_tasks = []
        for i, sub_bbox in enumerate(sub_segments_bbox):
             sub_segment_id = f"{segment_id}_{i}"
             sub_segment_tasks.append((sub_bbox, current_depth + 1, sub_segment_id))

        return segment_id, {}, sub_segment_tasks # Return empty results_data and the new tasks


class AdaptivePolynomialReconstructor:
    """
    Performs adaptive image reconstruction using polynomial approximation
    on recursively subdivided segments, with optional parallel processing
    and higher-resolution output.
    Also reassembles and processes error maps to derive a binary edge map.
    """

    def __init__(self, config_obj):
        """
        Initializes the reconstructor with configuration parameters.

        Args:
            config_obj: A module or object containing all configuration attributes
                        (e.g., from config.py).
        """
        self.config = config_obj
        self.original_image = None
        self.original_image_shape = None
        # Dictionary to store results from final segments (bbox, coeffs, degree, basis, rectangle, error, raw_error_maps)
        self.final_segment_data: Dict[str, Dict] = {}
        self.image_adaptive_results_dir = None

        # New attributes for reassembled error maps and binary edge map
        self.reassembled_raw_error_maps: Dict[str, np.ndarray] = {}
        self.reassembled_composite_error_map: Optional[np.ndarray] = None
        self.final_adaptive_binary_edge_map: Optional[np.ndarray] = None


    def load_image_grayscale(self, image_path: Path) -> np.ndarray:
        """Loads a grayscale image and normalizes it to [0, 1]."""
        if not image_path.exists():
            # Create a dummy image file if the sample doesn't exist for demonstration
            print(f"Sample image not found at {image_path}. Creating a dummy image.")
            dummy_img = np.zeros((256, 256), dtype=np.uint8)
            # Add a white square
            dummy_img[50:150, 50:150] = 255
            # Add a gradient
            for i in range(256):
                dummy_img[i, :] = i
            # Ensure the 'images' directory exists if saving dummy there
            os.makedirs(self.config.IMAGE_DIR, exist_ok=True)
            Image.fromarray(dummy_img).save(image_path)
            print(f"Dummy image created at {image_path}")

        print(f"Loading image: {image_path}")
        # Open and convert to grayscale ('L') and normalize to [0, 1]
        # Use float32 for consistency with approximation and error maps
        img = np.array(Image.open(image_path).convert('L'), dtype=np.float32) / 255.0
        return img


    def reassemble_results_from_coefficients(self, upscale_factor: int = 1) -> np.ndarray:
        """
        Reassembles the approximation image from stored segment bounding boxes,
        polynomial coefficients, degree, basis type, and rectangle.
        Optionally upscales the output image.

        Args:
            upscale_factor: Integer factor to upscale the output image resolution.
                            1 means original resolution.

        Returns:
            np.ndarray
                The reassembled approximation image. Returns an empty array if
                original image shape is not available or no segment data.
        """
        if self.original_image_shape is None:
            print("Error: Original image not loaded. Cannot reassemble.")
            return np.array([])

        if not self.final_segment_data:
             print("No segment data available. Cannot reassemble.")
             return np.array([])

        original_height, original_width = self.original_image_shape
        upscaled_height = original_height * upscale_factor
        upscaled_width = original_width * upscale_factor

        reassembled_approx = np.zeros((upscaled_height, upscaled_width), dtype=np.float32)

        for segment_id, results in self.final_segment_data.items():
            if 'bbox' in results and 'coefficients' in results and 'poly_degree' in results and 'poly_basis' in results and 'rectangle' in results:
                row_start, row_end, col_start, col_end = results['bbox']
                coefficients = results['coefficients']
                poly_degree = results['poly_degree']
                poly_basis = results['poly_basis']
                segment_rectangle = results['rectangle']

                # Calculate upscaled bbox coordinates
                upscaled_row_start = row_start * upscale_factor
                upscaled_row_end = row_end * upscale_factor
                upscaled_col_start = col_start * upscale_factor
                upscaled_col_end = col_end * upscale_factor

                upscaled_seg_height = upscaled_row_end - upscaled_row_start
                upscaled_seg_width = upscaled_col_end - upscaled_col_start

                # Ensure upscaled bbox coordinates are within the upscaled image bounds
                if upscaled_row_start < 0 or upscaled_row_end > upscaled_height or upscaled_col_start < 0 or upscaled_col_end > upscaled_width:
                     print(f"Warning: Upscaled segment {segment_id} bbox [{upscaled_row_start}:{upscaled_row_end}, {upscaled_col_start}:{upscaled_col_end}] is out of upscaled image bounds {(upscaled_height, upscaled_width)}. Skipping reassembly for this segment.")
                     continue

                if coefficients.size == 0:
                     continue # Skip if no coefficients

                # Define the rectangle for this segment relative to the original image [0,1]x[0,1]
                xmin, ymin, xmax, ymax = segment_rectangle

                # Create a grid of points over the *upscaled* segment's pixel coordinates
                # scaled to the rectangle [xmin, ymin] to [xmax, ymax] for evaluation.
                x_eval_scaled = np.linspace(xmin, xmax, upscaled_seg_width)
                y_eval_scaled = np.linspace(ymin, ymax, upscaled_seg_height)
                XX_eval_scaled, YY_eval_scaled = np.meshgrid(x_eval_scaled, y_eval_scaled)
                evaluation_points_scaled = np.vstack([XX_eval_scaled.ravel(), YY_eval_scaled.ravel()]).T

                try:
                    poly_dimension = int((poly_degree + 1) * (poly_degree + 2) / 2)
                    dummy_X = np.zeros((1, 2))
                    _, basis_func_generator = gen_vanderm2d(
                        X=dummy_X,
                        col=poly_dimension,
                        poly_basis=poly_basis,
                        rectangle=segment_rectangle
                    )

                    approx_flat = evaluate_polynomial_from_coeffs(
                        coefficients,
                        basis_func_generator,
                        evaluation_points_scaled
                    ).real

                    # Reshape and clip
                    reassembled_approx[upscaled_row_start:upscaled_row_end, upscaled_col_start:upscaled_col_end] = np.clip(approx_flat.reshape(upscaled_seg_height, upscaled_seg_width), 0, 1)

                except Exception as e:
                     print(f"Error evaluating polynomial for segment {segment_id} during reassembly: {e}")
                     # Fill this segment region with zeros in the reassembled image
                     reassembled_approx[upscaled_row_start:upscaled_row_end, upscaled_col_start:upscaled_col_end] = np.zeros((upscaled_seg_height, upscaled_seg_width), dtype=np.float32)

        return reassembled_approx

    def reassemble_raw_error_maps_for_all_types(self, upscale_factor: int = 1) -> None:
        """
        Reassembles all raw error maps (e.g., 'error_original', 'error_smoothed')
        from stored segment data into full-image maps.
        """
        if self.original_image_shape is None:
            print("Error: Original image not loaded. Cannot reassemble error maps.")
            return

        if not self.final_segment_data:
            print("No segment data available. Cannot reassemble error maps.")
            return

        original_height, original_width = self.original_image_shape
        upscaled_height = original_height * upscale_factor
        upscaled_width = original_width * upscale_factor

        # Initialize empty maps for each raw error type
        error_map_types = ['error_original', 'error_smoothed', 'diff_original_poly_smoothed', 'diff_smoothed_poly_original']
        temp_reassembled_error_maps = {
            err_type: np.zeros((upscaled_height, upscaled_width), dtype=np.float32)
            for err_type in error_map_types
        }

        for segment_id, results in self.final_segment_data.items():
            if 'bbox' in results and 'raw_error_maps' in results:
                row_start, row_end, col_start, col_end = results['bbox']
                segment_raw_error_maps = results['raw_error_maps']

                upscaled_row_start = row_start * upscale_factor
                upscaled_row_end = row_end * upscale_factor
                upscaled_col_start = col_start * upscale_factor
                upscaled_col_end = col_end * upscale_factor

                upscaled_seg_height = upscaled_row_end - upscaled_row_start
                upscaled_seg_width = upscaled_col_end - upscaled_col_start

                for err_type in error_map_types:
                    if err_type in segment_raw_error_maps and segment_raw_error_maps[err_type].size > 0:
                        # Resize segment error map to upscaled segment size if needed
                        segment_error_map = segment_raw_error_maps[err_type]
                        if segment_error_map.shape[0] != upscaled_seg_height or segment_error_map.shape[1] != upscaled_seg_width:
                            # Resize using PIL for error maps (convert to uint8 for PIL, then back to float32)
                            resized_segment_error = np.array(Image.fromarray((segment_error_map * 255).astype(np.uint8), 'L').resize(
                                (upscaled_seg_width, upscaled_seg_height), Image.Resampling.LANCZOS
                            ), dtype=np.float32) / 255.0
                        else:
                            resized_segment_error = segment_error_map

                        temp_reassembled_error_maps[err_type][upscaled_row_start:upscaled_row_end, upscaled_col_start:upscaled_col_end] = resized_segment_error

        self.reassembled_raw_error_maps = temp_reassembled_error_maps
        print("Reassembled all raw error maps.")


    def draw_segmentation_boundaries(self, image: np.ndarray, ax: plt.Axes, color='red', linewidth=1):
        """
        Draws the bounding boxes of the terminal segments on an image plot.

        Args:
            image : np.ndarray
                The image to draw on (e.g., the original or reconstructed image).
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

        # Determine the upscale factor used for the image being plotted
        if self.original_image_shape is not None and image.shape == self.original_image_shape:
            upscale_factor = 1
        elif self.original_image_shape is not None:
             # Calculate upscale factor based on height (assuming square pixels and consistent scaling)
             upscale_factor = image.shape[0] / self.original_image_shape[0]
        else:
             upscale_factor = 1 # Assume no upscaling if original shape unknown

        for segment_id, results in self.final_segment_data.items():
            if 'bbox' in results:
                row_start, row_end, col_start, col_end = results['bbox']

                # Scale bbox coordinates by the upscale factor of the image being plotted
                scaled_row_start = row_start * upscale_factor
                scaled_row_end = row_end * upscale_factor
                scaled_col_start = col_start * upscale_factor
                scaled_col_end = col_end * upscale_factor


                # Create a Rectangle patch
                # Rectangle takes (x, y) as the lower left corner, width, height
                # So, x = scaled_col_start, y = scaled_row_start, width = scaled_col_end - scaled_col_start, height = scaled_row_end - scaled_row_start
                rect = patches.Rectangle(
                    (scaled_col_start, scaled_row_start), # (x, y) of lower left corner
                    scaled_col_end - scaled_col_start,    # width
                    scaled_row_end - scaled_row_start,    # height
                    linewidth=linewidth,
                    edgecolor=color,
                    facecolor='none' # No fill
                )

                # Add the patch to the Axes
                ax.add_patch(rect)

    def draw_segment_error_heatmap(self, ax: plt.Axes, colormap='viridis'):
        """
        Draws a color-coded heatmap of the segmentation, where each segment's color
        intensity reflects its final error measure.

        Args:
            ax : plt.Axes
                The matplotlib Axes object to draw the heatmap on.
            colormap : str, optional
                The colormap to use for visualizing error. Default is 'viridis'.
        """
        if self.original_image_shape is None:
             print("Error: Original image shape not available. Cannot draw segment error heatmap.")
             ax.set_title("Segment Error Heatmap (No data)")
             ax.axis('off')
             return

        height, width = self.original_image_shape
        error_heatmap_image = np.zeros(self.original_image_shape, dtype=np.float32)

        # Collect all final error measures from segments that terminated by error or depth/size
        # Exclude segments that failed approximation or had missing error measure input
        final_errors = [
            results['final_error_measure'] for results in self.final_segment_data.values()
            if 'final_error_measure' in results and results['final_error_measure'] >= 0 # Only include valid errors
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
        # Handle case where all valid errors are the same
        if max_error - min_error < epsilon:
             normalized_errors = np.zeros_like(final_errors) # All errors are the same, normalize to 0
        else:
             normalized_errors = (final_errors - min_error) / (max_error - min_error + epsilon)


        # Create a mapping from segment_id to its normalized error
        segment_normalized_error = {}
        valid_segments_data = {
            segment_id: results for segment_id, results in self.final_segment_data.items()
            if 'final_error_measure' in results and results['final_error_measure'] >= 0
        }

        if valid_segments_data:
            min_valid_error = min(results['final_error_measure'] for results in valid_segments_data.values())
            max_valid_error = max(results['final_error_measure'] for results in valid_segments_data.values())

            for segment_id, results in valid_segments_data.items():
                 error_val = results['final_error_measure']
                 if max_valid_error - min_valid_error < epsilon:
                      normalized_err = 0.0 # All valid errors are the same
                 else:
                      normalized_err = (error_val - min_valid_error) / (max_valid_error - min_valid_error + epsilon)
                 segment_normalized_error[segment_id] = normalized_err


        # Fill the heatmap image with normalized error values for each segment
        for segment_id, results in self.final_segment_data.items():
            if 'bbox' in results and segment_id in segment_normalized_error:
                row_start, row_end, col_start, col_end = results['bbox']
                normalized_err = segment_normalized_error[segment_id]

                # Fill the segment area with the normalized error value
                error_heatmap_image[row_start:row_end, col_start:col_end] = normalized_err

        # Display the heatmap image
        im = ax.imshow(error_heatmap_image, cmap=colormap, origin='upper')
        fig = ax.get_figure() # Get the figure to add a colorbar
        fig.colorbar(im, ax=ax, label=f'Final Segment Error ({self.config.ERROR_MEASURE_TYPE}, Normalized)')
        ax.set_title(f'Segment Error Heatmap ({self.config.ERROR_MEASURE_TYPE})')
        ax.axis('off')

    def plot_error_distribution(self):
        """Plots a histogram of the final segment error measures."""
        if not self.final_segment_data:
            print("No segment data available. Cannot plot error distribution.")
            return

        final_errors = [
            results['final_error_measure'] for results in self.final_segment_data.values()
            if 'final_error_measure' in results and results['final_error_measure'] >= 0 # Only include valid errors
        ]

        if not final_errors:
            print("No valid final error measures found. Cannot plot error distribution.")
            return

        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        ax.hist(final_errors, bins=50, edgecolor='black')
        ax.set_title(f'Distribution of Final Segment Errors ({self.config.ERROR_MEASURE_TYPE})')
        ax.set_xlabel(f'Final Segment Error ({self.config.ERROR_MEASURE_TYPE})')
        ax.set_ylabel('Number of Segments')
        plt.tight_layout()

        # Save the plot
        if self.image_adaptive_results_dir:
            params_suffix = (
                f"deg{self.config.POLY_DEGREE}_{self.config.NODES_METHOD}"
                f"_measure-{self.config.ERROR_MEASURE_TYPE}_errthresh{str(self.config.ERROR_THRESHOLD).replace('.', 'p')}_depth{self.config.MAX_DEPTH}_min{self.config.MIN_SEGMENT_SIZE}"
                f"_basis{self.config.POLY_BASIS_USED}"
            )
            plot_filename = f"{self.config.IMAGE_NAME}_error_distribution_{params_suffix}.png"
            plot_filepath = os.path.join(str(self.image_adaptive_results_dir), plot_filename)
            try:
                plt.savefig(plot_filepath)
                print(f"Saved error distribution plot to {plot_filepath}")
            except Exception as e:
                print(f"Error saving error distribution plot to {plot_filepath}: {e}")
        else:
             print("Results directory not set. Skipping saving error distribution plot.")

        plt.close(fig)


    def plot_depth_segment_count(self):
        """Plots the number of segments at each depth level."""
        if not self.final_segment_data:
            print("No segment data available. Cannot plot depth vs. segment count.")
            return

        depth_counts = {}
        for results in self.final_segment_data.values():
            depth = results.get('depth', -1) # Use -1 for segments without depth info
            if depth >= 0:
                 depth_counts[depth] = depth_counts.get(depth, 0) + 1

        if not depth_counts:
            print("No valid depth information found in segment data. Cannot plot depth vs. segment count.")
            return

        depths = sorted(depth_counts.keys())
        counts = [depth_counts[d] for d in depths]

        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        ax.bar(depths, counts, edgecolor='black')
        ax.set_title('Number of Segments per Depth Level')
        ax.set_xlabel('Depth Level')
        ax.set_ylabel('Number of Segments')
        ax.set_xticks(depths) # Ensure all depths are shown as ticks
        plt.tight_layout()

        # Save the plot
        if self.image_adaptive_results_dir:
            params_suffix = (
                f"deg{self.config.POLY_DEGREE}_{self.config.NODES_METHOD}"
                f"_measure-{self.config.ERROR_MEASURE_TYPE}_errthresh{str(self.config.ERROR_THRESHOLD).replace('.', 'p')}_depth{self.config.MAX_DEPTH}_min{self.config.MIN_SEGMENT_SIZE}"
                f"_basis{self.config.POLY_BASIS_USED}"
            )
            plot_filename = f"{self.config.IMAGE_NAME}_depth_segment_count_{params_suffix}.png"
            plot_filepath = os.path.join(str(self.image_adaptive_results_dir), plot_filename)
            try:
                plt.savefig(plot_filepath)
                print(f"Saved depth vs. segment count plot to {plot_filepath}")
            except Exception as e:
                print(f"Error saving depth vs. segment count plot to {plot_filepath}: {e}")
        else:
             print("Results directory not set. Skipping saving depth vs. segment count plot.")

        plt.close(fig)

    def plot_segment_size_distribution(self):
        """Plots a histogram of the final segment sizes (area)."""
        if not self.final_segment_data:
            print("No segment data available. Cannot plot segment size distribution.")
            return

        segment_sizes = []
        for results in self.final_segment_data.values():
            if 'bbox' in results:
                 row_start, row_end, col_start, col_end = results['bbox']
                 height = row_end - row_start
                 width = col_end - col_start
                 segment_sizes.append(height * width)

        if not segment_sizes:
            print("No valid segment bounding boxes found. Cannot plot segment size distribution.")
            return

        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        ax.hist(segment_sizes, bins=50, edgecolor='black') # Adjust bins as needed
        ax.set_title('Distribution of Final Segment Sizes (Pixels)')
        ax.set_xlabel('Segment Size (Area in Pixels)')
        ax.set_ylabel('Number of Segments')
        plt.tight_layout()

        # Save the plot
        if self.image_adaptive_results_dir:
            params_suffix = (
                f"deg{self.config.POLY_DEGREE}_{self.config.NODES_METHOD}"
                f"_measure-{self.config.ERROR_MEASURE_TYPE}_errthresh{str(self.config.ERROR_THRESHOLD).replace('.', 'p')}_depth{self.config.MAX_DEPTH}_min{self.config.MIN_SEGMENT_SIZE}"
                f"_basis{self.config.POLY_BASIS_USED}"
            )
            plot_filename = f"{self.config.IMAGE_NAME}_segment_size_distribution_{params_suffix}.png"
            plot_filepath = os.path.join(str(self.image_adaptive_results_dir), plot_filename)
            try:
                plt.savefig(plot_filepath)
                print(f"Saved segment size distribution plot to {plot_filepath}")
            except Exception as e:
                print(f"Error saving segment size distribution plot to {plot_filepath}: {e}")
        else:
             print("Results directory not set. Skipping saving segment size distribution plot.")

        plt.close(fig)


    def run_reconstruction(self):
        """
        Runs the complete adaptive image reconstruction process.
        """
        if not _poly_approx_available:
            print("Skipping adaptive image reconstruction due to missing poly_approx modules.")
            return

        # --- Load the original image ---
        try:
            self.original_image = self.load_image_grayscale(self.config.IMAGE_PATH)
            self.original_image_shape = self.original_image.shape
        except FileNotFoundError as e:
            print(f"Error loading image: {e}")
            return
        except Exception as e:
            print(f"An unexpected error occurred loading the image: {e}")
            return

        # Define the initial segment (the whole image)
        initial_bbox = (0, self.original_image_shape[0], 0, self.original_image_shape[1])
        initial_segment_id = "root"

        # Create a subfolder for this specific image's results within the adaptive tests directory
        self.image_adaptive_results_dir = self.config.ADAPTIVE_RECONSTRUCTION_RESULTS_BASE_DIR / self.config.IMAGE_NAME
        os.makedirs(self.image_adaptive_results_dir, exist_ok=True)
        print(f"Saving results to: {self.image_adaptive_results_dir}")

        print(f"\nStarting adaptive image reconstruction for {self.config.IMAGE_FILENAME}...")
        start_time = time.time()

        # --- Parallel Processing Setup ---
        num_processes = self.config.NUM_PROCESSES
        if num_processes is None or num_processes <= 0:
            num_processes = 1 # Use single process if None or non-positive

        print(f"Using {num_processes} processes for segment processing.")

        # Use a Queue to manage segments that need processing
        segment_queue = Queue()
        # For the initial segment, get the image data and rectangle
        row_start, row_end, col_start, col_end = initial_bbox
        initial_segment_image = self.original_image[row_start:row_end, col_start:col_end]
        original_height, original_width = self.original_image_shape
        initial_segment_rectangle = (
            col_start / original_width,
            row_start / original_height,
            col_end / original_width,
            row_end / original_height
        )
        # Put the initial task with image data and rectangle into the queue
        segment_queue.put((initial_bbox, 0, initial_segment_id, initial_segment_image, initial_segment_rectangle))


        # List to store results from the worker processes
        results_from_workers = []

        # Prepare a picklable dictionary of necessary config parameters for the worker
        worker_config_params = {
            'POLY_DEGREE': self.config.POLY_DEGREE,
            'NODES_METHOD': self.config.NODES_METHOD,
            'ADMISSIBLE_MESH_TYPE': self.config.ADMISSIBLE_MESH_TYPE,
            'M_CHEB': self.config.M_CHEB,
            'POLY_BASIS_USED': self.config.POLY_BASIS_USED,
            'ERROR_MEASURE_TYPE': self.config.ERROR_MEASURE_TYPE,
            'ERROR_THRESHOLD': self.config.ERROR_THRESHOLD,
            'MAX_DEPTH': self.config.MAX_DEPTH,
            'MIN_SEGMENT_SIZE': self.config.MIN_SEGMENT_SIZE,
            # Add other necessary config parameters here if used by the worker
        }


        # Use a Pool of workers. No initializer needed as data is passed directly.
        with multiprocessing.Pool(processes=num_processes) as pool:
            # While there are segments in the queue, submit tasks to the pool
            # We need a way to track active tasks to know when to stop waiting for results
            active_results = []

            # Initially submit the first task
            if not segment_queue.empty():
                 task_with_data = segment_queue.get()
                 # Pass task (bbox, depth, id), image_segment, rectangle, and config_params
                 result = pool.apply_async(process_segment_worker, args=(task_with_data[:3], task_with_data[3], task_with_data[4], worker_config_params))
                 active_results.append(result)


            # Process results and add new tasks until no more active tasks and the queue is empty
            while active_results or not segment_queue.empty():
                # Check for completed tasks
                completed_results = [r for r in active_results if r.ready()]
                active_results = [r for r in active_results if not r.ready()] # Keep only pending results

                for result in completed_results:
                    try:
                        segment_id, results_data, sub_segment_tasks = result.get() # Get the result from the worker
                        if results_data:
                            # This was a terminal segment, store its data
                            self.final_segment_data[segment_id] = results_data
                        if sub_segment_tasks:
                            # This segment was subdivided, add new tasks to the queue
                            for sub_task_bbox, sub_task_depth, sub_task_id in sub_segment_tasks:
                                # Get the image data and rectangle for the new sub-task
                                # Access bbox elements by index directly in slicing
                                sub_segment_image = self.original_image[sub_task_bbox[0]:sub_task_bbox[1], sub_task_bbox[2]:sub_task_bbox[3]]
                                sub_segment_rectangle = (
                                    sub_task_bbox[2] / original_width, # col_start / original_width
                                    sub_task_bbox[0] / original_height, # row_start / original_height
                                    sub_task_bbox[3] / original_width, # col_end / original_width
                                    sub_task_bbox[1] / original_height # row_end / original_height
                                )
                                # Put the new task with image data and rectangle into the queue
                                segment_queue.put((sub_task_bbox, sub_task_depth, sub_task_id, sub_segment_image, sub_segment_rectangle))

                    except Exception as e:
                        print(f"Error collecting result from worker: {e}")
                        # Handle potential errors from worker processes

                # If there are tasks in the queue and the pool is not saturated, submit more
                # Simple heuristic: submit up to num_processes * 2 tasks ahead
                while not segment_queue.empty() and len(active_results) < num_processes * 2:
                    task_with_data = segment_queue.get()
                    # Pass task (bbox, depth, id), image_segment, rectangle, and config_params
                    result = pool.apply_async(process_segment_worker, args=(task_with_data[:3], task_with_data[3], task_with_data[4], worker_config_params))
                    active_results.append(result)

                # Add a small sleep to prevent busy-waiting if the queue is temporarily empty
                if not active_results and segment_queue.empty():
                     break # Exit loop if no active tasks and queue is empty
                time.sleep(0.01) # Sleep briefly


        end_time = time.time()
        print(f"\nAdaptive image reconstruction finished in {end_time - start_time:.4f} seconds.")
        print(f"Processed {len(self.final_segment_data)} final segments.")

        # --- Save the segment data and coefficients for compression analysis ---
        if self.config.SAVE_SEGMENT_DATA and self.final_segment_data:
            print(f"Saving segment data and coefficients to: {self.image_adaptive_results_dir}")

            # Construct a filename suffix based on parameters
            params_suffix = (
                f"deg{self.config.POLY_DEGREE}_{self.config.NODES_METHOD}"
                # f"_sigma{self.config.SIGMA:.1f}" # Include sigma if added to config
                f"_measure-{self.config.ERROR_MEASURE_TYPE}_errthresh{str(self.config.ERROR_THRESHOLD).replace('.', 'p')}_depth{self.config.MAX_DEPTH}_min{self.config.MIN_SEGMENT_SIZE}"
                f"_basis{self.config.POLY_BASIS_USED}" # Add basis to filename
                f"_procs{num_processes}" # Add number of processes
            )

            # Filename for the saved data
            data_filename = f"{self.config.IMAGE_NAME}_segment_data_{params_suffix}.pkl" # Using pickle for simplicity
            data_filepath = os.path.join(str(self.image_adaptive_results_dir), data_filename)

            try:
                with open(data_filepath, 'wb') as f:
                    pickle.dump(self.final_segment_data, f)
                print(f"Saved segment data and coefficients to {data_filepath}")
                print("This file contains the bounding boxes, polynomial degree, basis type, and coefficients for each terminal segment.")
                print("It can be used as the 'compressed' representation for compression analysis.")
            except Exception as e:
                print(f"Error saving segment data and coefficients to {data_filepath}: {e}")


        # --- Reassemble the final results from the stored coefficients for visualization ---
        # This step acts as a 'decoding' process to reconstruct the image from the saved data.
        print("\nReassembling final approximation image from coefficients...")
        if self.final_segment_data:
            # Reassemble at the specified upscale factor
            reassembled_approx_image = self.reassemble_results_from_coefficients(self.config.UPSCALE_FACTOR)
            print("Reassembly from coefficients complete.")

            if reassembled_approx_image.size > 0:
                # --- Calculate the actual reconstruction error on the reassembled image ---
                # Note: Error is calculated against the ORIGINAL image, not an upscaled version of it.
                # If you want to compare against a high-res ground truth, you would load that here.
                # For now, we'll downscale the upscaled reconstruction for error calculation against original.
                # A more rigorous approach might involve evaluating the polynomial at original pixel locations.
                # Let's stick to downscaling the reassembled image for simplicity in error calculation for now.

                # Downscale the reassembled image to the original image shape for error comparison
                # Use PIL for resizing
                reassembled_approx_original_size = np.array(Image.fromarray((reassembled_approx_image * 255).astype(np.uint8), 'L').resize(
                    (self.original_image_shape[1], self.original_image_shape[0]), Image.Resampling.LANCZOS
                ), dtype=np.float32) / 255.0


                actual_reconstruction_error_map = np.abs(self.original_image - reassembled_approx_original_size)

                # Normalize the actual reconstruction error for visualization
                normalized_reconstruction_error_for_plot = normalize_error_image(actual_reconstruction_error_map)
                print("Actual reconstruction error map calculated and normalized.")

                # --- Reassemble and process error maps for edge detection ---
                print("\nReassembling raw error maps for edge detection...")
                self.reassemble_raw_error_maps_for_all_types(self.config.UPSCALE_FACTOR)

                # Store the original ERROR_COMBINATION_STRATEGY to restore it later
                original_error_combination_strategy = self.config.ERROR_COMBINATION_STRATEGY

                if self.reassembled_raw_error_maps:
                    print("\nCombining reassembled error maps for edge detection...")
                    normalized_reassembled_errors = {
                        key: normalize_error_image(err_map)
                        for key, err_map in self.reassembled_raw_error_maps.items()
                        if err_map is not None and err_map.size > 0
                    }

                    if normalized_reassembled_errors:
                        # Define the list of possible raw error map keys for direct use
                        raw_error_map_keys = ['error_original', 'error_smoothed', 'diff_original_poly_smoothed', 'diff_smoothed_poly_original']

                        if self.config.ERROR_COMBINATION_STRATEGY in ['logical_and', 'logical_or']:
                            # For logical operations, first threshold individual maps
                            binary_maps_for_logical_op = []
                            for key in self.config.LOGICAL_OP_ERROR_KEYS:
                                if key in normalized_reassembled_errors:
                                    binary_map = apply_threshold(
                                        normalized_reassembled_errors[key],
                                        threshold_type=self.config.EDGE_THRESHOLD_TYPE,
                                        fixed_threshold=self.config.FIXED_EDGE_THRESHOLD
                                    )
                                    binary_maps_for_logical_op.append(binary_map)
                                else:
                                    print(f"Warning: Logical operation key '{key}' not found in reassembled error maps. Skipping.")

                            if binary_maps_for_logical_op:
                                if self.config.ERROR_COMBINATION_STRATEGY == 'logical_and':
                                    self.reassembled_composite_error_map = np.logical_and.reduce(binary_maps_for_logical_op).astype(np.float32)
                                else: # 'logical_or'
                                    self.reassembled_composite_error_map = np.logical_or.reduce(binary_maps_for_logical_op).astype(np.float32)
                                print(f"Reassembled composite error map created using '{self.config.ERROR_COMBINATION_STRATEGY}'.")
                            else:
                                print("No binary maps available for logical combination. Composite map will be empty.")
                                self.reassembled_composite_error_map = np.zeros_like(self.original_image, dtype=np.float32) # Default empty
                        
                        # NEW: Handle direct use of a single raw error map
                        elif self.config.ERROR_COMBINATION_STRATEGY in raw_error_map_keys:
                            chosen_key = self.config.ERROR_COMBINATION_STRATEGY
                            if chosen_key in normalized_reassembled_errors:
                                self.reassembled_composite_error_map = normalized_reassembled_errors[chosen_key]
                                print(f"Reassembled composite error map set to normalized '{chosen_key}'.")
                            else:
                                print(f"Error: Specified single error map key '{chosen_key}' not found in reassembled error maps. Composite map will be empty.")
                                self.reassembled_composite_error_map = np.zeros_like(self.original_image, dtype=np.float32) # Default empty
                        
                        else:
                            # For 'max' or 'weighted_sum', use combine_errors as before
                            self.reassembled_composite_error_map = combine_errors(
                                normalized_reassembled_errors,
                                strategy=self.config.ERROR_COMBINATION_STRATEGY,
                                weights=self.config.ERROR_COMBINATION_WEIGHTS
                            )
                            print("Reassembled composite error map created.")
                    else:
                        print("No normalized reassembled error maps to combine.")

                    # Apply thresholding to get the final binary edge map
                    if self.reassembled_composite_error_map is not None and self.reassembled_composite_error_map.size > 0:
                        self.final_adaptive_binary_edge_map = apply_threshold(
                            self.reassembled_composite_error_map,
                            threshold_type=self.config.EDGE_THRESHOLD_TYPE,
                            fixed_threshold=self.config.FIXED_EDGE_THRESHOLD
                        )
                        print("Final adaptive binary edge map created.")
                    else:
                        print("No composite error map to threshold.")
                else:
                    print("No reassembled raw error maps to process for edge detection.")


                # --- Save the reassembled approximation image and error map for visualization ---
                params_suffix = ( # Regenerate suffix to ensure consistency
                    f"deg{self.config.POLY_DEGREE}_{self.config.NODES_METHOD}"
                    # f"_sigma{self.config.SIGMA:.1f}" # Include sigma if added to config
                    f"_measure-{self.config.ERROR_MEASURE_TYPE}_errthresh{str(self.config.ERROR_THRESHOLD).replace('.', 'p')}_depth{self.config.MAX_DEPTH}_min{self.config.MIN_SEGMENT_SIZE}"
                    f"_basis{self.config.POLY_BASIS_USED}"
                    f"_procs{num_processes}"
                    f"_upscale{self.config.UPSCALE_FACTOR}" # Add upscale factor
                    f"_edge_strat-{original_error_combination_strategy}" # Use original strategy for filename
                    f"_edge_thresh-{self.config.EDGE_THRESHOLD_TYPE}" # Use self.config
                    + (f"-{str(self.config.FIXED_EDGE_THRESHOLD).replace('.', 'p')}" if self.config.EDGE_THRESHOLD_TYPE == 'fixed' else '') # Use self.config
                )

                if self.config.SAVE_RECONSTRUCTED_IMAGE:
                    print("\nSaving reassembled approximation image...")
                    try:
                        # Save the upscaled image using the existing save_images function
                        save_images({f"{self.config.IMAGE_NAME}_approx_from_coeffs_{params_suffix}.png": reassembled_approx_image}, str(self.image_adaptive_results_dir))
                    except Exception as e:
                        print(f"Error saving reassembled approximation image: {e}")

                if self.config.SAVE_ERROR_MAP_VIZ:
                    print("\nSaving actual error visualization image...")
                    # Create a matplotlib figure for AREH with a colorbar
                    fig_areh, ax_areh = plt.subplots(1, 1, figsize=(8, 8))
                    im_areh = ax_areh.imshow(normalized_reconstruction_error_for_plot, cmap=self.config.ERROR_HEATMAP_COLORMAP, origin='upper')
                    fig_areh.colorbar(im_areh, ax=ax_areh, label='Actual Reconstruction Error (Normalized)')
                    ax_areh.set_title('Actual Reconstruction Error Heatmap (vs Original)')
                    ax_areh.axis('off')
                    plot_filepath_areh = os.path.join(str(self.image_adaptive_results_dir), f"{self.config.IMAGE_NAME}_actual_error_viz_from_coeffs_{params_suffix}.png")
                    try:
                        plt.savefig(plot_filepath_areh)
                        print(f"Saved actual error visualization plot to {plot_filepath_areh}")
                    except Exception as e:
                        print(f"Error saving actual error visualization plot to {plot_filepath_areh}: {e}")
                    plt.close(fig_areh)


                # Save new edge detection related maps with colorbars
                if self.config.SAVE_RAW_ERROR_MAPS and self.reassembled_raw_error_maps: # Use self.config
                    print("\nSaving reassembled raw error maps...")
                    for key, err_map in self.reassembled_raw_error_maps.items():
                        if err_map is not None and err_map.size > 0:
                            raw_err_filename = f"{self.config.IMAGE_NAME}_reassembled_raw_error_{key}_{params_suffix}.png"
                            plot_filepath_raw_err = os.path.join(str(self.image_adaptive_results_dir), raw_err_filename)
                            
                            fig_raw_err, ax_raw_err = plt.subplots(1, 1, figsize=(8, 8))
                            normalized_raw_err = normalize_error_image(err_map)
                            im_raw_err = ax_raw_err.imshow(normalized_raw_err, cmap=self.config.EDGE_ERROR_HEATMAP_COLORMAP, origin='upper')
                            fig_raw_err.colorbar(im_raw_err, ax=ax_raw_err, label=f'Normalized Raw Error ({key.replace("_", " ").title()})')
                            ax_raw_err.set_title(f'Reassembled Raw Error Map: {key.replace("_", " ").title()}')
                            ax_raw_err.axis('off')
                            try:
                                plt.savefig(plot_filepath_raw_err)
                                print(f"Saved reassembled raw error map '{key}' plot to {plot_filepath_raw_err}")
                            except Exception as e:
                                print(f"Error saving reassembled raw error map '{key}' plot: {e}")
                            plt.close(fig_raw_err)

                if self.config.SAVE_COMPOSITE_ERROR_MAP and self.reassembled_composite_error_map is not None and self.reassembled_composite_error_map.size > 0: # Use self.config
                    print("\nSaving reassembled composite error map...")
                    composite_err_filename = f"{self.config.IMAGE_NAME}_reassembled_composite_error_{params_suffix}.png"
                    plot_filepath_composite_err = os.path.join(str(self.image_adaptive_results_dir), composite_err_filename)

                    fig_composite_err, ax_composite_err = plt.subplots(1, 1, figsize=(8, 8))
                    
                    # If logical ops were used, the composite map is already binary, so use gray colormap
                    if original_error_combination_strategy in ['logical_and', 'logical_or']: # Use original strategy for plotting
                        im_composite_err = ax_composite_err.imshow(self.reassembled_composite_error_map, cmap='gray', origin='upper')
                        ax_composite_err.set_title(f'Reassembled Composite Map ({original_error_combination_strategy.capitalize()})')
                        # No colorbar for binary maps typically, but can add if desired for 0/1 range
                    else:
                        normalized_composite_err = normalize_error_image(self.reassembled_composite_error_map)
                        im_composite_err = ax_composite_err.imshow(normalized_composite_err, cmap=self.config.EDGE_ERROR_HEATMAP_COLORMAP, origin='upper')
                        fig_composite_err.colorbar(im_composite_err, ax=ax_composite_err, label=f'Normalized Composite Error')
                        ax_composite_err.set_title(f'Reassembled Composite Error Heatmap ({original_error_combination_strategy.capitalize()})')
                    
                    ax_composite_err.axis('off')
                    try:
                        plt.savefig(plot_filepath_composite_err)
                        print(f"Saved composite error map plot to {plot_filepath_composite_err}")
                    except Exception as e:
                        print(f"Error saving composite error map plot: {e}")
                    plt.close(fig_composite_err)

                if self.config.SAVE_BINARY_EDGE_MAPS and self.final_adaptive_binary_edge_map is not None and self.final_adaptive_binary_edge_map.size > 0: # Use self.config
                    print("\nSaving final adaptive binary edge map...")
                    binary_edge_filename = f"{self.config.IMAGE_NAME}_final_adaptive_binary_edge_map_{params_suffix}.png"
                    binary_edge_filepath = self.image_adaptive_results_dir / binary_edge_filename
                    try:
                        binary_edge_map_uint8 = (self.final_adaptive_binary_edge_map * 255).astype(np.uint8)
                        Image.fromarray(binary_edge_map_uint8).save(binary_edge_filepath)
                        print(f"Saved final adaptive binary edge map to {binary_edge_filepath}")
                    except Exception as e:
                        print(f"Error saving final adaptive binary edge map: {e}")


                # --- NEW: Visualize Reconstruction Results Plot ---
                if self.config.SAVE_MAIN_PLOT: # Reusing this flag for the reconstruction plot
                    print("\nGenerating Reconstruction Results plot...")
                    fig_reco, axes_reco = plt.subplots(1, 4, figsize=(24, 6)) # 4 subplots

                    # Plot 1: Original Image
                    axes_reco[0].imshow(self.original_image, cmap='gray', vmin=0, vmax=1)
                    axes_reco[0].set_title('Original Image')
                    axes_reco[0].axis('off')

                    # Plot 2: Reconstructed Image (Approximation of Original)
                    axes_reco[1].imshow(reassembled_approx_image, cmap='gray', vmin=0, vmax=1)
                    axes_reco[1].set_title(f'Reconstructed Image (Upscale {self.config.UPSCALE_FACTOR}x)')
                    axes_reco[1].axis('off')

                    # Plot 3: Actual Reconstruction Error Heatmap (Normalized - vs Original)
                    im_reco_error = axes_reco[2].imshow(normalized_reconstruction_error_for_plot, cmap=self.config.ERROR_HEATMAP_COLORMAP, origin='upper')
                    fig_reco.colorbar(im_reco_error, ax=axes_reco[2], label='Actual Reconstruction Error (Normalized)')
                    axes_reco[2].set_title('Actual Reconstruction Error Heatmap')
                    axes_reco[2].axis('off')

                    # Plot 4: Reconstructed Image with Segmentation Boundaries
                    self.draw_segmentation_boundaries(reassembled_approx_image, axes_reco[3], color='red', linewidth=1)
                    axes_reco[3].set_title('Reconstructed Image with Segmentation')
                    axes_reco[3].axis('off')

                    plt.tight_layout()
                    plot_filename_reco = f"{self.config.IMAGE_NAME}_reconstruction_results_{params_suffix}.png"
                    plot_filepath_reco = os.path.join(str(self.image_adaptive_results_dir), plot_filename_reco)
                    try:
                        plt.savefig(plot_filepath_reco)
                        print(f"Saved Reconstruction Results plot to {plot_filepath_reco}")
                    except Exception as e:
                        print(f"Error saving Reconstruction Results plot: {e}")
                    plt.close(fig_reco)


                # --- NEW: Visualize Edge Detection Results Plot ---
                # This plot replaces the old SAVE_MAIN_PLOT's edge-related subplots
                if self.config.SAVE_MAIN_PLOT: # Assuming SAVE_MAIN_PLOT also controls this, or you can add a new config flag
                    print("\nGenerating Edge Detection Results plot...")
                    fig_edge, axes_edge = plt.subplots(1, 3, figsize=(18, 6)) # 3 subplots

                    # Plot 1: Original Image
                    axes_edge[0].imshow(self.original_image, cmap='gray', vmin=0, vmax=1)
                    axes_edge[0].set_title('Original Image')
                    axes_edge[0].axis('off')

                    # Plot 2: Chosen Error Map for Detection (Reassembled Composite or single raw error)
                    ax_chosen_error = axes_edge[1]
                    if self.reassembled_composite_error_map is not None and self.reassembled_composite_error_map.size > 0:
                        raw_error_map_keys_for_plot = ['error_original', 'error_smoothed', 'diff_original_poly_smoothed', 'diff_smoothed_poly_original']

                        if original_error_combination_strategy in ['logical_and', 'logical_or']:
                            im_chosen_error = ax_chosen_error.imshow(self.reassembled_composite_error_map, cmap='gray', origin='upper')
                            ax_chosen_error.set_title(f'Chosen Error Map ({original_error_combination_strategy.capitalize()})')
                        elif original_error_combination_strategy in raw_error_map_keys_for_plot:
                            im_chosen_error = ax_chosen_error.imshow(self.reassembled_composite_error_map, cmap=self.config.EDGE_ERROR_HEATMAP_COLORMAP, origin='upper')
                            ax_chosen_error.set_title(f'Chosen Error Map ({original_error_combination_strategy.replace("_", " ").title()})')
                            fig_edge.colorbar(im_chosen_error, ax=ax_chosen_error, label='Normalized Error')
                        else: # 'max' or 'weighted_sum'
                            normalized_composite_for_plot = normalize_error_image(self.reassembled_composite_error_map)
                            im_chosen_error = ax_chosen_error.imshow(normalized_composite_for_plot, cmap=self.config.EDGE_ERROR_HEATMAP_COLORMAP, origin='upper')
                            ax_chosen_error.set_title(f'Chosen Error Map ({original_error_combination_strategy.capitalize()})')
                            fig_edge.colorbar(im_chosen_error, ax=ax_chosen_error, label='Normalized Error')
                        
                        ax_chosen_error.axis('off')
                    else:
                        ax_chosen_error.set_title("Chosen Error Map Unavailable")
                        ax_chosen_error.axis('off')

                    # Plot 3: Final Adaptive Binary Edge Map
                    ax_final_edge = axes_edge[2]
                    if self.final_adaptive_binary_edge_map is not None and self.final_adaptive_binary_edge_map.size > 0:
                        ax_final_edge.imshow(self.final_adaptive_binary_edge_map, cmap='gray')
                        ax_final_edge.set_title(f'Final Adaptive Binary Edge Map\n(Threshold: {self.config.EDGE_THRESHOLD_TYPE})')
                        ax_final_edge.axis('off')
                    else:
                        ax_final_edge.set_title("Binary Edge Map Unavailable")
                        ax_final_edge.axis('off')

                    plt.tight_layout()
                    plot_filename_edge = f"{self.config.IMAGE_NAME}_edge_detection_results_{params_suffix}.png"
                    plot_filepath_edge = os.path.join(str(self.image_adaptive_results_dir), plot_filename_edge)
                    try:
                        plt.savefig(plot_filepath_edge)
                        print(f"Saved Edge Detection Results plot to {plot_filepath_edge}")
                    except Exception as e:
                        print(f"Error saving Edge Detection Results plot: {e}")
                    plt.close(fig_edge)


                # --- Visualize Segment Error Heatmap ---
                if self.config.SAVE_SEGMENT_ERROR_HEATMAP_PLOT:
                    print("\nGenerating segment error heatmap plot...")
                    fig2, ax2 = plt.subplots(1, 1, figsize=(8, 8)) # Create a new figure for this plot

                    self.draw_segment_error_heatmap(ax2, colormap=self.config.SEGMENT_ERROR_COLORMAP)

                    plt.tight_layout()

                    # --- Save the Segment Error Heatmap Plot ---
                    plot_filename_segment_error = f"{self.config.IMAGE_NAME}_segment_error_heatmap_{params_suffix}.png"
                    plot_filepath_segment_error = os.path.join(str(self.image_adaptive_results_dir), plot_filename_segment_error)

                    try:
                        plt.savefig(plot_filepath_segment_error)
                        print(f"\nSaved segment error heatmap plot to {plot_filepath_segment_error}")
                    except Exception as e:
                        print(f"\nError saving segment error heatmap plot to {plot_filepath_segment_error}: {e}")

                    # Close the segment error heatmap figure
                    plt.close(fig2)

                # --- Generate Additional Visualization Plots ---
                if self.config.SAVE_ERROR_DISTRIBUTION_PLOT:
                     self.plot_error_distribution()

                if self.config.SAVE_DEPTH_SEGMENT_COUNT_PLOT:
                     self.plot_depth_segment_count()

                if self.config.SAVE_SEGMENT_SIZE_DISTRIBUTION_PLOT:
                     self.plot_segment_size_distribution()
                
                # --- NEW: Generate Edge Strategy Comparison Plot ---
                if self.config.SAVE_EDGE_STRATEGY_COMPARISON_PLOT and self.config.EDGE_STRATEGIES_TO_COMPARE:
                    print("\nGenerating Edge Strategy Comparison Plot...")
                    num_strategies = len(self.config.EDGE_STRATEGIES_TO_COMPARE)
                    if num_strategies == 0:
                        print("No edge strategies specified for comparison. Skipping comparison plot.")
                    else:
                        # Determine grid size for subplots
                        cols = min(num_strategies, 4) # Max 4 columns for readability
                        rows = (num_strategies + cols - 1) // cols

                        fig_comp, axes_comp = plt.subplots(rows, cols, figsize=(6 * cols, 6 * rows))
                        axes_comp = axes_comp.flatten() # Flatten for easy iteration

                        compared_edge_maps = {} # Store results for comparison

                        raw_error_map_keys = ['error_original', 'error_smoothed', 'diff_original_poly_smoothed', 'diff_smoothed_poly_original']

                        for i, strategy_name in enumerate(self.config.EDGE_STRATEGIES_TO_COMPARE):
                            temp_composite_error_map = None
                            
                            # Re-calculate composite and binary maps for each strategy
                            if strategy_name in ['logical_and', 'logical_or']:
                                binary_maps_for_logical_op = []
                                for key in self.config.LOGICAL_OP_ERROR_KEYS:
                                    if key in normalized_reassembled_errors:
                                        binary_map = apply_threshold(
                                            normalized_reassembled_errors[key],
                                            threshold_type=self.config.EDGE_THRESHOLD_TYPE,
                                            fixed_threshold=self.config.FIXED_EDGE_THRESHOLD
                                        )
                                        binary_maps_for_logical_op.append(binary_map)
                                if binary_maps_for_logical_op:
                                    if strategy_name == 'logical_and':
                                        temp_composite_error_map = np.logical_and.reduce(binary_maps_for_logical_op).astype(np.float32)
                                    else: # 'logical_or'
                                        temp_composite_error_map = np.logical_or.reduce(binary_maps_for_logical_op).astype(np.float32)
                            elif strategy_name in raw_error_map_keys:
                                if strategy_name in normalized_reassembled_errors:
                                    temp_composite_error_map = normalized_reassembled_errors[strategy_name]
                            else: # 'max' or 'weighted_sum'
                                temp_composite_error_map = combine_errors(
                                    normalized_reassembled_errors,
                                    strategy=strategy_name,
                                    weights=self.config.ERROR_COMBINATION_WEIGHTS
                                )

                            if temp_composite_error_map is not None and temp_composite_error_map.size > 0:
                                temp_binary_edge_map = apply_threshold(
                                    temp_composite_error_map,
                                    threshold_type=self.config.EDGE_THRESHOLD_TYPE,
                                    fixed_threshold=self.config.FIXED_EDGE_THRESHOLD
                                )
                                compared_edge_maps[strategy_name] = temp_binary_edge_map
                            else:
                                compared_edge_maps[strategy_name] = np.zeros_like(self.original_image, dtype=np.uint8) # Empty map if generation failed

                        # Plot the collected binary edge maps
                        for i, strategy_name in enumerate(self.config.EDGE_STRATEGIES_TO_COMPARE):
                            ax = axes_comp[i]
                            edge_map = compared_edge_maps.get(strategy_name, np.zeros_like(self.original_image, dtype=np.uint8))
                            ax.imshow(edge_map, cmap='gray')
                            title_text = strategy_name.replace("_", " ").title()
                            if strategy_name in ['logical_and', 'logical_or']:
                                title_text += f"\n(Keys: {', '.join(self.config.LOGICAL_OP_ERROR_KEYS)})"
                            ax.set_title(title_text)
                            ax.axis('off')
                        
                        # Hide unused subplots
                        for j in range(i + 1, len(axes_comp)):
                            fig_comp.delaxes(axes_comp[j])

                        plt.tight_layout()
                        plot_filename_comp = f"{self.config.IMAGE_NAME}_edge_strategy_comparison_{params_suffix}.png"
                        plot_filepath_comp = os.path.join(str(self.image_adaptive_results_dir), plot_filename_comp)
                        try:
                            plt.savefig(plot_filepath_comp)
                            print(f"Saved Edge Strategy Comparison Plot to {plot_filepath_comp}")
                        except Exception as e:
                            print(f"Error saving Edge Strategy Comparison Plot: {e}")
                        plt.close(fig_comp)


            else:
                 print("Reassembled image is empty. Skipping visualization and saving.")


        else:
            print("\nNo segments were successfully processed. Cannot reassemble, plot, or save.")


        print("\nAdaptive image reconstruction process finished.")


# --- Helper function for multiprocessing pool initialization ---
# This initializer now does nothing as we pass data directly to the worker.
def _worker_init():
    """Initializer for worker processes (does nothing now)."""
    pass


if __name__ == "__main__":
    # Ensure the script is run as the main program when using multiprocessing
    # This is crucial on Windows for the 'spawn' start method
    multiprocessing.freeze_support()

    # Instantiate the reconstructor with the loaded configuration
    reconstructor = AdaptivePolynomialReconstructor(config) # Only pass config

    # Run the reconstruction process
    reconstructor.run_reconstruction()
