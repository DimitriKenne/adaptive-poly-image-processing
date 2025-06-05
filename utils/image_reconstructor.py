# image_reconstructor.py

import sys
import os
import time
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Union, Optional, Callable
import numpy as np
import matplotlib.pyplot as plt # Import matplotlib
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
    import config.config_reconstructor as config_reconstructor # Import config_reconstructor.py as config_reconstructor
except ImportError as e:
    print(f"Error importing config file: {e}. Please ensure your project structure is correct and config_reconstructor.py exists.")
    sys.exit(1)


# Import necessary functions from poly_approx package
try:
    from poly_approx.image_poly_approximation import image_poly_approximation_segment, save_images
    from poly_approx.image_reconstruction_metrics import calculate_error_measure, normalize_error_image as normalize_error_image_metrics # Renamed to avoid conflict
    from poly_approx.poly_projector import evaluate_polynomial_from_coeffs, gen_vanderm2d
    from poly_approx.polynomial_bases import graded_lexicographic_multi_indices

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

    def normalize_error_image_metrics(error_map, epsilon=1e-8): # Renamed dummy as well
         print("Dummy normalize_error_image_metrics called.")
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

# --- Helper function for multiprocessing pool initialization ---
def _worker_init():
    """Initializer for worker processes (does nothing now)."""
    pass


# --- Worker function for multiprocessing ---
def process_segment_worker(
    task: Tuple,
    image_segment: np.ndarray,
    segment_rectangle: Tuple[float, float, float, float],
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
    height, width = image_segment.shape

    print(f"\n--- Worker processing segment {segment_id} at depth {current_depth} with shape {image_segment.shape} ---")

    if height == 0 or width == 0:
        print(f"Worker skipping empty segment {segment_id}.")
        return segment_id, {}, None

    # --- Step 1 & 2: Polynomial Approximation and Raw Error Map Calculation ---
    try:
        approximation_results = image_poly_approximation_segment(
            image_segment=image_segment,
            rectangle=segment_rectangle,
            poly_degree=config_params['POLY_DEGREE'],
            nodes_method=config_params['NODES_METHOD'],
            admissible_mesh_type=config_params['ADMISSIBLE_MESH_TYPE'],
            m_cheb=config_params['M_CHEB'],
            poly_basis=config_params['POLY_BASIS_USED']
        )
        if not approximation_results:
            raise ValueError("image_poly_approximation_segment returned empty results.")

        raw_error_maps = {
            'error_original': approximation_results.get('error_original', np.zeros_like(image_segment)),
            'error_smoothed': approximation_results.get('error_smoothed', np.zeros_like(image_segment)),
            'diff_original_poly_smoothed': approximation_results.get('diff_original_poly_smoothed', np.zeros_like(image_segment)),
            'diff_smoothed_poly_original': approximation_results.get('diff_smoothed_poly_original', np.zeros_like(image_segment)),
        }
        polynomial_coefficients_for_reconstruction = approximation_results.get('coefficients_original', np.array([]))

        if polynomial_coefficients_for_reconstruction.size == 0 or np.any(np.isnan(polynomial_coefficients_for_reconstruction)) or np.any(np.isinf(polynomial_coefficients_for_reconstruction)):
            raise ValueError("Polynomial coefficients are invalid (empty, NaN, or Inf).")

        print(f"Worker approximation complete for segment {segment_id}.")

    except Exception as e:
        print(f"Worker Error during polynomial approximation for segment {segment_id}: {e}")
        results_data = {
            'bbox': segment_bbox,
            'coefficients': np.array([]),
            'depth': current_depth,
            'poly_degree': config_params['POLY_DEGREE'],
            'poly_basis': config_params['POLY_BASIS_USED'],
            'rectangle': segment_rectangle,
            'final_error_measure': -1.0,
            'status': 'approximation_failed',
            'raw_error_maps': {key: np.zeros_like(image_segment) for key in ['error_original', 'error_smoothed', 'diff_original_poly_smoothed', 'diff_smoothed_poly_original']}
        }
        return segment_id, results_data, None

    # --- Step 3: Evaluate Reconstruction Quality (Compute M(S)) ---
    error_map_for_measure = raw_error_maps.get('error_original')

    if error_map_for_measure is None:
         print(f"Worker Error: Could not get 'error_original' map for error measure in segment {segment_id}.")
         results_data = {
             'bbox': segment_bbox,
             'coefficients': polynomial_coefficients_for_reconstruction,
             'depth': current_depth,
             'poly_degree': config_params['POLY_DEGREE'],
             'poly_basis': config_params['POLY_BASIS_USED'],
             'rectangle': segment_rectangle,
             'final_error_measure': -1.0,
             'status': 'error_measure_input_missing',
             'raw_error_maps': raw_error_maps
         }
         return segment_id, results_data, None

    try:
        segment_error_measure = calculate_error_measure(error_map_for_measure, measure_type=config_params['ERROR_MEASURE_TYPE'])
        print(f"Worker Error measure ({config_params['ERROR_MEASURE_TYPE']}) for segment {segment_id}: {segment_error_measure:.6f}")
    except Exception as e:
        print(f"Worker Error calculating error measure for segment {segment_id}: {e}")
        results_data = {
            'bbox': segment_bbox,
            'coefficients': polynomial_coefficients_for_reconstruction,
            'depth': current_depth,
            'poly_degree': config_params['POLY_DEGREE'],
            'poly_basis': config_params['POLY_BASIS_USED'],
            'rectangle': segment_rectangle,
            'final_error_measure': -1.0,
            'status': 'error_measure_failed',
            'raw_error_maps': raw_error_maps
        }
        return segment_id, results_data, None


    # --- Step 4: Decision and Refinement ---
    if segment_error_measure <= config_params['ERROR_THRESHOLD']:
        print(f"Worker Stopping for segment {segment_id}: Error measure {segment_error_measure:.6f} <= {config_params['ERROR_THRESHOLD']}.")
        results_data = {
            'bbox': segment_bbox,
            'coefficients': polynomial_coefficients_for_reconstruction,
            'depth': current_depth,
            'poly_degree': config_params['POLY_DEGREE'],
            'poly_basis': config_params['POLY_BASIS_USED'],
            'rectangle': segment_rectangle,
            'final_error_measure': segment_error_measure,
            'status': 'terminated_by_error',
            'raw_error_maps': raw_error_maps
        }
        return segment_id, results_data, None

    elif current_depth >= config_params['MAX_DEPTH']:
         print(f"Worker Max depth ({config_params['MAX_DEPTH']}) reached for segment {segment_id}. Stopping recursion.")
         results_data = {
             'bbox': segment_bbox,
             'coefficients': polynomial_coefficients_for_reconstruction,
             'depth': current_depth,
             'poly_degree': config_params['POLY_DEGREE'],
             'poly_basis': config_params['POLY_BASIS_USED'],
             'rectangle': segment_rectangle,
             'final_error_measure': segment_error_measure,
             'status': 'terminated_by_depth',
             'raw_error_maps': raw_error_maps
         }
         return segment_id, results_data, None

    elif height <= config_params['MIN_SEGMENT_SIZE'] or width <= config_params['MIN_SEGMENT_SIZE']:
         print(f"Worker Segment size ({width}x{height}) below minimum ({config_params['MIN_SEGMENT_SIZE']}) for segment {segment_id}. Stopping recursion.")
         results_data = {
             'bbox': segment_bbox,
             'coefficients': polynomial_coefficients_for_reconstruction,
             'depth': current_depth,
             'poly_degree': config_params['POLY_DEGREE'],
             'poly_basis': config_params['POLY_BASIS_USED'],
             'rectangle': segment_rectangle,
             'final_error_measure': segment_error_measure,
             'status': 'terminated_by_size',
             'raw_error_maps': raw_error_maps
         }
         return segment_id, results_data, None

    else:
        print(f"Worker Subdividing segment {segment_id}: Error measure {segment_error_measure:.6f} > {config_params['ERROR_THRESHOLD']}.")

        mid_row = segment_bbox[0] + height // 2
        mid_col = segment_bbox[2] + width // 2

        sub_segments_bbox = []
        if mid_row > segment_bbox[0] and mid_col > segment_bbox[2]:
            sub_segments_bbox.append((segment_bbox[0], mid_row, segment_bbox[2], mid_col))
        if mid_row > segment_bbox[0] and segment_bbox[3] > mid_col:
             sub_segments_bbox.append((segment_bbox[0], mid_row, mid_col, segment_bbox[3]))
        if segment_bbox[1] > mid_row and mid_col > segment_bbox[2]:
            sub_segments_bbox.append((mid_row, segment_bbox[1], segment_bbox[2], mid_col))
        if segment_bbox[1] > mid_row and segment_bbox[3] > mid_col:
            sub_segments_bbox.append((mid_row, segment_bbox[1], mid_col, segment_bbox[3]))

        sub_segment_tasks = []
        for i, sub_bbox in enumerate(sub_segments_bbox):
             sub_segment_id = f"{segment_id}_{i}"
             sub_segment_tasks.append((sub_bbox, current_depth + 1, sub_segment_id))

        return segment_id, {}, sub_segment_tasks


class AdaptivePolynomialReconstructor:
    """
    Performs adaptive image reconstruction using polynomial approximation
    on recursively subdivided segments, with optional parallel processing
    and higher-resolution output.
    """

    def __init__(self, config_obj):
        """
        Initializes the reconstructor with configuration parameters.

        Args:
            config_obj: A module or object containing all configuration attributes
                        (e.g., from config_reconstructor.py).
        """
        self.config = config_obj
        self.original_image = None
        self.original_image_shape = None
        self.final_segment_data: Dict[str, Dict] = {}
        self.image_adaptive_results_dir = None

        # Apply global Matplotlib settings from config
        plt.rcParams.update(self.config.MATPLOTLIB_PARAMS)


    def load_image_grayscale(self, image_path: Path) -> np.ndarray:
        """Loads a grayscale image and normalizes it to [0, 1]."""
        if not image_path.exists():
            print(f"Sample image not found at {image_path}. Creating a dummy image.")
            dummy_img = np.zeros((256, 256), dtype=np.uint8)
            dummy_img[50:150, 50:150] = 255
            for i in range(256):
                dummy_img[i, :] = i
            os.makedirs(self.config.IMAGE_DIR, exist_ok=True)
            Image.fromarray(dummy_img).save(image_path)
            print(f"Dummy image created at {image_path}")

        print(f"Loading image: {image_path}")
        img = np.array(Image.open(image_path).convert('L'), dtype=np.float32) / 255.0
        return img


    def reassemble_results_from_coefficients(self, upscale_factor: int = 1) -> np.ndarray:
        """
        Reassembles the approximation image from stored segment bounding boxes,
        polynomial coefficients, degree, basis type, and rectangle.
        Optionally upscales the output image.
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

                upscaled_row_start = row_start * upscale_factor
                upscaled_row_end = row_end * upscale_factor
                upscaled_col_start = col_start * upscale_factor
                upscaled_col_end = col_end * upscale_factor

                upscaled_seg_height = upscaled_row_end - upscaled_row_start
                upscaled_seg_width = upscaled_col_end - upscaled_col_start

                if upscaled_row_start < 0 or upscaled_row_end > upscaled_height or upscaled_col_start < 0 or upscaled_col_end > upscaled_width:
                     print(f"Warning: Upscaled segment {segment_id} bbox [{upscaled_row_start}:{upscaled_row_end}, {upscaled_col_start}:{upscaled_col_end}] is out of upscaled image bounds {(upscaled_height, upscaled_width)}. Skipping reassembly for this segment.")
                     continue

                if coefficients.size == 0 or np.any(np.isnan(coefficients)) or np.any(np.isinf(coefficients)):
                     print(f"Warning: Segment {segment_id} has invalid coefficients (empty, NaN, or Inf). Filling with zeros.")
                     reassembled_approx[upscaled_row_start:upscaled_row_end, upscaled_col_start:upscaled_col_end] = np.zeros((upscaled_seg_height, upscaled_seg_width), dtype=np.float32)
                     continue

                xmin, ymin, xmax, ymax = segment_rectangle

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

                    reassembled_approx[upscaled_row_start:upscaled_row_end, upscaled_col_start:upscaled_col_end] = np.clip(approx_flat.reshape(upscaled_seg_height, upscaled_seg_width), 0, 1)

                except Exception as e:
                     print(f"Error evaluating polynomial for segment {segment_id} during reassembly: {e}. Filling with zeros.")
                     reassembled_approx[upscaled_row_start:upscaled_row_end, upscaled_col_start:upscaled_col_end] = np.zeros((upscaled_seg_height, upscaled_seg_width), dtype=np.float32)

        return reassembled_approx

    def draw_segmentation_boundaries(self, image: np.ndarray, ax: plt.Axes, color='red', linewidth=1):
        """
        Draws the bounding boxes of the terminal segments on an image plot.
        """
        ax.imshow(image, cmap='gray', vmin=0, vmax=1)
        ax.axis('off')

        if self.original_image_shape is None and image.shape == self.original_image_shape:
            upscale_factor_for_drawing = 1
        elif self.original_image_shape is not None:
             upscale_factor_for_drawing = image.shape[0] / self.original_image_shape[0]
        else:
             upscale_factor_for_drawing = 1

        for segment_id, results in self.final_segment_data.items():
            if 'bbox' in results:
                row_start, row_end, col_start, col_end = results['bbox']

                scaled_row_start = row_start * upscale_factor_for_drawing
                scaled_row_end = row_end * upscale_factor_for_drawing
                scaled_col_start = col_start * upscale_factor_for_drawing
                scaled_col_end = col_end * upscale_factor_for_drawing # Fixed: changed upscale_contrast to upscale_factor_for_drawing

                rect = patches.Rectangle(
                    (scaled_col_start, scaled_row_start),
                    scaled_col_end - scaled_col_start,
                    scaled_row_end - scaled_row_start,
                    linewidth=linewidth,
                    edgecolor=color,
                    facecolor='none'
                )
                ax.add_patch(rect)

    def draw_segment_error_heatmap(self, ax: plt.Axes, colormap='viridis'):
        """
        Draws a color-coded heatmap of the segmentation, where each segment's color
        intensity reflects its final error measure.
        """
        if self.original_image_shape is None:
             print("Error: Original image shape not available. Cannot draw segment error heatmap.")
             ax.axis('off')
             return

        height, width = self.original_image_shape
        error_heatmap_image = np.zeros(self.original_image_shape, dtype=np.float32)

        final_errors = [
            results['final_error_measure'] for results in self.final_segment_data.values()
            if 'final_error_measure' in results and results['final_error_measure'] >= 0
        ]

        if not final_errors:
            print("No valid final error measures found in segment results. Cannot generate segment error heatmap.")
            ax.axis('off')
            return

        min_error = np.min(final_errors)
        max_error = np.max(final_errors)
        epsilon = 1e-8
        if max_error - min_error < epsilon:
             normalized_errors = np.zeros_like(final_errors)
        else:
             normalized_errors = (final_errors - min_error) / (max_error - min_error + epsilon)


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
                      normalized_err = 0.0
                 else:
                      normalized_err = (error_val - min_valid_error) / (max_valid_error - min_valid_error + epsilon)
                 segment_normalized_error[segment_id] = normalized_err

        for segment_id, results in self.final_segment_data.items():
            if 'bbox' in results and segment_id in segment_normalized_error:
                row_start, row_end, col_start, col_end = results['bbox']
                normalized_err = segment_normalized_error[segment_id]

                error_heatmap_image[row_start:row_end, col_start:col_end] = normalized_err

        im = ax.imshow(error_heatmap_image, cmap=colormap, origin='upper')
        fig = ax.get_figure()
        fig.colorbar(im, ax=ax, label=f'Final Segment Error ({self.config.ERROR_MEASURE_TYPE}, Normalized)')
        ax.axis('off')

    def plot_error_distribution(self):
        """Plots a histogram of the final segment error measures."""
        if not self.final_segment_data:
            print("No segment data available. Cannot plot error distribution.")
            return

        final_errors = [
            results['final_error_measure'] for results in self.final_segment_data.values()
            if 'final_error_measure' in results and results['final_error_measure'] >= 0
        ]

        if not final_errors:
            print("No valid final error measures found. Cannot plot error distribution.")
            return

        fig, ax = plt.subplots(1, 1, figsize=(self.config.PLOT_FIGURE_WIDTH_INCHES, self.config.PLOT_FIGURE_HEIGHT_INCHES))
        ax.hist(final_errors, bins=50, edgecolor='black')
        ax.set_xlabel(f'Final Segment Error ({self.config.ERROR_MEASURE_TYPE})')
        ax.set_ylabel('Number of Segments')
        plt.tight_layout()

        if self.image_adaptive_results_dir:
            params_suffix = (
                f"deg{self.config.POLY_DEGREE}_{self.config.NODES_METHOD}"
                f"_measure-{self.config.ERROR_MEASURE_TYPE}_errthresh{str(self.config.ERROR_THRESHOLD).replace('.', 'p')}_depth{self.config.MAX_DEPTH}_min{self.config.MIN_SEGMENT_SIZE}"
                f"_basis{self.config.POLY_BASIS_USED}"
            )
            plot_filename = f"{self.config.IMAGE_NAME}_error_distribution_{params_suffix}.{self.config.MATPLOTLIB_PARAMS['savefig.format']}"
            plot_filepath = os.path.join(str(self.image_adaptive_results_dir), plot_filename)
            try:
                plt.savefig(plot_filepath, dpi=self.config.PLOT_DPI, format=self.config.MATPLOTLIB_PARAMS['savefig.format'])
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
            depth = results.get('depth', -1)
            if depth >= 0:
                 depth_counts[depth] = depth_counts.get(depth, 0) + 1

        if not depth_counts:
            print("No valid depth information found in segment data. Cannot plot depth vs. segment count.")
            return

        depths = sorted(depth_counts.keys())
        counts = [depth_counts[d] for d in depths]

        fig, ax = plt.subplots(1, 1, figsize=(self.config.PLOT_FIGURE_WIDTH_INCHES, self.config.PLOT_FIGURE_HEIGHT_INCHES))
        ax.bar(depths, counts, edgecolor='black')
        ax.set_xlabel('Depth Level')
        ax.set_ylabel('Number of Segments')
        ax.set_xticks(depths)
        plt.tight_layout()

        if self.image_adaptive_results_dir:
            params_suffix = (
                f"deg{self.config.POLY_DEGREE}_{self.config.NODES_METHOD}"
                f"_measure-{self.config.ERROR_MEASURE_TYPE}_errthresh{str(self.config.ERROR_THRESHOLD).replace('.', 'p')}_depth{self.config.MAX_DEPTH}_min{self.config.MIN_SEGMENT_SIZE}"
                f"_basis{self.config.POLY_BASIS_USED}"
            )
            plot_filename = f"{self.config.IMAGE_NAME}_depth_segment_count_{params_suffix}.{self.config.MATPLOTLIB_PARAMS['savefig.format']}"
            plot_filepath = os.path.join(str(self.image_adaptive_results_dir), plot_filename)
            try:
                plt.savefig(plot_filepath, dpi=self.config.PLOT_DPI, format=self.config.MATPLOTLIB_PARAMS['savefig.format'])
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

        fig, ax = plt.subplots(1, 1, figsize=(self.config.PLOT_FIGURE_WIDTH_INCHES, self.config.PLOT_FIGURE_HEIGHT_INCHES))
        ax.hist(segment_sizes, bins=50, edgecolor='black')
        ax.set_xlabel('Segment Size (Area in Pixels)')
        ax.set_ylabel('Number of Segments')
        plt.tight_layout()

        if self.image_adaptive_results_dir:
            params_suffix = (
                f"deg{self.config.POLY_DEGREE}_{self.config.NODES_METHOD}"
                f"_measure-{self.config.ERROR_MEASURE_TYPE}_errthresh{str(self.config.ERROR_THRESHOLD).replace('.', 'p')}_depth{self.config.MAX_DEPTH}_min{self.config.MIN_SEGMENT_SIZE}"
                f"_basis{self.config.POLY_BASIS_USED}"
            )
            plot_filename = f"{self.config.IMAGE_NAME}_segment_size_distribution_{params_suffix}.{self.config.MATPLOTLIB_PARAMS['savefig.format']}"
            plot_filepath = os.path.join(str(self.image_adaptive_results_dir), plot_filename)
            try:
                plt.savefig(plot_filepath, dpi=self.config.PLOT_DPI, format=self.config.MATPLOTLIB_PARAMS['savefig.format'])
                print(f"Saved segment size distribution plot to {plot_filepath}")
            except Exception as e:
                print(f"Error saving segment size distribution plot: {plot_filepath}: {e}")
        else:
             print("Results directory not set. Skipping saving segment size distribution plot.")

        plt.close(fig)


    def run_reconstruction(self) -> Tuple[str, Tuple[int, int], int]:
        """
        Runs the complete adaptive image reconstruction process.
        
        Returns:
            Tuple[str, Tuple[int, int], int]: A tuple containing the params_suffix
            generated for this run, the original image shape (height, width),
            and the upscale factor used.
        """
        if not _poly_approx_available:
            print("Skipping adaptive image reconstruction due to missing poly_approx modules.")
            return "", (0,0), 1 # Return dummy values if not available

        try:
            self.original_image = self.load_image_grayscale(self.config.IMAGE_PATH)
            self.original_image_shape = self.original_image.shape
        except FileNotFoundError as e:
            print(f"Error loading image: {e}")
            return "", (0,0), 1
        except Exception as e:
            print(f"An unexpected error occurred loading the image: {e}")
            return "", (0,0), 1

        initial_bbox = (0, self.original_image_shape[0], 0, self.original_image_shape[1])
        initial_segment_id = "root"

        # Use the new reconstruction results directory
        self.image_adaptive_results_dir = self.config.RECONSTRUCTION_RESULTS_BASE_DIR / self.config.IMAGE_NAME
        os.makedirs(self.image_adaptive_results_dir, exist_ok=True)
        print(f"Saving results to: {self.image_adaptive_results_dir}")

        print(f"\nStarting adaptive image reconstruction for {self.config.IMAGE_FILENAME}...")
        start_time = time.time()

        num_processes = self.config.NUM_PROCESSES
        if num_processes is None or num_processes <= 0:
            num_processes = 1

        print(f"Using {num_processes} processes for segment processing.")

        segment_queue = Queue()
        row_start, row_end, col_start, col_end = initial_bbox
        initial_segment_image = self.original_image[row_start:row_end, col_start:col_end]
        original_height, original_width = self.original_image_shape
        initial_segment_rectangle = (
            col_start / original_width,
            row_start / original_height,
            col_end / original_width,
            row_end / original_height
        )
        segment_queue.put((initial_bbox, 0, initial_segment_id, initial_segment_image, initial_segment_rectangle))

        results_from_workers = []

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
        }

        with multiprocessing.Pool(processes=num_processes) as pool:
            active_results = []
            if not segment_queue.empty():
                 task_with_data = segment_queue.get()
                 result = pool.apply_async(process_segment_worker, args=(task_with_data[:3], task_with_data[3], task_with_data[4], worker_config_params))
                 active_results.append(result)

            while active_results or not segment_queue.empty():
                completed_results = [r for r in active_results if r.ready()]
                active_results = [r for r in active_results if not r.ready()]

                for result in completed_results:
                    try:
                        segment_id, results_data, sub_segment_tasks = result.get()
                        if results_data:
                            self.final_segment_data[segment_id] = results_data
                        if sub_segment_tasks:
                            for sub_task_bbox, sub_task_depth, sub_task_id in sub_segment_tasks:
                                sub_segment_image = self.original_image[sub_task_bbox[0]:sub_task_bbox[1], sub_task_bbox[2]:sub_task_bbox[3]]
                                sub_segment_rectangle = (
                                    sub_task_bbox[2] / original_width,
                                    sub_task_bbox[0] / original_height,
                                    sub_task_bbox[3] / original_width,
                                    sub_task_bbox[1] / original_height
                                )
                                segment_queue.put((sub_task_bbox, sub_task_depth, sub_task_id, sub_segment_image, sub_segment_rectangle))

                    except Exception as e:
                        print(f"Error collecting result from worker: {e}")

                while not segment_queue.empty() and len(active_results) < num_processes * 2:
                    task_with_data = segment_queue.get()
                    result = pool.apply_async(process_segment_worker, args=(task_with_data[:3], task_with_data[3], task_with_data[4], worker_config_params))
                    active_results.append(result)

                if not active_results and segment_queue.empty():
                     break
                time.sleep(0.01)


        end_time = time.time()
        print(f"\nAdaptive image reconstruction finished in {end_time - start_time:.4f} seconds.")
        print(f"Processed {len(self.final_segment_data)} final segments.")

        params_suffix = (
            f"deg{self.config.POLY_DEGREE}_{self.config.NODES_METHOD}"
            f"_measure-{self.config.ERROR_MEASURE_TYPE}_errthresh{str(self.config.ERROR_THRESHOLD).replace('.', 'p')}_depth{self.config.MAX_DEPTH}_min{self.config.MIN_SEGMENT_SIZE}"
            f"_basis{self.config.POLY_BASIS_USED}"
            f"_procs{num_processes}"
        )

        if self.config.SAVE_SEGMENT_DATA and self.final_segment_data:
            print(f"Saving segment data and coefficients to: {self.image_adaptive_results_dir}")

            data_filename = f"{self.config.IMAGE_NAME}_segment_data_{params_suffix}.pkl"
            data_filepath = os.path.join(str(self.image_adaptive_results_dir), data_filename)

            try:
                with open(data_filepath, 'wb') as f:
                    pickle.dump(self.final_segment_data, f)
                print(f"Saved segment data and coefficients to {data_filepath}")
                print("This file contains the bounding boxes, polynomial degree, basis type, and coefficients for each terminal segment.")
                print("It can be used as the 'compressed' representation for compression analysis and for subsequent edge detection.")
            except Exception as e:
                print(f"Error saving segment data and coefficients to {data_filepath}: {e}")


        print("\nReassembling final approximation image from coefficients...")
        if self.final_segment_data:
            reassembled_approx_image = self.reassemble_results_from_coefficients(self.config.UPSCALE_FACTOR)
            print("Reassembly from coefficients complete.")

            if reassembled_approx_image.size > 0:
                reassembled_approx_original_size = np.array(Image.fromarray((reassembled_approx_image * 255).astype(np.uint8), 'L').resize(
                    (self.original_image_shape[1], self.original_image_shape[0]), Image.Resampling.LANCZOS
                ), dtype=np.float32) / 255.0

                actual_reconstruction_error_map = np.abs(self.original_image - reassembled_approx_original_size)
                normalized_reconstruction_error_for_plot = normalize_error_image_metrics(actual_reconstruction_error_map)
                print("Actual reconstruction error map calculated and normalized.")

                # params_suffix for filenames (includes upscale factor for plots)
                plot_params_suffix = (
                    params_suffix +
                    f"_upscale{self.config.UPSCALE_FACTOR}"
                )

                # --- Save Individual Plots for LaTeX Subfigures (Reconstruction Specific) ---
                if self.config.SAVE_MAIN_PLOT:
                    print("\nSaving Original Image plot...")
                    fig_orig, ax_orig = plt.subplots(1, 1, figsize=(self.config.PLOT_FIGURE_WIDTH_INCHES, self.config.PLOT_FIGURE_HEIGHT_INCHES))
                    ax_orig.imshow(self.original_image, cmap='gray', vmin=0, vmax=1)
                    ax_orig.axis('off')
                    plot_filename_orig = f"{self.config.IMAGE_NAME}_original_image_{plot_params_suffix}.{self.config.MATPLOTLIB_PARAMS['savefig.format']}"
                    plot_filepath_orig = os.path.join(str(self.image_adaptive_results_dir), plot_filename_orig)
                    try:
                        plt.savefig(plot_filepath_orig, dpi=self.config.PLOT_DPI, format=self.config.MATPLOTLIB_PARAMS['savefig.format'])
                        print(f"Saved Original Image plot to {plot_filepath_orig}")
                    except Exception as e:
                        print(f"Error saving Original Image plot: {e}")
                    plt.close(fig_orig)

                if self.config.SAVE_MAIN_PLOT:
                    print("\nSaving Reconstructed Image plot...")
                    fig_reco, ax_reco = plt.subplots(1, 1, figsize=(self.config.PLOT_FIGURE_WIDTH_INCHES, self.config.PLOT_FIGURE_HEIGHT_INCHES))
                    ax_reco.imshow(reassembled_approx_image, cmap='gray', vmin=0, vmax=1)
                    ax_reco.axis('off')
                    plot_filename_reco = f"{self.config.IMAGE_NAME}_reconstructed_image_{plot_params_suffix}.{self.config.MATPLOTLIB_PARAMS['savefig.format']}"
                    plot_filepath_reco = os.path.join(str(self.image_adaptive_results_dir), plot_filename_reco)
                    try:
                        plt.savefig(plot_filepath_reco, dpi=self.config.PLOT_DPI, format=self.config.MATPLOTLIB_PARAMS['savefig.format'])
                        print(f"Saved Reconstructed Image plot to {plot_filepath_reco}")
                    except Exception as e:
                        print(f"Error saving Reconstructed Image plot: {e}")
                    plt.close(fig_reco)

                if self.config.SAVE_MAIN_PLOT:
                    print("\nSaving Actual Reconstruction Error Heatmap plot...")
                    fig_err, ax_err = plt.subplots(1, 1, figsize=(self.config.PLOT_FIGURE_WIDTH_INCHES, self.config.PLOT_FIGURE_HEIGHT_INCHES))
                    im_err = ax_err.imshow(normalized_reconstruction_error_for_plot, cmap=self.config.ERROR_HEATMAP_COLORMAP, origin='upper')
                    fig_err.colorbar(im_err, ax=ax_err, label='Actual Reconstruction Error (Normalized)')
                    ax_err.axis('off')
                    plot_filename_err = f"{self.config.IMAGE_NAME}_actual_error_heatmap_{plot_params_suffix}.{self.config.MATPLOTLIB_PARAMS['savefig.format']}"
                    plot_filepath_err = os.path.join(str(self.image_adaptive_results_dir), plot_filename_err)
                    try:
                        plt.savefig(plot_filepath_err, dpi=self.config.PLOT_DPI, format=self.config.MATPLOTLIB_PARAMS['savefig.format'])
                        print(f"Saved Actual Reconstruction Error Heatmap plot to {plot_filepath_err}")
                    except Exception as e:
                        print(f"Error saving Actual Reconstruction Error Heatmap plot: {e}")
                    plt.close(fig_err)

                if self.config.SAVE_SEGMENT_ERROR_HEATMAP_PLOT:
                    print("\nGenerating Reconstructed Image with Segmentation plot...")
                    fig_reco_seg, ax_reco_seg = plt.subplots(1, 1, figsize=(self.config.PLOT_FIGURE_WIDTH_INCHES, self.config.PLOT_FIGURE_HEIGHT_INCHES))
                    self.draw_segmentation_boundaries(reassembled_approx_image, ax_reco_seg, color='red', linewidth=1)
                    ax_reco_seg.axis('off')

                    plt.tight_layout()
                    plot_filename_reco_seg = f"{self.config.IMAGE_NAME}_reconstruction_segmentation_{plot_params_suffix}.{self.config.MATPLOTLIB_PARAMS['savefig.format']}"
                    plot_filepath_reco_seg = os.path.join(str(self.image_adaptive_results_dir), plot_filename_reco_seg)
                    try:
                        plt.savefig(plot_filepath_reco_seg, dpi=self.config.PLOT_DPI, format=self.config.MATPLOTLIB_PARAMS['savefig.format'])
                        print(f"Saved Reconstructed Image with Segmentation plot to {plot_filepath_reco_seg}")
                    except Exception as e:
                        print(f"Error saving Reconstructed Image with Segmentation plot: {e}")
                    plt.close(fig_reco_seg)

                if self.config.SAVE_SEGMENT_ERROR_HEATMAP_PLOT:
                    print("\nGenerating segment error heatmap plot...")
                    fig2, ax2 = plt.subplots(1, 1, figsize=(self.config.PLOT_FIGURE_WIDTH_INCHES, self.config.PLOT_FIGURE_HEIGHT_INCHES))
                    self.draw_segment_error_heatmap(ax2, colormap=self.config.SEGMENT_ERROR_COLORMAP)
                    plt.tight_layout()
                    plot_filename_segment_error = f"{self.config.IMAGE_NAME}_segment_error_heatmap_{plot_params_suffix}.{self.config.MATPLOTLIB_PARAMS['savefig.format']}"
                    plot_filepath_segment_error = os.path.join(str(self.image_adaptive_results_dir), plot_filename_segment_error)
                    try:
                        plt.savefig(plot_filepath_segment_error, dpi=self.config.PLOT_DPI, format=self.config.MATPLOTLIB_PARAMS['savefig.format'])
                        print(f"\nSaved segment error heatmap plot to {plot_filepath_segment_error}")
                    except Exception as e:
                        print(f"\nError saving segment error heatmap plot to {plot_filepath_segment_error}: {e}")
                    plt.close(fig2)

                if self.config.SAVE_ERROR_DISTRIBUTION_PLOT:
                     self.plot_error_distribution()

                if self.config.SAVE_DEPTH_SEGMENT_COUNT_PLOT:
                     self.plot_depth_segment_count()

                if self.config.SAVE_SEGMENT_SIZE_DISTRIBUTION_PLOT:
                     self.plot_segment_size_distribution()
            else:
                 print("Reassembled image is empty. Skipping visualization and saving.")
        else:
            print("\nNo segments were successfully processed. Cannot reassemble, plot, or save.")

        return params_suffix, self.original_image_shape, self.config.UPSCALE_FACTOR


# --- Helper function for multiprocessing pool initialization ---
def _worker_init():
    """Initializer for worker processes (does nothing now)."""
    pass


if __name__ == "__main__":
    # This block is for testing image_reconstructor.py independently.
    # For full workflow, use run_experiments.py
    reconstructor = AdaptivePolynomialReconstructor(config_reconstructor)
    _ = reconstructor.run_reconstruction() # Capture returns but don't use them here
