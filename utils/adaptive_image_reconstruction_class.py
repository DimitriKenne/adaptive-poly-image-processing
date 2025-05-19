# utils/adaptive_image_reconstruction_class.py

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

# Determine the project root dynamically based on the location of this file
# If this file is in 'utils/', the project root is two levels up.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
# Add project root to Python path to allow importing 'config' and 'poly_approx'
sys.path.insert(0, str(PROJECT_ROOT))


# Import configuration from the 'config' directory
try:
    import config.config as config # Import config.py as config
except ImportError:
    print("Error: config/config.py not found. Please ensure your project structure is correct.")
    sys.exit(1)


# Import necessary functions from poly_approx package
# These imports assume 'poly_approx' is a package directly under PROJECT_ROOT
try:
    # image_poly_approximation_segment is now expected to return coefficients
    from poly_approx.image_poly_approximation import image_poly_approximation_segment, save_images
    # Import calculate_error_measure and normalize_error_image from image_reconstruction_metrics
    from poly_approx.image_reconstruction_metrics import calculate_error_measure, normalize_error_image
    # Import evaluate_polynomial_from_coeffs and gen_vanderm2d from poly_projector
    from poly_approx.poly_projector import evaluate_polynomial_from_coeffs, gen_vanderm2d
    # Import graded_lexicographic_multi_indices from polynomial_bases (assuming it's in poly_approx)
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


class AdaptivePolynomialReconstructor:
    """
    Performs adaptive image reconstruction using polynomial approximation
    on recursively subdivided segments.
    """

    def __init__(self, config):
        """
        Initializes the reconstructor with configuration parameters.

        Args:
            config: A module or object containing configuration attributes
                    (e.g., from config.py).
        """
        self.config = config
        self.original_image = None
        self.original_image_shape = None
        # Dictionary to store results from final segments (bbox, coeffs, degree, basis, rectangle, error)
        self.final_segment_data: Dict[str, Dict] = {}
        self.image_adaptive_results_dir = None


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


    def process_segment(
        self,
        segment_bbox: Tuple[int, int, int, int], # (row_start, row_end, col_start, col_end)
        current_depth: int,
        segment_id: str
    ):
        """
        Recursively processes an image segment for polynomial approximation and adaptive refinement.
        Stores polynomial coefficients, basis type, segment bounding boxes, and final error measure
        for terminal segments in self.final_segment_data.

        Args:
            segment_bbox : Tuple[int, int, int, int]
                Bounding box of the current segment (row_start, row_end, col_start, col_end).
            current_depth : int
                Current recursion depth.
            segment_id : str
                Unique identifier for the current segment.
        """
        row_start, row_end, col_start, col_end = segment_bbox
        segment_image = self.original_image[row_start:row_end, col_start:col_end]

        height, width = segment_image.shape

        print(f"\n--- Processing segment {segment_id} at depth {current_depth} with shape {segment_image.shape} ---")

        if height == 0 or width == 0:
            print(f"Skipping empty segment {segment_id}.")
            return # Skip empty segments

        # Define the rectangle for this segment relative to the original image [0,1]x[0,1]
        original_height, original_width = self.original_image_shape
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
                poly_degree=self.config.POLY_DEGREE,
                nodes_method=self.config.NODES_METHOD,
                admissible_mesh_type=self.config.ADMISSIBLE_MESH_TYPE,
                m_cheb=self.config.M_CHEB,
                poly_basis=self.config.POLY_BASIS_USED # Pass poly_basis
            )
            # Extract raw error maps and polynomial coefficients
            raw_error_maps = {
                'error_original': approximation_results.get('error_original', np.zeros_like(segment_image)),
                # 'error_smoothed': approximation_results.get('error_smoothed', np.zeros_like(segment_image)), # Not used for M(S)
                'diff_original_poly_smoothed': approximation_results.get('diff_original_poly_smoothed', np.zeros_like(segment_image)),
                # 'diff_smoothed_poly_original': approximation_results.get('diff_smoothed_poly_original', np.zeros_like(segment_image)) # Not used for M(S)
            }
            # We will use the coefficients from the smoothed approximation for reconstruction and error evaluation
            polynomial_coefficients = approximation_results.get('coefficients_smoothed', np.array([])) # Get the smoothed coefficients

            print(f"Approximation complete for segment {segment_id}.")

        except Exception as e:
            print(f"Error during polynomial approximation for segment {segment_id}: {e}")
            print(f"Stopping processing for segment {segment_id}.")
            # Store empty coefficients and status for this segment if approximation fails
            self.final_segment_data[segment_id] = {
                'bbox': segment_bbox,
                'coefficients': np.array([]), # Store empty coefficients
                'depth': current_depth,
                'poly_degree': self.config.POLY_DEGREE, # Store poly degree used
                'poly_basis': self.config.POLY_BASIS_USED, # Store poly basis used
                'rectangle': segment_rectangle, # Store the segment rectangle
                'final_error_measure': -1.0, # Indicate error during approximation
                'status': 'approximation_failed'
            }
            return


        # --- Step 3: Evaluate Reconstruction Quality (Compute M(S)) ---
        # Calculate the error measure M(S) on a chosen RAW error map to guide subdivision.
        # Using the raw 'error_original' map for M(S) to be more sensitive to original detail.
        error_map_for_measure = raw_error_maps.get('error_original')

        if error_map_for_measure is None:
             print(f"Error: Could not get 'error_original' map for error measure in segment {segment_id}.")
             print(f"Stopping processing for segment {segment_id}.")
             self.final_segment_data[segment_id] = {
                 'bbox': segment_bbox,
                 'coefficients': polynomial_coefficients, # Store the coefficients
                 'depth': current_depth,
                 'poly_degree': self.config.POLY_DEGREE, # Store poly degree used
                 'poly_basis': self.config.POLY_BASIS_USED, # Store poly basis used
                 'rectangle': segment_rectangle, # Store the segment rectangle
                 'final_error_measure': -1.0, # Indicate error during measure calculation
                 'status': 'error_measure_input_missing'
             }
             return

        try:
            # Calculate error measure on the RAW error map
            segment_error_measure = calculate_error_measure(error_map_for_measure, measure_type=self.config.ERROR_MEASURE_TYPE)
            print(f"Error measure ({self.config.ERROR_MEASURE_TYPE}) for segment {segment_id}: {segment_error_measure:.6f}")
        except ValueError as e:
            print(f"Error calculating error measure for segment {segment_id}: {e}")
            print(f"Stopping processing for segment {segment_id}.")
            self.final_segment_data[segment_id] = {
                'bbox': segment_bbox,
                'coefficients': polynomial_coefficients, # Store the coefficients
                'depth': current_depth,
                'poly_degree': self.config.POLY_DEGREE, # Store poly degree used
                'poly_basis': self.config.POLY_BASIS_USED, # Store poly basis used
                'rectangle': segment_rectangle, # Store the segment rectangle
                'final_error_measure': -1.0, # Indicate error during measure calculation
                'status': 'error_measure_failed'
            }
            return
        except Exception as e:
            print(f"An unexpected error occurred calculating error measure for segment {segment_id}: {e}")
            print(f"Stopping processing for segment {segment_id}.")
            self.final_segment_data[segment_id] = {
                'bbox': segment_bbox,
                'coefficients': polynomial_coefficients, # Store the coefficients
                'depth': current_depth,
                'poly_degree': self.config.POLY_DEGREE, # Store poly degree used
                'poly_basis': self.config.POLY_BASIS_USED, # Store poly basis used
                'rectangle': segment_rectangle, # Store the segment rectangle
                'final_error_measure': -1.0, # Indicate error during measure calculation
                'status': 'error_measure_failed'
            }
            return


        # --- Step 4: Decision and Refinement ---
        # Stopping criterion met: Either error is low enough, max depth reached, or segment is too small.
        if segment_error_measure <= self.config.ERROR_THRESHOLD:
            print(f"Stopping for segment {segment_id}: Error measure {segment_error_measure:.6f} <= {self.config.ERROR_THRESHOLD}.")

            # Store the segment bbox, coefficients, and final error measure for this final segment
            self.final_segment_data[segment_id] = {
                'bbox': segment_bbox,
                'coefficients': polynomial_coefficients, # Store the coefficients
                'depth': current_depth,
                'poly_degree': self.config.POLY_DEGREE, # Store poly degree used
                'poly_basis': self.config.POLY_BASIS_USED, # Store poly basis used
                'rectangle': segment_rectangle, # Store the segment rectangle
                'final_error_measure': segment_error_measure, # Store the final error measure
                'status': 'terminated_by_error' # Indicate termination by error
            }

        elif current_depth >= self.config.MAX_DEPTH:
             print(f"Max depth ({self.config.MAX_DEPTH}) reached for segment {segment_id}. Stopping recursion.")
             # Store the segment bbox, coefficients, and final error measure for this final segment
             self.final_segment_data[segment_id] = {
                 'bbox': segment_bbox,
                 'coefficients': polynomial_coefficients, # Store the coefficients
                 'depth': current_depth,
                 'poly_degree': self.config.POLY_DEGREE, # Store poly degree used
                 'poly_basis': self.config.POLY_BASIS_USED, # Store poly basis used
                 'rectangle': segment_rectangle, # Store the segment rectangle
                 'final_error_measure': segment_error_measure, # Store the final error measure
                 'status': 'terminated_by_depth' # Indicate termination by depth
             }

        elif height <= self.config.MIN_SEGMENT_SIZE or width <= self.config.MIN_SEGMENT_SIZE:
             print(f"Segment size ({width}x{height}) below minimum ({self.config.MIN_SEGMENT_SIZE}) for segment {segment_id}. Stopping recursion.")
             # Store the segment bbox, coefficients, and final error measure for this final segment
             self.final_segment_data[segment_id] = {
                 'bbox': segment_bbox,
                 'coefficients': polynomial_coefficients, # Store the coefficients
                 'depth': current_depth,
                 'poly_degree': self.config.POLY_DEGREE, # Store poly degree used
                 'poly_basis': self.config.POLY_BASIS_USED, # Store poly basis used
                 'rectangle': segment_rectangle, # Store the segment rectangle
                 'final_error_measure': segment_error_measure, # Store the final error measure
                 'status': 'terminated_by_size' # Indicate termination by size
             }

        else:
            # Error is too high and stopping criteria not met: Subdivide and recurse.
            print(f"Subdividing segment {segment_id}: Error measure {segment_error_measure:.6f} > {self.config.ERROR_THRESHOLD}.")

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
                self.process_segment( # Recursive call uses self.process_segment
                    sub_bbox,
                    current_depth + 1,
                    sub_segment_id
                )


    def reassemble_results_from_coefficients(self) -> np.ndarray:
        """
        Reassembles the approximation image from stored segment bounding boxes,
        polynomial coefficients, degree, basis type, and rectangle.

        Returns:
            np.ndarray
                The reassembled approximation image.
        """
        if self.original_image_shape is None:
            print("Error: Original image not loaded. Cannot reassemble.")
            return np.array([])

        height, width = self.original_image_shape
        reassembled_approx = np.zeros(self.original_image_shape, dtype=np.float32)

        for segment_id, results in self.final_segment_data.items():
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
                     print(f"Warning: Segment {segment_id} bbox [{row_start}:{row_end}, {col_start}:{col_end}] is out of original image bounds {self.original_image_shape}. Skipping reassembly for this segment.")
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
                    # Get the multi-indices for this degree (needed by gen_vanderm2d internally)
                    # multi_indices = graded_lexicographic_multi_indices(poly_dimension) # Not directly used here but needed by gen_vanderm2d

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

        for segment_id, results in self.final_segment_data.items():
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
            for segment_id, results in self.final_segment_data.items()
            if 'final_error_measure' in results and results['final_error_measure'] >= 0
        }

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

        # Start the recursive processing from the root segment
        self.process_segment(
            initial_bbox,
            0, # Start at depth 0
            initial_segment_id
        )

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
            reassembled_approx_image = self.reassemble_results_from_coefficients()
            print("Reassembly from coefficients complete.")

            if reassembled_approx_image.size > 0:
                # --- Calculate the actual reconstruction error on the reassembled image ---
                actual_reconstruction_error_map = np.abs(self.original_image - reassembled_approx_image)

                # Normalize the actual reconstruction error for visualization
                normalized_reconstruction_error_for_plot = normalize_error_image(actual_reconstruction_error_map)
                print("Actual reconstruction error map calculated and normalized.")

                # --- Save the reassembled approximation image and error map for visualization ---
                params_suffix = ( # Regenerate suffix to ensure consistency
                    f"deg{self.config.POLY_DEGREE}_{self.config.NODES_METHOD}"
                    # f"_sigma{self.config.SIGMA:.1f}" # Include sigma if added to config
                    f"_measure-{self.config.ERROR_MEASURE_TYPE}_errthresh{str(self.config.ERROR_THRESHOLD).replace('.', 'p')}_depth{self.config.MAX_DEPTH}_min{self.config.MIN_SEGMENT_SIZE}"
                    f"_basis{self.config.POLY_BASIS_USED}"
                )

                if self.config.SAVE_RECONSTRUCTED_IMAGE:
                    print("\nSaving reassembled approximation image...")
                    try:
                        save_images({f"{self.config.IMAGE_NAME}_approx_from_coeffs_{params_suffix}.png": reassembled_approx_image}, str(self.image_adaptive_results_dir))
                    except Exception as e:
                        print(f"Error saving reassembled approximation image: {e}")

                if self.config.SAVE_ERROR_MAP_VIZ:
                    print("\nSaving actual error visualization image...")
                    try:
                        save_images({f"{self.config.IMAGE_NAME}_actual_error_viz_from_coeffs_{params_suffix}.png": normalized_reconstruction_error_for_plot}, str(self.image_adaptive_results_dir))
                    except Exception as e:
                        print(f"Error saving actual error visualization image: {e}")


                # --- Visualize Main Results (Original, Reconstructed, Error, Segmentation) ---
                if self.config.SAVE_MAIN_PLOT:
                    print("\nGenerating main plot...")
                    num_subplots = 4
                    fig1, axes1 = plt.subplots(1, num_subplots, figsize=(6 * num_subplots, 6))

                    # Plot 1: Original Image
                    axes1[0].imshow(self.original_image, cmap='gray', vmin=0, vmax=1)
                    axes1[0].set_title('Original Image')
                    axes1[0].axis('off')

                    # Plot 2: Approximate Image (Reconstruction from Coefficients)
                    axes1[1].imshow(reassembled_approx_image, cmap='gray', vmin=0, vmax=1)
                    axes1[1].set_title('Approximate Image (Reconstructed from Coeffs)')
                    axes1[1].axis('off')

                    # Plot 3: Actual Reconstruction Error Heatmap (Normalized)
                    im1 = axes1[2].imshow(normalized_reconstruction_error_for_plot, cmap=self.config.ERROR_HEATMAP_COLORMAP, origin='upper')
                    fig1.colorbar(im1, ax=axes1[2], label='Actual Reconstruction Error (Normalized)')
                    axes1[2].set_title('Actual Reconstruction Error Heatmap')
                    axes1[2].axis('off')

                    # Plot 4: Reconstructed Image with Segmentation Boundaries
                    self.draw_segmentation_boundaries(reassembled_approx_image, axes1[3])

                    plt.tight_layout()

                    # --- Save the Main Plot ---
                    plot_filename_main = f"{self.config.IMAGE_NAME}_adaptive_reconstruction_plot_with_segmentation_{params_suffix}.png"
                    plot_filepath_main = os.path.join(str(self.image_adaptive_results_dir), plot_filename_main)

                    try:
                        plt.savefig(plot_filepath_main)
                        print(f"\nSaved main adaptive image reconstruction plot to {plot_filepath_main}")
                    except Exception as e:
                        print(f"\nError saving main adaptive image reconstruction plot to {plot_filepath_main}: {e}")

                    # Close the main plot figure
                    plt.close(fig1)


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

            else:
                 print("Reassembled image is empty. Skipping visualization and saving.")


        else:
            print("\nNo segments were successfully processed. Cannot reassemble, plot, or save.")


        print("\nAdaptive image reconstruction process finished.")


if __name__ == "__main__":
    # Instantiate the reconstructor with the loaded configuration
    reconstructor = AdaptivePolynomialReconstructor(config)

    # Run the reconstruction process
    reconstructor.run_reconstruction()
