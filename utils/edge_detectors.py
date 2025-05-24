# poly_approx/edge_detectors.py

import sys
import os
import time
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches # For drawing rectangles
import matplotlib.cm as cm # For colormap functionality
from PIL import Image
from typing import Dict, List, Tuple, Union, Optional, Callable
import multiprocessing # For parallel processing
import traceback # For worker error reporting
from queue import Queue # For managing segments to process
import pickle # For saving segment data

# Add project root to Python path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import configuration
try:
    import config.edge_config
except ImportError:
    print("Error: config/edge_config.py not found. Ensure project structure is correct.")
    sys.exit(1)

# Import necessary functions from poly_approx
try:
    from poly_approx.image_poly_approximation import image_poly_approximation_segment, save_images
    from poly_approx.edge_processing import (
        normalize_error_image,
        combine_errors,
        apply_threshold,
        calculate_edge_quality_measure,
        calculate_gradient_magnitude,
        get_band_around_edges,
        _scipy_available, # Keep availability flags as they are checked
        _skimage_available,
    )

    _poly_approx_available = True
except ImportError as e:
    print(f"Could not import modules from poly_approx: {e}")
    print("Ensure project structure is correct and poly_approx is in Python path.")
    print("Edge detection functionality will be limited or unavailable.")
    _poly_approx_available = False

    # Define dummy functions if imports fail for robustness
    def image_poly_approximation_segment(*args, **kwargs):
        print("Dummy image_poly_approximation_segment called.")
        dummy_height, dummy_width = 100, 100
        return {
            'error_original': np.zeros((dummy_height, dummy_width)),
            'error_smoothed': np.zeros((dummy_height, dummy_width)),
            'diff_original_poly_smoothed': np.zeros((dummy_height, dummy_width)),
            'diff_smoothed_poly_original': np.zeros((dummy_height, dummy_width)),
            'coefficients_smoothed': np.array([])
        }
    def save_images(*args, **kwargs):
        print("Dummy save_images called.")
    def normalize_error_image(error_map, epsilon=1e-8):
         print("Dummy normalize_error_image called.")
         if error_map.size == 0: return np.zeros_like(error_map, dtype=np.float32)
         min_val = np.min(error_map)
         max_val = np.max(error_map)
         if max_val - min_val < epsilon: return np.zeros_like(error_map, dtype=np.float32)
         return ((error_map - min_val) / (max_val - min_val + epsilon)).astype(np.float32)
    def combine_errors(error_dict, strategy='max', weights=None):
        print(f"Dummy combine_errors called with strategy: {strategy}")
        if error_dict: return np.zeros_like(list(error_dict.values())[0], dtype=np.float32)
        return np.zeros((100, 100), dtype=np.float32)
    def apply_threshold(image, threshold_type='fixed', fixed_threshold=0.5):
        print(f"Dummy apply_threshold called with threshold_type: {threshold_type}")
        if image.size == 0: return np.zeros_like(image, dtype=np.uint8)
        return np.zeros_like(image, dtype=np.uint8)
    def calculate_edge_quality_measure(*args, **kwargs):
        print("Dummy calculate_edge_quality_measure called.")
        return 0.0
    def calculate_gradient_magnitude(*args, **kwargs):
        print("Dummy calculate_gradient_magnitude called.")
        return np.zeros((100, 100), dtype=np.float32)
    def get_band_around_edges(*args, **kwargs):
        print("Dummy get_band_around_edges called.")
        return np.zeros((100, 100), dtype=np.uint8)

    _scipy_available = False
    _skimage_available = False

# Helper function for error measure calculation
def _calculate_error_measure(error_map: np.ndarray, metric_type: str = 'mse') -> float:
    """
    Calculates an error measure (MSE or MAE) from an error map.
    """
    if error_map is None or error_map.size == 0:
        return 0.0
    
    if metric_type == 'mse':
        return np.mean(np.square(error_map))
    elif metric_type == 'mae':
        return np.mean(np.abs(error_map))
    else:
        print(f"Warning: Unknown error metric type '{metric_type}'. Defaulting to MSE.")
        return np.mean(np.square(error_map))


# --- Worker function for multiprocessing ---
def _process_segment_worker(
    segment_task_data: Dict,
) -> Tuple[Dict, Optional[List[Dict]], Optional[str]]:
    """
    Worker function to process a single image segment.
    This function is generic and used by both Simple and Adaptive detectors.

    Args:
        segment_task_data: Dictionary with segment data, rectangle, pixel_coords,
                           config_params, and adaptive info.

    Returns:
        A tuple: results_data, sub_segment_tasks (None), error_message.
    """
    segment_data = segment_task_data['segment_data']
    segment_rectangle = segment_task_data['rectangle']
    pixel_coords = segment_task_data['pixel_coords']
    config_params = segment_task_data['config_params']
    is_adaptive = segment_task_data.get('is_adaptive', False)
    current_depth = segment_task_data.get('current_depth', 0)
    segment_id = segment_task_data.get('segment_id', 'simple_seg')

    height, width = segment_data.shape

    if height == 0 or width == 0:
        return {'status': 'skipped_empty', 'pixel_coords': pixel_coords, 'segment_id': segment_id, 'current_depth': current_depth, 'config_params': config_params}, None, None # Include config_params even for skipped
    try:
        # Polynomial Approximation and Raw Error Maps
        approximation_results = image_poly_approximation_segment(
            image_segment=segment_data,
            rectangle=segment_rectangle,
            poly_degree=config_params['POLY_DEGREE'],
            nodes_method=config_params['NODES_METHOD'],
            admissible_mesh_type=config_params['ADMISSIBLE_MESH_TYPE'],
            m_cheb=config_params['M_CHEB'],
            poly_basis=config_params['POLY_BASIS_USED']
        )
        raw_error_maps = {
            key: approximation_results.get(key, np.zeros_like(segment_data))
            for key in ['error_original', 'error_smoothed', 'diff_original_poly_smoothed', 'diff_smoothed_poly_original']
        }
        polynomial_coefficients = approximation_results.get('coefficients_smoothed', np.array([]))
        results_data = {'raw_error_maps': raw_error_maps, 'coefficients': polynomial_coefficients}

        # Segment-specific Binary Edge Map (needed for quality measure calculation)
        combination_strategy = config_params['ERROR_COMBINATION_STRATEGY']
        edge_threshold_type = config_params['EDGE_THRESHOLD_TYPE']
        fixed_edge_threshold = config_params['FIXED_EDGE_THRESHOLD']
        error_combination_weights = config_params['ERROR_COMBINATION_WEIGHTS']
        logical_op_error_keys = config_params['LOGICAL_OP_ERROR_KEYS']

        segment_binary_edge_map = None
        composite_segment_error_map = None # Also store the composite error map for the segment

        if combination_strategy in ['max', 'weighted_sum']:
            segment_normalized_errors = {
                 key: normalize_error_image(err_map)
                 for key, err_map in raw_error_maps.items()
                 if err_map is not None and err_map.size > 0
            }
            if segment_normalized_errors:
                 composite_segment_error_map = combine_errors(
                     segment_normalized_errors,
                     strategy=combination_strategy,
                     weights=error_combination_weights
                 )
                 if composite_segment_error_map is not None and composite_segment_error_map.size > 0:
                      segment_binary_edge_map = apply_threshold(
                          composite_segment_error_map,
                          threshold_type=edge_threshold_type,
                          fixed_threshold=fixed_edge_threshold
                      )

        elif combination_strategy in ['logical_and', 'logical_or']:
             segment_binary_maps = []
             for key in logical_op_error_keys:
                 if key in raw_error_maps and raw_error_maps[key] is not None and raw_error_maps[key].size > 0:
                     normalized_segment_error_map = normalize_error_image(raw_error_maps[key])
                     if normalized_segment_error_map.size > 0:
                          binary_map = apply_threshold(
                             normalized_segment_error_map,
                             threshold_type=edge_threshold_type,
                             fixed_threshold=fixed_edge_threshold
                          )
                          segment_binary_maps.append(binary_map)

             if segment_binary_maps:
                 composite_segment_error_map = segment_binary_maps[0] # For logical ops, the composite IS the binary result
                 for i in range(1, len(segment_binary_maps)):
                     if composite_segment_error_map.shape == segment_binary_maps[i].shape:
                          if combination_strategy == 'logical_and':
                              composite_segment_error_map = np.logical_and(composite_segment_error_map, segment_binary_maps[i]).astype(np.uint8)
                          elif combination_strategy == 'logical_or':
                               composite_segment_error_map = np.logical_or(composite_segment_error_map, segment_binary_maps[i]).astype(np.uint8)
                 segment_binary_edge_map = composite_segment_error_map # Assign the combined binary map as the segment's binary edge map


        results_data['segment_binary_edge_map'] = segment_binary_edge_map # Store for potential later use/plotting
        results_data['composite_segment_error_map'] = composite_segment_error_map # Store the composite map


        # Adaptive Only: Evaluate Edge Quality M(S) - Old Adaptive Strategy
        # This part is largely for the old adaptive strategy and may not be used by the new one.
        quality_measure = 0.0
        status = 'processed_simple' # Default status for simple processing

        if is_adaptive:
            status = 'processed_adaptive' # Indicate adaptive processing occurred

            # This part is for the old adaptive strategy, new strategy uses error measure directly
            # The calculate_edge_quality_measure function in edge_processing.py
            # calculates gradient magnitude and generates the band internally.
            # We just need to pass the image data and configuration parameters.
            # Pass the segment_binary_edge_map to calculate_edge_quality_measure
            # if the measure type requires it (e.g., for connectivity or density).
            # However, the current implementation of calculate_edge_quality_measure
            # in edge_processing.py seems to generate its own band/edges from
            # image_source_for_band, so we'll stick to the arguments it expects.
            if 'EDGE_QUALITY_MEASURE_TYPE' in config_params and config_params['EDGE_QUALITY_MEASURE_TYPE'] is not None:
                quality_measure = calculate_edge_quality_measure(
                    image_for_quality=raw_error_maps.get('error_original', np.zeros_like(segment_data)),
                    image_source_for_band=segment_data, # Use original segment for band source
                    measure_type=config_params['EDGE_QUALITY_MEASURE_TYPE'],
                    band_width=config_params.get('EDGE_QUALITY_BAND_WIDTH', 100),
                    band_threshold_type=config_params['EDGE_THRESHOLD_TYPE'], # Use edge threshold type for band source
                    fixed_band_threshold=config_params['FIXED_EDGE_THRESHOLD'], # Use fixed edge threshold for band source
                    quantile=None # Quantile calculated globally per depth in main process
                )
                results_data['edge_quality_measure'] = quality_measure

        results_data['pixel_coords'] = pixel_coords
        results_data['status'] = status
        results_data['segment_id'] = segment_id # Ensure segment_id is in results
        results_data['current_depth'] = current_depth # Ensure current_depth is in results
        results_data['config_params'] = config_params # Include config_params in the results


        return results_data, None, None # No sub_segment_tasks returned by worker

    except Exception as e:
        print(f"Worker Error processing segment {segment_id} at {pixel_coords}: {e}")
        traceback.print_exc()
        # Include config_params in the returned dictionary even in case of error
        return {'pixel_coords': pixel_coords, 'status': f'worker_failed: {e}', 'segment_id': segment_id, 'current_depth': current_depth, 'config_params': config_params}, None, str(e)


class SimpleEdgeDetector:
    """
    Performs simple edge detection on the entire image without segmentation.
    Provides methods to retrieve raw error maps and segment-specific binary maps
    from the global combined error map.
    """
    def __init__(self, config=config.edge_config.edge_config):
        self.config = config
        self.original_image: Optional[np.ndarray] = None
        self.original_image_shape: Optional[Tuple[int, int]] = None
        self.raw_error_maps: Dict[str, np.ndarray] = {} # Stores raw error maps for the full image
        self.global_composite_error_map: Optional[np.ndarray] = None # Stores the combined error map for the full image
        self.final_binary_edge_map: Optional[np.ndarray] = None # The final binary edge map for the full image
        self.image_edge_results_dir: Optional[Path] = None

        self.image_edge_results_dir = self.config.IMAGE_EDGE_RESULTS_DIR
        os.makedirs(self.image_edge_results_dir, exist_ok=True)
        print(f"Saving edge detection results to: {self.image_edge_results_dir}")


    def load_image_grayscale(self, image_path: Path) -> np.ndarray:
        """Loads a grayscale image and normalizes it to [0, 1]."""
        if not image_path.exists():
            print(f"Sample image not found at {image_path}. Creating a dummy image.")
            dummy_img = np.zeros((200, 200), dtype=np.uint8)
            dummy_img[50:150, 50:150] = 255 # White square
            os.makedirs(image_path.parent, exist_ok=True)
            Image.fromarray(dummy_img).save(image_path)
            print(f"Dummy image created at {image_path}")

        print(f"Loading image: {image_path}")
        img = np.array(Image.open(image_path).convert('L'), dtype=np.float32) / 255.0
        return img

    def process_full_image(self) -> None:
        """Processes the entire image as a single 'segment' to get raw error maps and binary edge map."""
        if self.original_image is None:
            print("No image loaded to process.")
            return

        print("\nProcessing full image (no segmentation)...")

        height, width = self.original_image_shape
        full_image_pixel_coords = (0, 0, width, height)
        full_image_rectangle = (0.0, 0.0, 1.0, 1.0) # Normalized coordinates for the whole image

        worker_config_params = {
            'POLY_DEGREE': self.config.POLY_DEGREE,
            'NODES_METHOD': self.config.NODES_METHOD,
            'ADMISSIBLE_MESH_TYPE': self.config.ADMISSIBLE_MESH_TYPE,
            'M_CHEB': self.config.M_CHEB,
            'POLY_BASIS_USED': self.config.POLY_BASIS_USED,
            'ERROR_COMBINATION_STRATEGY': self.config.ERROR_COMBINATION_STRATEGY,
            'EDGE_THRESHOLD_TYPE': self.config.EDGE_THRESHOLD_TYPE,
            'FIXED_EDGE_THRESHOLD': self.config.FIXED_EDGE_THRESHOLD,
            'ERROR_COMBINATION_WEIGHTS': self.config.ERROR_COMBINATION_WEIGHTS,
            'LOGICAL_OP_ERROR_KEYS': self.config.LOGICAL_OP_ERROR_KEYS,
            'EDGE_QUALITY_MEASURE_TYPE': self.config.EDGE_QUALITY_MEASURE_TYPE, # Pass old adaptive params too
            'EDGE_QUALITY_BAND_WIDTH': self.config.EDGE_QUALITY_BAND_WIDTH,
        }

        task_data = {
            'segment_data': self.original_image,
            'rectangle': full_image_rectangle,
            'pixel_coords': full_image_pixel_coords,
            'config_params': worker_config_params,
            'is_adaptive': False,
            'segment_id': "full_image", # Unique ID for tracking
            'current_depth': 0
        }

        # Process the single full image using the worker function
        num_processes = self.config.NUM_PROCESSES or os.cpu_count() or 1
        with multiprocessing.Pool(processes=num_processes) as pool:
            result = pool.apply_async(_process_segment_worker, args=(task_data,))
            try:
                processed_results, _, error_message = result.get()
                if error_message:
                    print(f"Worker for full image reported error: {error_message}")
                    self.raw_error_maps = {} # Clear in case of error
                    self.global_composite_error_map = None
                    self.final_binary_edge_map = None
                    return
                
                self.raw_error_maps = processed_results.get('raw_error_maps', {})
                
                # The worker now returns 'composite_segment_error_map' which is the combined error map
                # (or binary map for logical ops) for the segment it processed.
                # For the full image, this is our global composite error map.
                self.global_composite_error_map = processed_results.get('composite_segment_error_map')

                # The final_binary_edge_map is derived from the global composite error map
                if self.global_composite_error_map is not None and self.global_composite_error_map.size > 0:
                    # If the combination strategy was logical_and/or, composite_segment_error_map is already binary
                    if self.config.ERROR_COMBINATION_STRATEGY in ['logical_and', 'logical_or']:
                        self.final_binary_edge_map = self.global_composite_error_map.astype(np.uint8)
                    else:
                        self.final_binary_edge_map = apply_threshold(
                            self.global_composite_error_map,
                            threshold_type=self.config.EDGE_THRESHOLD_TYPE,
                            fixed_threshold=self.config.FIXED_EDGE_THRESHOLD
                        )
                else:
                    self.final_binary_edge_map = None


            except Exception as e:
                print(f"Error processing full image: {e}")
                self.raw_error_maps = {}
                self.global_composite_error_map = None
                self.final_binary_edge_map = None
                return

        if self.final_binary_edge_map is None or self.final_binary_edge_map.size == 0:
            print("Failed to generate a final binary edge map from full image processing.")
            return

        print("Finished processing full image.")

    def get_raw_error_maps(self) -> Dict[str, np.ndarray]:
        """
        Returns the raw error maps generated for the full image.
        These maps are not normalized.
        """
        return self.raw_error_maps

    def get_global_composite_error_map(self) -> Optional[np.ndarray]:
        """
        Returns the combined error map for the full image.
        This map is normalized (0-1) and potentially binary if logical ops were used.
        """
        return self.global_composite_error_map

    def get_final_binary_edge_map(self) -> Optional[np.ndarray]:
        """
        Returns the final binary edge map for the full image (from simple detection).
        """
        return self.final_binary_edge_map

    def get_segment_binary_map_from_global_composite(self, pixel_coords: Tuple[int, int, int, int]) -> Optional[np.ndarray]:
        """
        Produces a binary edge map for a specific segment of the image
        by extracting the corresponding raw error maps from the global simple detector,
        then applying the combination and thresholding logic locally to these segment raw maps.

        Args:
            pixel_coords: A tuple (x_start, y_start, x_end, y_end) defining the segment.

        Returns:
            A NumPy array representing the binary edge map for the segment,
            or None if raw error maps are not available or segment is invalid.
        """
        if not self.raw_error_maps:
            print("Error: Global raw error maps not available. Run process_full_image() first.")
            return None

        x_start, y_start, x_end, y_end = pixel_coords
        
        # Ensure coordinates are within image bounds
        img_height, img_width = self.original_image_shape
        if not (0 <= x_start < x_end <= img_width and 0 <= y_start < y_end <= img_height):
            print(f"Error: Invalid pixel coordinates for segment: {pixel_coords}. Must be within (0,0,{img_width},{img_height}).")
            return None

        # Extract slices of raw error maps for the given segment
        segment_raw_error_maps = {}
        for key, full_map in self.raw_error_maps.items():
            if full_map is not None and full_map.size > 0:
                segment_raw_error_maps[key] = full_map[y_start:y_end, x_start:x_end]
            else:
                # Provide an empty array of correct shape if the map is missing or empty
                segment_raw_error_maps[key] = np.zeros((y_end - y_start, x_end - x_start), dtype=np.float32)

        if not segment_raw_error_maps:
            print(f"Warning: No valid raw error maps found for segment at {pixel_coords}.")
            return np.zeros((y_end - y_start, x_end - x_start), dtype=np.uint8)

        # Re-apply the combination and thresholding logic for this segment
        combination_strategy = self.config.ERROR_COMBINATION_STRATEGY
        edge_threshold_type = self.config.EDGE_THRESHOLD_TYPE
        fixed_edge_threshold = self.config.FIXED_EDGE_THRESHOLD
        error_combination_weights = self.config.ERROR_COMBINATION_WEIGHTS
        logical_op_error_keys = self.config.LOGICAL_OP_ERROR_KEYS

        segment_binary_map = None

        if combination_strategy in ['max', 'weighted_sum']:
            segment_normalized_errors = {
                 key: normalize_error_image(err_map)
                 for key, err_map in segment_raw_error_maps.items()
                 if err_map is not None and err_map.size > 0
            }
            if segment_normalized_errors:
                 composite_segment_error_map = combine_errors(
                     segment_normalized_errors,
                     strategy=combination_strategy,
                     weights=error_combination_weights
                 )
                 if composite_segment_error_map is not None and composite_segment_error_map.size > 0:
                      segment_binary_map = apply_threshold(
                          composite_segment_error_map,
                          threshold_type=edge_threshold_type,
                          fixed_threshold=fixed_edge_threshold
                      )

        elif combination_strategy in ['logical_and', 'logical_or']:
             segment_binary_maps = []
             for key in logical_op_error_keys:
                 if key in segment_raw_error_maps and segment_raw_error_maps[key] is not None and segment_raw_error_maps[key].size > 0:
                     normalized_segment_error_map = normalize_error_image(segment_raw_error_maps[key])
                     if normalized_segment_error_map.size > 0:
                          binary_map = apply_threshold(
                             normalized_segment_error_map,
                             threshold_type=edge_threshold_type,
                             fixed_threshold=fixed_edge_threshold
                          )
                          segment_binary_maps.append(binary_map)

             if segment_binary_maps:
                 # The combined binary map for logical ops is the result
                 combined_logical_map = segment_binary_maps[0]
                 for i in range(1, len(segment_binary_maps)):
                     if combined_logical_map.shape == segment_binary_maps[i].shape:
                          if combination_strategy == 'logical_and':
                              combined_logical_map = np.logical_and(combined_logical_map, segment_binary_maps[i]).astype(np.uint8)
                          elif combination_strategy == 'logical_or':
                               combined_logical_map = np.logical_or(combined_logical_map, segment_binary_maps[i]).astype(np.uint8)
                 segment_binary_map = combined_logical_map


        # Ensure a binary map is always returned, even if empty
        if segment_binary_map is None or segment_binary_map.size == 0:
            return np.zeros((y_end - y_start, x_end - x_start), dtype=np.uint8)

        return segment_binary_map


    def save_results(self) -> None:
        """Saves the final binary edge map and optionally raw error maps."""
        if self.final_binary_edge_map is None or self.final_binary_edge_map.size == 0:
            print("No final binary edge map to save.")
            return

        print("\nSaving results...")
        # Filename suffix now reflects "no segmentation"
        filename_suffix = f"noseg_p{self.config.POLY_DEGREE}_{self.config.NODES_METHOD}_err-{self.config.ERROR_COMBINATION_STRATEGY}_thresh-{self.config.EDGE_THRESHOLD_TYPE}"
        if self.config.EDGE_THRESHOLD_TYPE == 'fixed':
             filename_suffix += f"-{str(self.config.FIXED_EDGE_THRESHOLD).replace('.', 'p')}"
        elif self.config.ERROR_COMBINATION_STRATEGY in ['logical_and', 'logical_or']:
             key_initials = ''.join([key[key.find('_')+1] if '_' in key else key[0] for key in self.config.LOGICAL_OP_ERROR_KEYS])
             filename_suffix += f"_keys-{key_initials}"

        edge_map_filename = f"{self.config.IMAGE_BASE_NAME}_simple_edge_map_{filename_suffix}.png"
        edge_map_filepath = self.image_edge_results_dir / edge_map_filename

        if self.config.SAVE_FINAL_EDGE_MAP:
            try:
                binary_edge_map_uint8 = (self.final_binary_edge_map * 255).astype(np.uint8)
                Image.fromarray(binary_edge_map_uint8).save(edge_map_filepath)
                print(f"Saved final binary edge map to {edge_map_filepath}")
            except Exception as e:
                print(f"Error saving final binary edge map: {e}")

        if self.config.SAVE_RAW_ERROR_MAPS and self.raw_error_maps:
            print("\nSaving raw error maps...")
            for key, err_map in self.raw_error_maps.items():
                if err_map is not None and err_map.size > 0:
                    raw_err_filename = f"{self.config.IMAGE_BASE_NAME}_raw_error_{key}_{filename_suffix}.png"
                    raw_err_filepath = self.image_edge_results_dir / raw_err_filename
                    try:
                         normalized_raw_err = normalize_error_image(err_map)
                         save_images({raw_err_filename: normalized_raw_err}, str(self.image_edge_results_dir))
                    except Exception as e:
                         print(f"Error saving raw error map '{key}': {e}")

        if self.config.SAVE_NORMALIZED_ERROR_MAPS and self.raw_error_maps:
             print("\nSaving normalized error maps...")
             for key, err_map in self.raw_error_maps.items():
                 if err_map is not None and err_map.size > 0:
                     normalized_err_filename = f"{self.config.IMAGE_BASE_NAME}_normalized_error_{key}_{filename_suffix}.png"
                     normalized_err_filepath = self.image_edge_results_dir / normalized_err_filename
                     try:
                          normalized_err = normalize_error_image(err_map)
                          save_images({normalized_err_filename: normalized_err}, str(self.image_edge_results_dir))
                     except Exception as e:
                          print(f"Error saving normalized error map '{key}': {e}")

        if self.config.SAVE_COMPOSITE_ERROR_MAP and self.global_composite_error_map is not None and self.global_composite_error_map.size > 0:
             # Save the global composite error map if it's not a binary map from logical ops
             if self.config.ERROR_COMBINATION_STRATEGY not in ['logical_and', 'logical_or']:
                 print("\nSaving composite error map...")
                 composite_err_filename = f"{self.config.IMAGE_BASE_NAME}_composite_error_{filename_suffix}.png"
                 composite_err_filepath = self.image_edge_results_dir / composite_err_filename
                 try:
                     normalized_composite_err = normalize_error_image(self.global_composite_error_map)
                     save_images({composite_err_filename: normalized_composite_err}, str(self.image_edge_results_dir))
                 except Exception as e:
                     print(f"Error saving composite error map: {e}")
             else:
                 print("Skipping saving composite error map (already binary from logical ops).")


    def plot_results(self) -> None:
        """Generates and saves the main visualization plot."""
        if self.original_image is None or self.final_binary_edge_map is None:
            print("Cannot generate plot: Original image or final edge map is missing.")
            return

        print("\nGenerating main plot...")
        num_subplots = 3
        fig, axes = plt.subplots(1, num_subplots, figsize=(6 * num_subplots, 6))

        # Plot 1: Original Image (no segmentation grid)
        ax1 = axes[0]
        ax1.imshow(self.original_image, cmap='gray', vmin=0, vmax=1)
        ax1.set_title(f'Original Image')
        ax1.axis('off')

        # Plot 2: Heatmap (based on global composite error map for visualization)
        ax2 = axes[1]
        heatmap_data = None
        heatmap_title = ""

        if self.global_composite_error_map is not None and self.global_composite_error_map.size > 0:
            # If logical ops were used, the composite map is already binary, so just display it
            if self.config.ERROR_COMBINATION_STRATEGY in ['logical_and', 'logical_or']:
                heatmap_data = self.global_composite_error_map
                heatmap_title = f'Binary Composite Map ({self.config.ERROR_COMBINATION_STRATEGY.capitalize()})'
                im = ax2.imshow(heatmap_data, cmap='gray', origin='upper') # Use gray for binary
            else:
                heatmap_data = normalize_error_image(self.global_composite_error_map)
                heatmap_title = f'Composite Error Heatmap ({self.config.ERROR_COMBINATION_STRATEGY.capitalize()})'
                im = ax2.imshow(heatmap_data, cmap=self.config.ERROR_HEATMAP_COLORMAP, origin='upper')
            
            fig.colorbar(im, ax=ax2, label=heatmap_title)
            ax2.set_title(f'{heatmap_title}')
            ax2.axis('off')
        else:
             ax2.set_title("Heatmap Data Unavailable")
             ax2.axis('off')

        # Plot 3: Binary Edge Map (from full image processing)
        ax3 = axes[2]
        ax3.imshow(self.final_binary_edge_map, cmap='gray')
        ax3.set_title(f'Detected Edges \n(deg={self.config.POLY_DEGREE}, nodes={self.config.NODES_METHOD})')
        ax3.axis('off')

        plt.tight_layout()

        # Save the Plot
        filename_suffix = f"noseg_p{self.config.POLY_DEGREE}_{self.config.NODES_METHOD}_err-{self.config.ERROR_COMBINATION_STRATEGY}_thresh-{self.config.EDGE_THRESHOLD_TYPE}"
        if self.config.EDGE_THRESHOLD_TYPE == 'fixed':
             filename_suffix += f"-{str(self.config.FIXED_EDGE_THRESHOLD).replace('.', 'p')}"
        elif self.config.ERROR_COMBINATION_STRATEGY in ['logical_and', 'logical_or']:
             key_initials = ''.join([key[key.find('_')+1] if '_' in key else key[0] for key in self.config.LOGICAL_OP_ERROR_KEYS])
             filename_suffix += f"_keys-{key_initials}"

        plot_filename = f"{self.config.IMAGE_BASE_NAME}_simple_edge_detection_plot_{filename_suffix}.png"
        plot_filepath = self.image_edge_results_dir / plot_filename

        if self.config.SAVE_MAIN_PLOT:
            try:
                plt.savefig(plot_filepath)
                print(f"\nSaved plot to {plot_filepath}")
            except Exception as e:
                print(f"\nError saving plot: {e}")

        plt.close(fig)


    def run_detection(self) -> None:
        """Runs the complete simple edge detection process (on the entire image)."""
        if not _poly_approx_available:
            print("Skipping simple edge detection due to missing poly_approx modules.")
            return

        print(f"\nStarting simple edge detection for {self.config.IMAGE_FILENAME} (no segmentation)...")
        start_time = time.time()

        try:
            self.original_image = self.load_image_grayscale(self.config.IMAGE_PATH)
            self.original_image_shape = self.original_image.shape
            print(f"Original image shape: {self.original_image_shape}")
        except Exception as e:
            print(f"Error loading image: {e}")
            return

        # Process the entire image as a single segment
        self.process_full_image()
        
        self.save_results()
        self.plot_results()

        end_time = time.time()
        print(f"\nSimple edge detection finished in {end_time - start_time:.4f} seconds.")


class AdaptiveEdgeDetector:
    """
    Performs adaptive edge detection based on polynomial approximation error.
    This strategy recursively refines approximations in high-error regions.
    """
    def __init__(self, config=config.edge_config.edge_config):
        self.config = config
        self.original_image: Optional[np.ndarray] = None
        self.original_image_shape: Optional[Tuple[int, int]] = None
        self.image_edge_results_dir: Optional[Path] = None
        
        self.global_simple_detector: Optional[SimpleEdgeDetector] = None # To store the initial global detector
        self.final_adaptive_edge_map: Optional[np.ndarray] = None # The final output edge map

        self.final_segment_data: Dict[str, Dict] = {} # Stores data for all terminal segments for plotting/analysis
        self.segment_count: int = 0 # Initialize segment counter

        self.image_edge_results_dir = self.config.IMAGE_EDGE_RESULTS_DIR
        os.makedirs(self.image_edge_results_dir, exist_ok=True)
        print(f"Saving adaptive detection results to: {self.image_edge_results_dir}")


    def load_image_grayscale(self, image_path: Path) -> np.ndarray:
        """Loads a grayscale image and normalizes it to [0, 1]."""
        if not image_path.exists():
            print(f"Sample image not found at {image_path}. Creating a dummy image.")
            dummy_img = np.zeros((200, 200), dtype=np.uint8)
            dummy_img[50:150, 50:150] = 255
            os.makedirs(image_path.parent, exist_ok=True)
            Image.fromarray(dummy_img).save(image_path)
            print(f"Dummy image created at {image_path}")

        print(f"Loading image: {image_path}")
        img = np.array(Image.open(image_path).convert('L'), dtype=np.float32) / 255.0
        return img


    def _process_segment_recursively(
        self,
        segment_image_data: np.ndarray,
        pixel_coords: Tuple[int, int, int, int],
        current_depth: int
    ) -> None:
        """
        Recursively processes image segments for adaptive edge detection.
        """
        x_start, y_start, x_end, y_end = pixel_coords
        height, width = y_end - y_start, x_end - x_start
        segment_id = f"seg_{x_start}_{y_start}_{width}x{height}_d{current_depth}"

        self.segment_count += 1
        if self.segment_count % 100 == 0 or current_depth == 0: # Print every 100 segments or at depth 0
            print(f"Processing segment {self.segment_count} (ID: {segment_id}, Depth: {current_depth}, Size: {width}x{height})")

        # Ensure segment is not empty
        if height == 0 or width == 0:
            self.final_segment_data[segment_id] = {'pixel_coords': pixel_coords, 'status': 'skipped_empty', 'current_depth': current_depth}
            return

        # Initialize local_raw_error_maps with empty arrays for robustness
        # This ensures it always has the expected keys and correct types, even if approximation fails.
        local_raw_error_maps: Dict[str, np.ndarray] = {
            'error_original': np.zeros_like(segment_image_data),
            'error_smoothed': np.zeros_like(segment_image_data),
            'diff_original_poly_smoothed': np.zeros_like(segment_image_data),
            'diff_smoothed_poly_original': np.zeros_like(segment_image_data)
        }
        local_error_measure: float = 0.0

        try:
            approximation_results = image_poly_approximation_segment(
                image_segment=segment_image_data,
                rectangle=(0.0, 0.0, 1.0, 1.0),
                poly_degree=self.config.POLY_DEGREE,
                nodes_method=self.config.NODES_METHOD,
                admissible_mesh_type=self.config.ADMISSIBLE_MESH_TYPE,
                m_cheb=self.config.M_CHEB,
                poly_basis=self.config.POLY_BASIS_USED
            )
            # Update local_raw_error_maps with actual results if successful
            for key in local_raw_error_maps.keys():
                local_raw_error_maps[key] = approximation_results.get(key, np.zeros_like(segment_image_data))

            local_raw_error_map = local_raw_error_maps.get('error_original')
            local_error_measure = _calculate_error_measure(local_raw_error_map, self.config.ERROR_METRIC_TYPE)
            print(f"  Segment {segment_id} - Local Error: {local_error_measure:.6f}") # Debug print
        except Exception as e:
            print(f"Error during local polynomial approximation for segment {segment_id} at {pixel_coords}: {e}")
            traceback.print_exc()
            self.final_segment_data[segment_id] = {'pixel_coords': pixel_coords, 'status': f'local_processing_failed: {e}', 'current_depth': current_depth}
            # local_raw_error_maps is already initialized with zeros, so no need to re-populate here.
            # Default to black for failed segments
            self.final_adaptive_edge_map[y_start:y_end, x_start:x_end] = 0
            return # Exit if approximation fails


        # Condition 1: Smooth Region - Natural Termination (Low Local Error)
        # Now using the single ERROR_THRESHOLD
        if local_error_measure < self.config.ERROR_THRESHOLD:
            # Action: This segment is considered locally smooth, meaning it likely contains no significant edges.
            print(f"  Segment {segment_id} - Terminating: Low Error (No Edge)") # Debug print
            self.final_adaptive_edge_map[y_start:y_end, x_start:x_end] = 0 # Set to black (no edge)
            self.final_segment_data[segment_id] = {'pixel_coords': pixel_coords, 'status': 'terminated_low_error_no_edge', 'current_depth': current_depth, 'local_error': local_error_measure}
            return

        # Condition 2: Forced Termination (Minimum Size or Maximum Depth)
        if height <= self.config.MIN_SEGMENT_SIZE or width <= self.config.MIN_SEGMENT_SIZE or current_depth >= self.config.MAX_DEPTH:
            # Action: This segment has reached a predefined limit for subdivision.
            # A final decision is made for it based on its local simple edge map.
            
            # Re-apply the combination and thresholding logic for this segment
            # to get its binary edge map, similar to _process_segment_worker
            combination_strategy = self.config.ERROR_COMBINATION_STRATEGY
            edge_threshold_type = self.config.EDGE_THRESHOLD_TYPE
            fixed_edge_threshold = self.config.FIXED_EDGE_THRESHOLD
            error_combination_weights = self.config.ERROR_COMBINATION_WEIGHTS
            logical_op_error_keys = self.config.LOGICAL_OP_ERROR_KEYS

            local_binary_edge_map = None

            if combination_strategy in ['max', 'weighted_sum']:
                segment_normalized_errors = {
                     key: normalize_error_image(err_map)
                     for key, err_map in local_raw_error_maps.items()
                     if err_map is not None and err_map.size > 0
                }
                if segment_normalized_errors:
                     composite_segment_error_map = combine_errors(
                         segment_normalized_errors,
                         strategy=combination_strategy,
                         weights=error_combination_weights
                     )
                     if composite_segment_error_map is not None and composite_segment_error_map.size > 0:
                          local_binary_edge_map = apply_threshold(
                              composite_segment_error_map,
                              threshold_type=edge_threshold_type,
                              fixed_threshold=fixed_edge_threshold
                          )

            elif combination_strategy in ['logical_and', 'logical_or']:
                 segment_binary_maps = []
                 for key in logical_op_error_keys:
                     # Ensure key exists in local_raw_error_maps before accessing
                     if key in local_raw_error_maps and local_raw_error_maps[key] is not None and local_raw_error_maps[key].size > 0:
                         normalized_segment_error_map = normalize_error_image(local_raw_error_maps[key])
                         if normalized_segment_error_map.size > 0:
                              binary_map = apply_threshold(
                                 normalized_segment_error_map,
                                 threshold_type=edge_threshold_type,
                                 fixed_threshold=fixed_edge_threshold
                              )
                              segment_binary_maps.append(binary_map)

                 if segment_binary_maps: # Check if list is not empty before accessing index 0
                     combined_logical_map = segment_binary_maps[0]
                     for i in range(1, len(segment_binary_maps)):
                         if combined_logical_map.shape == segment_binary_maps[i].shape:
                              if combination_strategy == 'logical_and':
                                  combined_logical_map = np.logical_and(combined_logical_map, segment_binary_maps[i]).astype(np.uint8)
                              elif combination_strategy == 'logical_or':
                                   combined_logical_map = np.logical_or(combined_logical_map, segment_binary_maps[i]).astype(np.uint8)
                     local_binary_edge_map = combined_logical_map

            # Ensure a binary map is always returned, even if empty
            if local_binary_edge_map is None or local_binary_edge_map.size == 0:
                local_binary_edge_map = np.zeros((height, width), dtype=np.uint8)

            print(f"  Segment {segment_id} - Terminating: Size/Depth (Applying local binary map. Sum of white pixels: {np.sum(local_binary_edge_map)})") # Debug print
            self.final_adaptive_edge_map[y_start:y_end, x_start:x_end] = local_binary_edge_map
            self.final_segment_data[segment_id] = {'pixel_coords': pixel_coords, 'status': 'terminated_by_size_or_depth_edge_decision', 'current_depth': current_depth, 'local_error': local_error_measure}
            return

        # Condition 3: High Error - Subdivision (Default Case if not Terminated)
        # local_error_measure >= self.config.ERROR_THRESHOLD AND not forced termination
        print(f"  Segment {segment_id} - Subdividing") # Debug print
        self.final_segment_data[segment_id] = {'pixel_coords': pixel_coords, 'status': 'subdividing', 'current_depth': current_depth, 'local_error': local_error_measure}

        # Subdivide the current segment into four (quadtree).
        mid_row = y_start + height // 2
        mid_col = x_start + width // 2

        sub_segments_pixel_coords = []
        if mid_row > y_start and mid_col > x_start: sub_segments_pixel_coords.append((x_start, y_start, mid_col, mid_row))
        if mid_row > y_start and x_end > mid_col: sub_segments_pixel_coords.append((mid_col, y_start, x_end, mid_row))
        if y_end > mid_row and mid_col > x_start: sub_segments_pixel_coords.append((x_start, mid_row, mid_col, y_end))
        if y_end > mid_row and x_end > mid_col: sub_segments_pixel_coords.append((mid_col, mid_row, x_end, y_end))

        for sub_pixel_coords in sub_segments_pixel_coords:
            sub_x_start, sub_y_start, sub_x_end, sub_y_end = sub_pixel_coords
            sub_segment_image_data = self.original_image[sub_y_start:sub_y_end, sub_x_start:sub_x_end]
            self._process_segment_recursively(
                sub_segment_image_data,
                sub_pixel_coords,
                current_depth + 1
            )


    def draw_segmentation_boundaries(self, image: np.ndarray, ax: plt.Axes, color='blue', linewidth=1):
        """Draws the bounding boxes of terminal segments on an image plot."""
        if self.original_image_shape is None:
             print("Error: Original image shape not available for drawing boundaries.")
             return

        scale_factor_y = image.shape[0] / self.original_image_shape[0]
        scale_factor_x = image.shape[1] / self.original_image_shape[1]

        for segment_id, results in self.final_segment_data.items():
            # Only draw terminal segments (not 'subdividing' or 'initial_global_check_low_error' if it was the only one)
            if 'pixel_coords' in results and results.get('status') not in ['subdividing', 'initial_global_check_low_error']:
                x_start, y_start, x_end, y_end = results['pixel_coords']
                scaled_y_start = y_start * scale_factor_y
                scaled_y_end = y_end * scale_factor_y
                scaled_x_start = x_start * scale_factor_x
                scaled_x_end = x_end * scale_factor_x

                rect = patches.Rectangle(
                    (scaled_x_start, scaled_y_start),
                    scaled_x_end - scaled_x_start,
                    scaled_y_end - scaled_y_start,
                    linewidth=linewidth,
                    edgecolor=color,
                    facecolor='none'
                )
                ax.add_patch(rect)


    def draw_segment_status_heatmap(self, ax: plt.Axes, colormap='viridis'):
        """Draws a color-coded heatmap of segment status."""
        if self.original_image_shape is None or not self.final_segment_data:
             print("No data to draw segment status heatmap.")
             ax.set_title("Segment Status Heatmap (No data)")
             ax.axis('off')
             return

        height, width = self.original_image_shape
        status_heatmap_image = np.zeros(self.original_image_shape, dtype=np.float32)

        # Map status to a numerical value for visualization
        status_to_value = {
            'terminated_low_error_no_edge': 0.1, # Dark for smooth regions (no edge)
            'terminated_by_size_or_depth_edge_decision': 0.7, # Mid-bright for forced termination with edge decision
            'subdividing': 0.5,                      # Mid-range for regions that were subdivided
            'initial_global_check_low_error': 0.0, # For the case where the whole image is black
            'skipped_empty': 0.05, # Very dark, almost black, for skipped segments
            'local_processing_failed': 0.95 # Very bright for segments where local processing failed
        }
        
        # If the initial global check returned a black image, fill the whole heatmap as background
        if self.final_adaptive_edge_map is not None and not np.any(self.final_adaptive_edge_map > 0) and 'global_terminated' in self.final_segment_data:
            status_heatmap_image.fill(status_to_value['initial_global_check_low_error'])
        else:
            for segment_id, results in self.final_segment_data.items():
                if 'pixel_coords' in results and results.get('status') is not None:
                    x_start, y_start, x_end, y_end = results['pixel_coords']
                    status_val = status_to_value.get(results['status'], 0.0) # Default to 0.0 if status not found
                    status_heatmap_image[y_start:y_end, x_start:x_end] = status_val

        im = ax.imshow(status_heatmap_image, cmap=colormap, origin='upper')
        fig = ax.get_figure()
        # Update legend labels to reflect the new statuses
        labels = {
            'terminated_low_error_no_edge': 'Terminated Low Error (No Edge)',
            'terminated_by_size_or_depth_edge_decision': 'Terminated By Size/Depth (Edge Decision)',
            'subdividing': 'Subdividing',
            'initial_global_check_low_error': 'Initial Global Check Low Error',
            'skipped_empty': 'Skipped (Empty Segment)',
            'local_processing_failed': 'Local Processing Failed'
        }
        # Filter handles and labels to only include statuses that actually occurred
        actual_statuses = sorted(list(set(r['status'] for r in self.final_segment_data.values() if 'status' in r)))
        handles = [plt.Rectangle((0,0),1,1, color=im.cmap(status_to_value[s])) for s in actual_statuses]
        labels_for_legend = [labels[s] for s in actual_statuses]

        ax.legend(handles, labels_for_legend, title="Segment Status", bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
        
        ax.set_title('Segment Status Heatmap')
        ax.axis('off')


    def plot_results(self) -> None:
        """Generates and saves the main visualization plot for adaptive edge detection."""
        if self.original_image is None or self.final_adaptive_edge_map is None:
            print("Cannot generate plot: Original image or final edge map is missing.")
            return

        print("\nGenerating main adaptive edge detection plot...")
        num_subplots = 3
        fig, axes = plt.subplots(1, num_subplots, figsize=(6 * num_subplots, 6))

        # Plot 1: Original Image with Adaptive Segmentation Boundaries
        ax1 = axes[0]
        ax1.imshow(self.original_image, cmap='gray', vmin=0, vmax=1)
        ax1.set_title('Original Image with Adaptive Segmentation')
        ax1.axis('off')
        self.draw_segmentation_boundaries(self.original_image, ax1)

        # Plot 2: Segment Status Heatmap
        ax2 = axes[1]
        self.draw_segment_status_heatmap(ax2, colormap=self.config.ERROR_HEATMAP_COLORMAP)

        # Plot 3: Final Adaptive Edge Map
        ax3 = axes[2]
        ax3.imshow(self.final_adaptive_edge_map, cmap='gray')
        ax3.set_title(f'Final Adaptive Edge Map\n(Error Threshold: {self.config.ERROR_THRESHOLD:.6f})') # Updated title
        ax3.axis('off')

        plt.tight_layout()

        # Save the Plot
        params_suffix = (
            f"p{self.config.POLY_DEGREE}_{self.config.NODES_METHOD}_err-{self.config.ERROR_COMBINATION_STRATEGY}_thresh-{self.config.EDGE_THRESHOLD_TYPE}"
            f"_depth{self.config.MAX_DEPTH}_min{self.config.MIN_SEGMENT_SIZE}_errthresh-{str(self.config.ERROR_THRESHOLD).replace('.', 'p')}" # Updated suffix
        )

        plot_filename = f"{self.config.IMAGE_BASE_NAME}_adaptive_edge_detection_plot_{params_suffix}.png"
        plot_filepath = self.image_edge_results_dir / plot_filename

        if self.config.SAVE_MAIN_PLOT:
            try:
                plt.savefig(plot_filepath, bbox_inches='tight') # Use bbox_inches to include legend
                print(f"\nSaved plot to {plot_filepath}")
            except Exception as e:
                print(f"\nError saving plot: {e}")

        plt.close(fig)

    def save_results(self) -> None:
        """Saves the final adaptive edge map and optionally segment data."""
        if self.final_adaptive_edge_map is None:
            print("No final adaptive edge map to save.")
            return

        print("\nSaving final adaptive edge map...")
        params_suffix = (
            f"p{self.config.POLY_DEGREE}_{self.config.NODES_METHOD}_err-{self.config.ERROR_COMBINATION_STRATEGY}_thresh-{self.config.EDGE_THRESHOLD_TYPE}"
            f"_depth{self.config.MAX_DEPTH}_min{self.config.MIN_SEGMENT_SIZE}_errthresh-{str(self.config.ERROR_THRESHOLD).replace('.', 'p')}" # Updated suffix
        )

        edge_map_filename = f"{self.config.IMAGE_BASE_NAME}_adaptive_edge_map_{params_suffix}.png"
        edge_map_filepath = self.image_edge_results_dir / edge_map_filename

        if self.config.SAVE_FINAL_EDGE_MAP: # Reusing this config for edge map
            try:
                edge_map_uint8 = (self.final_adaptive_edge_map * 255).astype(np.uint8)
                Image.fromarray(edge_map_uint8).save(edge_map_filepath)
                print(f"Saved final adaptive edge map to {edge_map_filepath}")
            except Exception as e:
                print(f"Error saving final adaptive edge map: {e}")

        if self.config.SAVE_ADDITIONAL_PLOTS and self.final_segment_data:
             print("\nSaving final adaptive segment data...")
             data_filename = f"{self.config.IMAGE_BASE_NAME}_adaptive_segment_data_{params_suffix}.pkl"
             data_filepath = self.image_edge_results_dir / data_filename
             try:
                 with open(data_filepath, 'wb') as f:
                     pickle.dump(self.final_segment_data, f)
                 print(f"Saved final adaptive segment data to {data_filepath}")
             except Exception as e:
                 print(f"Error saving final adaptive segment data: {e}")


    def run_detection(self) -> None:
        """Runs the complete adaptive edge detection process."""
        if not _poly_approx_available:
            print("Skipping adaptive edge detection due to missing poly_approx modules.")
            return

        print(f"\nStarting adaptive edge detection for {self.config.IMAGE_FILENAME}...")
        start_time = time.time()

        try:
            self.original_image = self.load_image_grayscale(self.config.IMAGE_PATH)
            self.original_image_shape = self.original_image.shape
            print(f"Original image shape: {self.original_image_shape}")
        except Exception as e:
            print(f"Error loading image: {e}")
            return

        # 1- Create a simple edge detector of the original image.
        # This is primarily to get the global error measure, not to produce a global binary map
        # that directly contributes to the final adaptive map.
        self.global_simple_detector = SimpleEdgeDetector(self.config)
        self.global_simple_detector.original_image = self.original_image
        self.global_simple_detector.original_image_shape = self.original_image_shape
        self.global_simple_detector.process_full_image()
        
        # Retrieve the original raw error map (global error), compute an error measure.
        global_raw_error_map = self.global_simple_detector.get_raw_error_maps().get('error_original')
        global_error_measure = _calculate_error_measure(global_raw_error_map, self.config.ERROR_METRIC_TYPE)

        print(f"Global error measure ({self.config.ERROR_METRIC_TYPE}): {global_error_measure:.6f}")

        # Initial global check using the single ERROR_THRESHOLD
        if global_error_measure < self.config.ERROR_THRESHOLD:
            print(f"Global error measure ({global_error_measure:.6f}) is below threshold ({self.config.ERROR_THRESHOLD:.6f}). Returning black image.")
            self.final_adaptive_edge_map = np.zeros_like(self.original_image, dtype=np.uint8)
            # Add a single entry to final_segment_data to indicate global termination for plotting
            self.final_segment_data['global_terminated'] = {'pixel_coords': (0,0,self.original_image_shape[1],self.original_image_shape[0]), 'status': 'initial_global_check_low_error', 'current_depth': 0}
        else:
            # Initialize the final adaptive edge map with an all-black image
            self.final_adaptive_edge_map = np.zeros_like(self.original_image, dtype=np.uint8)
            print("Global error is high. Starting recursive adaptive refinement.")

            # Start recursive processing from the root segment (full image)
            initial_pixel_coords = (0, 0, self.original_image_shape[1], self.original_image_shape[0])
            self._process_segment_recursively(
                self.original_image,
                initial_pixel_coords,
                current_depth=0
            )
        
        self.save_results()
        self.plot_results()

        end_time = time.time()
        print(f"\nAdaptive edge detection finished in {end_time - start_time:.4f} seconds.")


# Example usage
if __name__ == "__main__":
    multiprocessing.freeze_support()

    # Example of how to use SimpleEdgeDetector
    # simple_detector = SimpleEdgeDetector(config.edge_config.edge_config)
    # simple_detector.run_detection()

    # Example of how to use AdaptiveEdgeDetector
    adaptive_detector = AdaptiveEdgeDetector(config.edge_config.edge_config)
    adaptive_detector.run_detection()
