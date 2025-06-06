# edge_detector.py

import sys
import os
import time
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Union, Optional, Callable
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from PIL import Image
import multiprocessing

# Determine the project root dynamically
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import configuration for detector
try:
    import config.config_detector as config_detector
except ImportError as e:
    print(f"Error importing config_detector file: {e}. Please ensure your project structure is correct and config_detector.py exists.")
    sys.exit(1)

# Import necessary functions from poly_approx package
try:
    from poly_approx.image_reconstruction_metrics import normalize_error_image as normalize_error_image_metrics
    from poly_approx.edge_processing import combine_errors, apply_threshold
    _poly_approx_available = True
except ImportError as e:
    print(f"Could not import modules from poly_approx for edge detection: {e}")
    print("Please ensure your project structure is correct and poly_approx is in your Python path.")
    print("Edge detection will be skipped.")
    _poly_approx_available = False

    # Define dummy functions if imports fail
    def normalize_error_image_metrics(error_map, epsilon=1e-8):
         print("Dummy normalize_error_image_metrics called.")
         if error_map.size == 0:
             return np.zeros_like(error_map, dtype=np.float32)
         min_val = np.min(error_map)
         max_val = np.max(error_map)
         if max_val - min_val < epsilon:
             return np.zeros_like(error_map, dtype=np.float32)
         return ((error_map - min_val) / (max_val - min_val + epsilon)).astype(np.float32)

    def combine_errors(error_dict, strategy='max', weights=None):
        print(f"Dummy combine_errors called with strategy: {strategy}")
        if error_dict: return np.zeros_like(list(error_dict.values())[0], dtype=np.float32)
        return np.zeros((100, 100), dtype=np.float32)

    def apply_threshold(image, threshold_type='fixed', fixed_threshold=0.5):
        print(f"Dummy apply_threshold called with threshold_type: {threshold_type}")
        if image.size == 0: return np.zeros_like(image, dtype=np.uint8)
        return np.zeros_like(image, dtype=np.uint8)


class EdgeDetector:
    """
    Performs edge detection based on error maps generated during
    adaptive polynomial image reconstruction.
    """

    def __init__(self, config_obj, image_name: str, original_image_path: Path):
        """
        Initializes the EdgeDetector with configuration parameters.

        Args:
            config_obj: A module or object containing all configuration attributes
                        (e.g., from config_detector.py).
            image_name: The name of the image being processed (e.g., "Shepp_Logan_phantom").
            original_image_path: The path to the original image file.
        """
        self.config = config_obj
        self.image_name = image_name
        self.original_image_path = original_image_path
        self.original_image = None # Will be loaded
        self.original_image_shape = None

        # Directory where edge detection plots will be saved (new separate folder)
        self.image_results_dir = self.config.EDGE_DETECTION_RESULTS_BASE_DIR / self.image_name
        os.makedirs(self.image_results_dir, exist_ok=True) # Ensure it's created

        self.final_segment_data: Dict[str, Dict] = {} # Loaded from .pkl
        self.reassembled_raw_error_maps: Dict[str, np.ndarray] = {}
        self.reassembled_composite_error_map: Optional[np.ndarray] = None
        self.final_adaptive_binary_edge_map: Optional[np.ndarray] = None

        # Apply global Matplotlib settings from config
        plt.rcParams.update(self.config.MATPLOTLIB_PARAMS)

    def load_image_grayscale(self) -> np.ndarray:
        """Loads a grayscale original image and normalizes it to [0, 1]."""
        if not self.original_image_path.exists():
            print(f"Error: Original image not found at {self.original_image_path}. Cannot perform edge detection.")
            return np.array([])

        print(f"Loading original image for context: {self.original_image_path}")
        img = np.array(Image.open(self.original_image_path).convert('L'), dtype=np.float32) / 255.0
        return img

    def load_reconstruction_data(self, reco_params_suffix: str) -> bool:
        """
        Loads the final segment data (coefficients and raw error maps)
        from a pickled file saved by the reconstructor.

        Args:
            reco_params_suffix: The suffix used in the filename to locate the correct .pkl file.

        Returns:
            bool: True if data was loaded successfully, False otherwise.
        """
        # Construct the path to the reconstruction data using the path defined in config_detector
        reco_data_base_dir = self.config.RECONSTRUCTION_RESULTS_BASE_DIR_FOR_LOADING / self.image_name
        data_filename = f"{self.image_name}_segment_data_{reco_params_suffix}.pkl"
        data_filepath = reco_data_base_dir / data_filename

        if not data_filepath.exists():
            print(f"Error: Reconstruction segment data not found at {data_filepath}.")
            print("Please ensure the image_reconstructor.py script has been run")
            print("and SAVE_SEGMENT_DATA is set to True in config_reconstructor.py.")
            return False

        print(f"Loading reconstruction segment data from: {data_filepath}")
        try:
            with open(data_filepath, 'rb') as f:
                self.final_segment_data = pickle.load(f)
            print("Reconstruction segment data loaded successfully.")
            return True
        except Exception as e:
            print(f"Error loading reconstruction segment data: {e}")
            return False

    def reassemble_raw_error_maps_for_all_types(self, original_image_shape: Tuple[int, int], upscale_factor: int = 1) -> None:
        """
        Reassembles all raw error maps (e.g., 'error_original', 'error_smoothed')
        from stored segment data into full-image maps.
        """
        if not self.final_segment_data:
            print("No segment data available. Cannot reassemble error maps.")
            return

        original_height, original_width = original_image_shape
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

                if upscaled_seg_height <= 0 or upscaled_seg_width <= 0:
                    print(f"Warning: Skipping invalid segment dimensions for {segment_id}: {upscaled_seg_width}x{upscaled_seg_height}")
                    continue

                for err_type in error_map_types:
                    if err_type in segment_raw_error_maps and segment_raw_error_maps[err_type].size > 0:
                        segment_error_map = segment_raw_error_maps[err_type]
                        # Resize segment error map to upscaled segment size if needed
                        # Check if resizing is actually necessary
                        if segment_error_map.shape[0] != upscaled_seg_height or segment_error_map.shape[1] != upscaled_seg_width:
                            try:
                                # Resize using PIL for error maps (convert to uint8 for PIL, then back to float32)
                                resized_segment_error = np.array(Image.fromarray((segment_error_map * 255).astype(np.uint8), 'L').resize(
                                    (upscaled_seg_width, upscaled_seg_height), Image.Resampling.LANCZOS
                                ), dtype=np.float32) / 255.0
                            except ValueError as ve:
                                print(f"Error resizing error map for segment {segment_id}, type {err_type}: {ve}. Original shape: {segment_error_map.shape}, Target: ({upscaled_seg_width}, {upscaled_seg_height}). Filling with zeros.")
                                resized_segment_error = np.zeros((upscaled_seg_height, upscaled_seg_width), dtype=np.float32)
                            except Exception as e:
                                print(f"Unexpected error during resizing for segment {segment_id}, type {err_type}: {e}. Filling with zeros.")
                                resized_segment_error = np.zeros((upscaled_seg_height, upscaled_seg_width), dtype=np.float32)
                        else:
                            resized_segment_error = segment_error_map

                        temp_reassembled_error_maps[err_type][upscaled_row_start:upscaled_row_end, upscaled_col_start:upscaled_col_end] = resized_segment_error

        self.reassembled_raw_error_maps = temp_reassembled_error_maps
        print("Reassembled all raw error maps.")

    def plot_chosen_error_map(self, params_suffix: str, threshold: float):
        """Saves the chosen error map plot for LaTeX subfigure."""
        if self.reassembled_composite_error_map is None or self.reassembled_composite_error_map.size == 0:
            print("No composite error map to plot.")
            return

        print("\nSaving Chosen Error Map for Edge Detection plot...")
        fig_chosen_err, ax_chosen_err = plt.subplots(1, 1, figsize=(self.config.PLOT_FIGURE_WIDTH_INCHES, self.config.PLOT_FIGURE_HEIGHT_INCHES))
        
        normalized_composite_for_plot = normalize_error_image_metrics(self.reassembled_composite_error_map)
        im_chosen_error = ax_chosen_err.imshow(normalized_composite_for_plot, cmap=self.config.EDGE_ERROR_HEATMAP_COLORMAP, origin='upper')
        fig_chosen_err.colorbar(im_chosen_error, ax=ax_chosen_err, label='Normalized Error')
        ax_chosen_err.axis('off')
        
        plot_filename_chosen_err = f"{self.image_name}_chosen_error_map_{params_suffix}.{self.config.MATPLOTLIB_PARAMS['savefig.format']}"
        plot_filepath_chosen_err = os.path.join(str(self.image_results_dir), plot_filename_chosen_err)
        try:
            plt.savefig(plot_filepath_chosen_err, dpi=self.config.PLOT_DPI, format=self.config.MATPLOTLIB_PARAMS['savefig.format'])
            print(f"Saved Chosen Error Map for Edge Detection plot to {plot_filepath_chosen_err}")
        except Exception as e:
            print(f"Error saving Chosen Error Map for Edge Detection plot: {e}")
        plt.close(fig_chosen_err)

    def plot_final_binary_edge_map(self, params_suffix: str, threshold: float):
        """Saves the final adaptive binary edge map plot for LaTeX subfigure."""
        if self.final_adaptive_binary_edge_map is None or self.final_adaptive_binary_edge_map.size == 0:
            print("No final binary edge map to plot.")
            return

        print("\nSaving Final Adaptive Binary Edge Map plot (Matplotlib)...")
        fig_final_edge, ax_final_edge = plt.subplots(1, 1, figsize=(self.config.PLOT_FIGURE_WIDTH_INCHES, self.config.PLOT_FIGURE_HEIGHT_INCHES))
        ax_final_edge.imshow(self.final_adaptive_binary_edge_map, cmap='gray')
        ax_final_edge.axis('off')
        
        # This is the preferred name
        plot_filename_final_edge = f"{self.image_name}_final_binary_edge_map_{params_suffix}.{self.config.MATPLOTLIB_PARAMS['savefig.format']}"
        plot_filepath_final_edge = os.path.join(str(self.image_results_dir), plot_filename_final_edge)
        try:
            plt.savefig(plot_filepath_final_edge, dpi=self.config.PLOT_DPI, format=self.config.MATPLOTLIB_PARAMS['savefig.format'])
            print(f"Saved Final Adaptive Binary Edge Map plot to {plot_filepath_final_edge}")
        except Exception as e:
            print(f"Error saving Final Adaptive Binary Edge Map plot: {e}")
        plt.close(fig_final_edge)

    def plot_edge_strategy_comparison(self, params_suffix: str, threshold: float):
        """
        Generates and saves a comparison plot of binary edge maps
        using different error map combination strategies in a 3x2 grid.
        """
        if not self.config.SAVE_EDGE_STRATEGY_COMPARISON_PLOT or not self.config.EDGE_STRATEGIES_TO_COMPARE:
            return

        if not self.reassembled_raw_error_maps:
            print("No reassembled raw error maps available for strategy comparison.")
            return

        print("\nGenerating Edge Strategy Comparison Plot...")
        num_strategies = len(self.config.EDGE_STRATEGIES_TO_COMPARE)
        
        # Set to 3 rows and 2 columns for the subplot grid
        rows = 3
        cols = 2
        
        # Adjust figsize for the new layout (e.g., slightly wider to accommodate 2 columns)
        # Multiply by a factor slightly greater than 1 to ensure enough space for titles/labels
        fig_comp, axes_comp = plt.subplots(rows, cols, figsize=(self.config.PLOT_FIGURE_WIDTH_INCHES * cols * 0.8, self.config.PLOT_FIGURE_HEIGHT_INCHES * rows * 0.8))
        axes_comp = axes_comp.flatten() # Flatten for easy iteration

        compared_edge_maps = {}

        normalized_reassembled_errors_for_comparison = {
            key: normalize_error_image_metrics(err_map)
            for key, err_map in self.reassembled_raw_error_maps.items()
            if err_map is not None and err_map.size > 0
        }

        # Define a mapping for desired display names and also add the (a), (b) labels
        display_name_map = {
            'error_original': '$E_O$',
            'error_smoothed': '$E_S$',
            'diff_original_poly_smoothed': '$D_O$',
            'diff_smoothed_poly_original': '$D_S$',
            'max': 'Max Strategy',
            'weighted_sum': 'Weighted Sum Strategy',
        }
        
        # Create a list of alphabetical labels (a), (b), ...
        alphabetical_labels = [f"({chr(97 + i)})" for i in range(num_strategies)]


        for i, strategy_name in enumerate(self.config.EDGE_STRATEGIES_TO_COMPARE):
            ax = axes_comp[i] # Get the current subplot axis
            temp_composite_error_map = None
            
            if strategy_name in normalized_reassembled_errors_for_comparison:
                temp_composite_error_map = normalized_reassembled_errors_for_comparison[strategy_name]
            else:
                try:
                    temp_composite_error_map = combine_errors(
                        normalized_reassembled_errors_for_comparison,
                        strategy=strategy_name,
                        weights=self.config.ERROR_COMBINATION_WEIGHTS
                    )
                except ValueError as e:
                    print(f"Warning: Skipping strategy '{strategy_name}' due to error: {e}")
                    temp_composite_error_map = None

            if temp_composite_error_map is not None and temp_composite_error_map.size > 0:
                temp_binary_edge_map = apply_threshold(
                    temp_composite_error_map,
                    threshold_type=self.config.EDGE_THRESHOLD_TYPE,
                    fixed_threshold=threshold # Used the passed 'threshold' parameter here!
                )
                compared_edge_maps[strategy_name] = temp_binary_edge_map
            else:
                # If map generation failed, use a blank image for plotting
                if self.original_image_shape:
                    compared_edge_maps[strategy_name] = np.zeros(self.original_image_shape, dtype=np.uint8)
                else:
                    compared_edge_maps[strategy_name] = np.zeros((100,100), dtype=np.uint8) # Fallback

            # Plot the edge map
            edge_map = compared_edge_maps.get(strategy_name, np.zeros_like(self.original_image, dtype=np.uint8))
            ax.imshow(edge_map, cmap='gray')
            
            # Set title with alphabetical label
            title_text = display_name_map.get(strategy_name, strategy_name.replace("_", " ").title())
            ax.set_title(f"{alphabetical_labels[i]} {title_text}") # Add (a), (b) etc. to title
            ax.axis('off') # Hide axes

        # Remove any unused subplots (if num_strategies is less than rows*cols)
        for j in range(num_strategies, len(axes_comp)):
            fig_comp.delaxes(axes_comp[j])

        plt.tight_layout()
        plot_filename_comp = f"{self.image_name}_edge_strategy_comparison_{params_suffix}.{self.config.MATPLOTLIB_PARAMS['savefig.format']}"
        plot_filepath_comp = os.path.join(str(self.image_results_dir), plot_filename_comp)
        try:
            plt.savefig(plot_filepath_comp, dpi=self.config.PLOT_DPI, format=self.config.MATPLOTLIB_PARAMS['savefig.format'])
            print(f"Saved Edge Strategy Comparison Plot to {plot_filepath_comp}")
        except Exception as e:
            print(f"Error saving Edge Strategy Comparison Plot: {e}")
        plt.close(fig_comp)


    def run_detection(self, params_suffix: str, original_image_shape: Tuple[int, int], upscale_factor: int = 1, fixed_threshold_override: Optional[float] = None):
        """
        Runs the complete edge detection process.

        Args:
            params_suffix: The suffix used in the reconstruction's saved data filename.
            original_image_shape: The shape of the original image (height, width).
            upscale_factor: The upscale factor used during reconstruction.
            fixed_threshold_override: Optional. If provided, overrides the
                                      FIXED_EDGE_THRESHOLD from config for this run.
        """
        if not _poly_approx_available:
            print("Skipping edge detection due to missing poly_approx modules.")
            return

        self.original_image_shape = original_image_shape # Store original image shape
        print(f"\nStarting edge detection for {self.image_name}...")
        start_time = time.time()

        # Load the segment data (containing raw error maps)
        if not self.load_reconstruction_data(params_suffix):
            return

        # Override fixed threshold if provided
        current_fixed_threshold = fixed_threshold_override if fixed_threshold_override is not None else self.config.FIXED_EDGE_THRESHOLD
        
        # Reassemble and process error maps for edge detection
        print("\nReassembling raw error maps for edge detection...")
        self.reassemble_raw_error_maps_for_all_types(self.original_image_shape, upscale_factor)

        if self.reassembled_raw_error_maps:
            print("\nCombining reassembled error maps for edge detection...")
            normalized_reassembled_errors = {
                key: normalize_error_image_metrics(err_map)
                for key, err_map in self.reassembled_raw_error_maps.items()
                if err_map is not None and err_map.size > 0
            }

            if normalized_reassembled_errors:
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
                    fixed_threshold=current_fixed_threshold # Use the current (possibly overridden) threshold
                )
                print("Final adaptive binary edge map created.")
            else:
                print("No composite error map to threshold.")
        else:
            print("No reassembled raw error maps to process for edge detection.")

        # --- Save Plots ---
        # We need the original image for plotting context in some cases, so load it if needed.
        if self.original_image is None:
            self.original_image = self.load_image_grayscale()
            if self.original_image.size == 0:
                print("Could not load original image for plotting. Skipping some plots.")
                return


        # Create a params_suffix for edge detection plots to reflect current config
        # This will include the chosen strategy and threshold
        edge_params_suffix = (
            params_suffix +
            f"_edge_strat-{self.config.ERROR_COMBINATION_STRATEGY}"
            f"_edge_thresh-{self.config.EDGE_THRESHOLD_TYPE}"
            + (f"-{str(current_fixed_threshold).replace('.', 'p')}" if self.config.EDGE_THRESHOLD_TYPE == 'fixed' else '')
        )


        if self.config.SAVE_RAW_ERROR_MAPS and self.reassembled_raw_error_maps:
            print("\nSaving reassembled raw error maps...")
            for key, err_map in self.reassembled_raw_error_maps.items():
                if err_map is not None and err_map.size > 0:
                    raw_err_filename = f"{self.image_name}_reassembled_raw_error_{key}_{edge_params_suffix}.{self.config.MATPLOTLIB_PARAMS['savefig.format']}"
                    plot_filepath_raw_err = os.path.join(str(self.image_results_dir), raw_err_filename)
                    
                    fig_raw_err, ax_raw_err = plt.subplots(1, 1, figsize=(self.config.PLOT_FIGURE_WIDTH_INCHES, self.config.PLOT_FIGURE_HEIGHT_INCHES))
                    normalized_raw_err = normalize_error_image_metrics(err_map)
                    im_raw_err = ax_raw_err.imshow(normalized_raw_err, cmap=self.config.EDGE_ERROR_HEATMAP_COLORMAP, origin='upper')
                    fig_raw_err.colorbar(im_raw_err, ax=ax_raw_err, label=f'Normalized Raw Error ({key.replace("_", " ").title()})')
                    ax_raw_err.axis('off')
                    try:
                        plt.savefig(plot_filepath_raw_err, dpi=self.config.PLOT_DPI, format=self.config.MATPLOTLIB_PARAMS['savefig.format'])
                        print(f"Saved reassembled raw error map '{key}' plot to {plot_filepath_raw_err}")
                    except Exception as e:
                        print(f"Error saving reassembled raw error map '{key}' plot: {e}")
                    plt.close(fig_raw_err)

        if self.config.SAVE_COMPOSITE_ERROR_MAP and self.reassembled_composite_error_map is not None and self.reassembled_composite_error_map.size > 0:
            print("\nSaving reassembled composite error map...")
            composite_err_filename = f"{self.image_name}_reassembled_composite_error_{edge_params_suffix}.{self.config.MATPLOTLIB_PARAMS['savefig.format']}"
            plot_filepath_composite_err = os.path.join(str(self.image_results_dir), composite_err_filename)

            fig_composite_err, ax_composite_err = plt.subplots(1, 1, figsize=(self.config.PLOT_FIGURE_WIDTH_INCHES, self.config.PLOT_FIGURE_HEIGHT_INCHES))
            
            normalized_composite_err = normalize_error_image_metrics(self.reassembled_composite_error_map)
            im_composite_err = ax_composite_err.imshow(normalized_composite_err, cmap=self.config.EDGE_ERROR_HEATMAP_COLORMAP, origin='upper')
            fig_composite_err.colorbar(im_composite_err, ax=ax_composite_err, label=f'Normalized Composite Error')
            ax_composite_err.axis('off')
            try:
                plt.savefig(plot_filepath_composite_err, dpi=self.config.PLOT_DPI, format=self.config.MATPLOTLIB_PARAMS['savefig.format'])
                print(f"Saved composite error map plot to {plot_filepath_composite_err}")
            except Exception as e:
                print(f"Error saving composite error map plot: {e}")
            plt.close(fig_composite_err)

        # Individual plots for LaTeX subfigures
        self.plot_chosen_error_map(edge_params_suffix, current_fixed_threshold)
        if self.config.SAVE_BINARY_EDGE_MAPS:
            self.plot_final_binary_edge_map(edge_params_suffix, current_fixed_threshold)
        
        if self.config.SAVE_EDGE_STRATEGY_COMPARISON_PLOT:
            self.plot_edge_strategy_comparison(edge_params_suffix, current_fixed_threshold)

        end_time = time.time()
        print(f"\nEdge detection process finished in {end_time - start_time:.4f} seconds for {self.image_name}.")

if __name__ == "__main__":
    # Example usage:
    # This part would typically be driven by a higher-level script that first runs
    # the reconstructor, then the detector for each image.
    
    # Dummy values for demonstration purposes. In a real scenario, these
    # would come from the reconstructor's output or a main loop.
    dummy_image_name = "Shepp_Logan_phantom" # Example image name
    dummy_original_image_path = config_detector.PROJECT_ROOT / "images" / f"{dummy_image_name}.png"
    
    # You need to manually specify the params_suffix that corresponds to
    # the reconstruction run you want to use for edge detection.
    # This suffix includes reconstruction parameters that determine the .pkl filename.
    # This example suffix assumes the reconstructor used default config_reconstructor parameters
    # from the previous turn when it generated the .pkl file.
    dummy_reco_params_suffix = (
        f"deg{5}_{'full_mesh'}"
        f"_measure-{'rmse'}_errthresh{str(0.0001).replace('.', 'p')}_depth{15}_min{5}"
        f"_basis{1}" # Make sure this matches config_reconstructor.POLY_BASIS_USED during reco
        f"_procs{multiprocessing.cpu_count()}"
    )
    
    # Dummy original image shape and upscale factor
    # These should match what was used during the reconstruction phase.
    # For a phantom, assuming 256x256 often. For spiral, 256x256.
    # In a real run, these would be passed from the reconstruction phase.
    dummy_original_image_shape = (256, 256) # This is a placeholder for `if __name__ == "__main__":`
    dummy_upscale_factor = 3 # This is a placeholder for `if __name__ == "__main__":`

    # Determine fixed threshold based on dummy_image_name
    dummy_fixed_threshold_map = {
        "Shepp_Logan_phantom": 0.08,
        "spiral": 0.05,
        "spiral_and_zigzag": 0.1,
    }
    dummy_fixed_threshold = dummy_fixed_threshold_map.get(dummy_image_name, config_detector.FIXED_EDGE_THRESHOLD)


    detector = EdgeDetector(config_detector, dummy_image_name, dummy_original_image_path)
    detector.run_detection(
        params_suffix=dummy_reco_params_suffix,
        original_image_shape=dummy_original_image_shape,
        upscale_factor=dummy_upscale_factor,
        fixed_threshold_override=dummy_fixed_threshold # Override config for specific image
    )
