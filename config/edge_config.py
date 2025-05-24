# config/edge_config.py

import os
from pathlib import Path
from typing import Dict, List, Optional # Import necessary types

# Determine the project root dynamically based on the location of this file
# If this file is in 'config/', the project root is one level up.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Define the base path for the results folder
BASE_RESULTS_DIR = PROJECT_ROOT / "results"

# Define the subfolder for edge detection results (both simple and adaptive)
EDGE_DETECTION_RESULTS_BASE_DIR = BASE_RESULTS_DIR / "edge_detection_tests"

# Ensure the base results directory for edge detection tests exists
os.makedirs(EDGE_DETECTION_RESULTS_BASE_DIR, exist_ok=True)


class EdgeConfig:
    """
    Configuration parameters for both simple and adaptive edge detection.
    """

    # --- General Image and Results Settings ---
    IMAGE_FILENAME: str = "spiral_zigzag.png" # Default image filename
    # The full path to the image will be constructed using PROJECT_ROOT / "images" / IMAGE_FILENAME

    # --- Polynomial Approximation Parameters (applied per segment in both methods) ---
    POLY_DEGREE: int = 2       # Degree of polynomial approximation
    NODES_METHOD: str = 'leja' # Node selection method ('full_mesh', 'leja', 'fekete', 'padua')
    # Admissible mesh type (only relevant for 'leja', 'fekete', 'full_mesh')
    ADMISSIBLE_MESH_TYPE: str = 'cheb' # Options: 'cheb', 'uni'
    M_CHEB: int = 2            # Parameter 'm' for Chebyshev mesh construction (m > 1)
    POLY_BASIS_USED: int = 1   # Polynomial basis type (e.g., 1 for shifted monomials)


    # --- Error Map Combination Strategy (applied per segment) ---
    # Strategy for combining the different raw error maps ('error_original', 'error_smoothed', etc.)
    # Options: 'max', 'weighted_sum', 'logical_and', 'logical_or'
    ERROR_COMBINATION_STRATEGY: str = 'max'

    # Weights for 'weighted_sum' strategy (keys must match raw error map keys)
    # Only used if ERROR_COMBINATION_STRATEGY is 'weighted_sum'
    ERROR_COMBINATION_WEIGHTS: Dict[str, float] = {
        'error_original': 0.1,
        'error_smoothed': 0.2,
        'diff_original_poly_smoothed': 0.4,
        'diff_smoothed_poly_original': 0.3,
    }
    # Note: Weights will be normalized to sum to 1.0 if they don't.

    # Which raw error maps to use for 'logical_and' or 'logical_or' strategies
    # Provide a list of keys from ['error_original', 'error_smoothed', 'diff_original_poly_smoothed', 'diff_smoothed_poly_original']
    # These raw maps will be normalized (over the segment for adaptive, over full image for simple before thresholding)
    # before thresholding and logical combination.
    LOGICAL_OP_ERROR_KEYS: List[str] = ['error_original', 'diff_original_poly_smoothed']


    # --- Final Edge Thresholding Parameters (applied per segment after combination) ---
    # Type of thresholding to apply to the combined error map (or individual maps for logical ops)
    # 'fixed': Use a fixed threshold (0.0 to 1.0)
    # 'otsu': Use Otsu's method (requires scikit-image)
    EDGE_THRESHOLD_TYPE: str = 'otsu' # Changed to Otsu for better adaptive thresholding
    FIXED_EDGE_THRESHOLD: float = 0.2 # Lowered significantly if 'fixed' is used as fallback


    # --- Adaptive Edge Detection Parameters (New Strategy) ---
    # Error metric to use for adaptive decisions: 'mse' (Mean Squared Error) or 'mae' (Mean Absolute Error)
    ERROR_METRIC_TYPE: str = 'mse'

    # Single error threshold for segment subdivision/termination.
    # Used for both the initial global check and recursive local checks.
    ERROR_THRESHOLD: float = 1e-5 # Combines GLOBAL_ERROR_THRESHOLD and LOCAL_ERROR_THRESHOLD

    MAX_DEPTH: int = 5             # Maximum recursion depth for subdivision
    MIN_SEGMENT_SIZE: int = 16      # Minimum segment dimension (width or height) to stop subdivision


    # --- Old Adaptive Edge Detection Parameters (kept for reference, but not used in new strategy) ---
    # Edge Quality Measure Parameters (M(S)) for adaptive subdivision criterion
    # Measure type: 'gradient_magnitude_near_edges', 'variance_near_edges', 'mean_abs_error_near_edges'
    # NOTE: 'gradient_magnitude_near_edges' requires SciPy.
    EDGE_QUALITY_MEASURE_TYPE: str = 'variance_near_edges' # This will not be used in the new adaptive strategy
    EDGE_QUALITY_BAND_WIDTH: int = 100   # This will not be used in the new adaptive strategy

    # Dynamic Quality Threshold Parameter
    # Quantile (0.0-1.0) of the quality measure values (calculated on the full image)
    # to use as the threshold for M(S).
    QUALITY_THRESHOLD_QUANTILE: float = 0.8 # This will not be used in the new adaptive strategy


    # --- Parallel Processing Settings ---
    NUM_PROCESSES: Optional[int] = None # Number of worker processes for parallel processing.
                                        # None or <= 0 uses default (usually num_cores).


    # --- Visualization and Saving Settings ---
    SAVE_RAW_ERROR_MAPS: bool = False # Save the individual raw error maps for each segment (can be large)
    SAVE_NORMALIZED_ERROR_MAPS: bool = False # Save the individual normalized error maps for each segment
    SAVE_COMPOSITE_ERROR_MAP: bool = False # Save the composite error map for each segment
    SAVE_BINARY_EDGE_MAPS: bool = False # Save the final binary edge map for each segment

    SAVE_FINAL_EDGE_MAP: bool = True # Save the final reassembled binary edge map (for both simple and adaptive)
    SAVE_MAIN_PLOT: bool = True # Save the main visualization plot
    SAVE_ADDITIONAL_PLOTS: bool = True # Save additional plots (e.g., for adaptive stats)

    ERROR_HEATMAP_COLORMAP: str = 'viridis' # Colormap for error heatmaps

    # --- Derived Paths (calculated based on other settings) ---
    @property
    def IMAGE_PATH(self) -> Path:
        """Returns the full path to the input image."""
        return PROJECT_ROOT / "images" / self.IMAGE_FILENAME

    @property
    def IMAGE_BASE_NAME(self) -> str:
        """Returns the base name of the input image (without extension)."""
        return Path(self.IMAGE_FILENAME).stem

    @property
    def IMAGE_EDGE_RESULTS_DIR(self) -> Path:
        """Returns the image-specific results directory for edge detection."""
        # Access the module-level variable directly
        return EDGE_DETECTION_RESULTS_BASE_DIR / self.IMAGE_BASE_NAME


# Create an instance of the configuration to be imported
edge_config = EdgeConfig()
