# config.py

import os
from pathlib import Path
import multiprocessing
from typing import Dict, List, Optional # Import necessary types for Dict, List, Optional

# --- Project Paths ---
# Assuming this config file is in a 'config' subfolder within the project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent

IMAGE_DIR = PROJECT_ROOT / "images"
BASE_RESULTS_DIR = PROJECT_ROOT / "results"
ADAPTIVE_RECONSTRUCTION_RESULTS_BASE_DIR = BASE_RESULTS_DIR / "adaptive_image_reconstruction_tests"

# Ensure necessary directories exist
os.makedirs(IMAGE_DIR, exist_ok=True)
os.makedirs(BASE_RESULTS_DIR, exist_ok=True)
os.makedirs(ADAPTIVE_RECONSTRUCTION_RESULTS_BASE_DIR, exist_ok=True)


# --- Image Configuration ---
IMAGE_NAME = "spiral_and_zigzag" # Base name of the image (e.g., "spiral.png")
IMAGE_FILENAME = f"{IMAGE_NAME}.png"
IMAGE_PATH = IMAGE_DIR / IMAGE_FILENAME


# --- Polynomial Approximation Parameters ---
# These parameters are applied per segment
POLY_DEGREE = 10 # Degree of the polynomial approximation
NODES_METHOD = 'leja' # Method for selecting interpolation/approximation nodes: 'full_mesh', 'leja', 'fekete', 'padua'
ADMISSIBLE_MESH_TYPE = 'cheb' # Type of admissible mesh to generate: 'cheb' (only supported now)
M_CHEB = 2 # Parameter 'm' for Chebyshev mesh construction (m > 1)
POLY_BASIS_USED = 1 # Polynomial basis used for approximation (1: Shifted Monomials, 2: Monomial, 3: Chebyshev)


# --- Adaptive Segmentation Control Parameters ---
ERROR_MEASURE_TYPE = 'rmse' # Error measure for adaptive criterion: 'mse', 'mae', 'rmse'
ERROR_THRESHOLD = 0.0001 # Threshold for the error measure M(S)
MAX_DEPTH = 15 # Maximum recursive segmentation depth (0 is the whole image)
MIN_SEGMENT_SIZE = 5 # Minimum dimension (width or height) of a segment to stop subdivision


# --- Parallel Processing Parameters ---
# Set to None or 1 to disable parallel processing
# Set to an integer > 1 for a fixed number of processes
# Set to 0 to use the number of CPU cores
NUM_PROCESSES = multiprocessing.cpu_count() # Use all available CPU cores


# --- Higher Resolution Output ---
# Factor by which to upscale the reconstructed image (e.g., 2 for 2x resolution)
# Set to 1 for original resolution
UPSCALE_FACTOR = 3


# --- Results Saving and Visualization ---
SAVE_SEGMENT_DATA = True # Whether to save the segment data and coefficients (.pkl file)
SAVE_RECONSTRUCTED_IMAGE = True # Whether to save the reassembled approximation image
SAVE_ERROR_MAP_VIZ = True # Whether to save the normalized actual reconstruction error map visualization
SAVE_MAIN_PLOT = True # Whether to save the main plot (Original, Reconstructed, Error, Segmentation)
SAVE_SEGMENT_ERROR_HEATMAP_PLOT = True # Whether to save the segment error heatmap plot

# New visualization flags
SAVE_ERROR_DISTRIBUTION_PLOT = True # Plot histogram of final segment errors
SAVE_DEPTH_SEGMENT_COUNT_PLOT = True # Plot number of segments per depth level
SAVE_SEGMENT_SIZE_DISTRIBUTION_PLOT = True # Plot histogram of final segment sizes


# Colormaps for visualization
ERROR_HEATMAP_COLORMAP = 'viridis' # Colormap for the actual reconstruction error heatmap
SEGMENT_ERROR_COLORMAP = 'hot' # Colormap for the segment error heatmap


# --- Edge Detection Parameters (Moved from edge_config.py) ---
# Strategy for combining the different raw error maps ('error_original', 'error_smoothed', etc.)
# Options: 'max', 'weighted_sum', 'logical_and', 'logical_or',
#          or any of the raw error map keys directly (e.g., 'error_original', 'diff_original_poly_smoothed')
ERROR_COMBINATION_STRATEGY: str = 'weighted_sum'

# Weights for 'weighted_sum' strategy (keys must match raw error map keys)
# Only used if ERROR_COMBINATION_STRATEGY is 'weighted_sum'
ERROR_COMBINATION_WEIGHTS: Dict[str, float] = {
    'error_original': 0.1,
    'error_smoothed': 0.1,
    'diff_original_poly_smoothed': 0.6,
    'diff_smoothed_poly_original': 0.2,
}
# Note: Weights will be normalized to sum to 1.0 if they don't.

# Which raw error maps to use for 'logical_and' or 'logical_or' strategies
# Provide a list of keys from ['error_original', 'error_smoothed', 'diff_original_poly_smoothed', 'diff_smoothed_poly_original']
# These raw maps will be normalized (over the segment for adaptive, over full image for simple before thresholding)
# before thresholding and logical combination.
LOGICAL_OP_ERROR_KEYS: List[str] = ['error_original', 'diff_original_poly_smoothed', 'diff_smoothed_poly_original']

# Type of thresholding to apply to the combined error map (or individual maps for logical ops)
# 'fixed': Use a fixed threshold (0.0 to 1.0)
# 'otsu': Use Otsu's method (requires scikit-image)
EDGE_THRESHOLD_TYPE: str = 'fixed'
FIXED_EDGE_THRESHOLD: float = 0.1 # Only used if EDGE_THRESHOLD_TYPE is 'fixed'

# Saving flags for edge detection results
SAVE_RAW_ERROR_MAPS: bool = False # Save the individual reassembled raw error maps (can be large)
SAVE_COMPOSITE_ERROR_MAP: bool = False # Save the reassembled composite error map
SAVE_BINARY_EDGE_MAPS: bool = True # Save the final reassembled binary edge map

# Colormap for edge error heatmaps (distinguished from general error heatmap)
EDGE_ERROR_HEATMAP_COLORMAP: str = 'plasma'


# --- Edge Strategy Comparison Plot ---
SAVE_EDGE_STRATEGY_COMPARISON_PLOT: bool = True # Whether to generate and save a comparison plot of different edge strategies

# List of strategies to compare in the comparison plot.
# Each item should be a valid ERROR_COMBINATION_STRATEGY.
# Ensure 'logical_and'/'logical_or' strategies use LOGICAL_OP_ERROR_KEYS if needed.
EDGE_STRATEGIES_TO_COMPARE: List[str] = [
    'error_original',
    'error_smoothed',
    'max',
    'logical_and',
    'logical_or',
    'weighted_sum',
    'diff_original_poly_smoothed',
    'diff_smoothed_poly_original',
    # Add more strategies here if desired, e.g., 'diff_original_poly_smoothed'
]

# config/config.py
# ... other configurations ...

# Parameters for dynamic error threshold
INITIAL_ERROR_THRESHOLD = 0.1  # Starting error threshold at depth 0
ERROR_DECAY_RATE = 0.20        # Percentage decrease per depth level (e.g., 0.10 for 10% decrease)

# ... rest of your config file ...


# --- Other Parameters ---
# Add any other global parameters here
