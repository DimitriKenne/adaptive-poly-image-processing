# config.py

import os
from pathlib import Path
import multiprocessing

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
POLY_DEGREE = 5 # Degree of the polynomial approximation
NODES_METHOD = 'leja' # Method for selecting interpolation/approximation nodes: 'full_mesh', 'leja', 'fekete', 'padua'
ADMISSIBLE_MESH_TYPE = 'cheb' # Type of admissible mesh to generate: 'cheb' (only supported now)
M_CHEB = 2 # Parameter 'm' for Chebyshev mesh construction (m > 1)
# SIGMA = 1.0 # Standard deviation for Gaussian smoothing (can be kept here or in the main class)
POLY_BASIS_USED = 1 # Polynomial basis used for approximation (1: Shifted Monomials, 2: Monomial, 3: Chebyshev)


# --- Adaptive Segmentation Control Parameters ---
ERROR_MEASURE_TYPE = 'mse' # Error measure for adaptive criterion: 'mse', 'mae', 'rmse'
ERROR_THRESHOLD = 0.001 # Threshold for the error measure M(S)
MAX_DEPTH = 5 # Maximum recursive segmentation depth (0 is the whole image)
MIN_SEGMENT_SIZE = 8 # Minimum dimension (width or height) of a segment to stop subdivision


# --- Parallel Processing Parameters ---
# Set to None or 1 to disable parallel processing
# Set to an integer > 1 for a fixed number of processes
# Set to 0 to use the number of CPU cores
NUM_PROCESSES = multiprocessing.cpu_count() # Use all available CPU cores


# --- Higher Resolution Output ---
# Factor by which to upscale the reconstructed image (e.g., 2 for 2x resolution)
# Set to 1 for original resolution
UPSCALE_FACTOR = 2


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


# --- Other Parameters ---
# Add any other global parameters here

