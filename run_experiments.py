# run_experiments.py

import sys
import os
from pathlib import Path
from typing import Dict, List, Tuple
import multiprocessing
from PIL import Image # For getting image shape without full load

# Ensure project root is in path
# PROJECT_ROOT is where this script resides.
PROJECT_ROOT = Path(__file__).resolve().parent
# Add the 'utils' directory to the Python path to import modules from it
# This line is crucial for finding 'config' modules and for the 'utils' package.
sys.path.insert(0, str(PROJECT_ROOT))


# Import your modules explicitly from their packages
try:
    import config.config_reconstructor as config_reconstructor
    import config.config_detector as config_detector
    from utils.image_reconstructor import AdaptivePolynomialReconstructor # Corrected import path
    from utils.edge_detector import EdgeDetector # Corrected import path
except ImportError as e:
    print(f"Error importing necessary modules: {e}")
    print("Please ensure your project structure is correct and all config and core files are in place.")
    print("Expected structure: project_root/config/, project_root/utils/, project_root/poly_approx/ etc.")
    sys.exit(1)


def get_image_shape(image_path: Path) -> Tuple[int, int]:
    """Helper to get image shape without loading full image data."""
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found at {image_path}")
    with Image.open(image_path) as img:
        return img.height, img.width # PIL returns (width, height), we need (height, width)


def generate_reconstruction_params_suffix(reco_config) -> str:
    """Generates a consistent parameter suffix for reconstruction runs."""
    num_processes = reco_config.NUM_PROCESSES
    if num_processes is None or num_processes <= 0:
        num_processes = 1
    
    return (
        f"deg{reco_config.POLY_DEGREE}_{reco_config.NODES_METHOD}"
        f"_measure-{reco_config.ERROR_MEASURE_TYPE}_errthresh{str(reco_config.ERROR_THRESHOLD).replace('.', 'p')}_depth{reco_config.MAX_DEPTH}_min{reco_config.MIN_SEGMENT_SIZE}"
        f"_basis{reco_config.POLY_BASIS_USED}"
        f"_procs{num_processes}"
    )


def run_all_experiments():
    """
    Orchestrates the image reconstruction and edge detection experiments
    for multiple images and configurations.
    """
    print("Starting automated image processing experiments...")

    # Define images to process and their specific fixed thresholds
    # You can add more images or modify thresholds here.
    images_to_process = [
        {"name": "Shepp_Logan_phantom", "fixed_threshold": 0.04},
        {"name": "spiral", "fixed_threshold": 0.05},
        {"name": "spiral_and_zigzag", "fixed_threshold": 0.15},
    ]

    # Define edge detection strategies to test (besides the main one in config_detector)
    # This is for the comparison plot, make sure these are in config_detector.EDGE_STRATEGIES_TO_COMPARE
    # No need to iterate explicitly here, as the EdgeDetector itself generates the comparison plot
    # based on its config.
    
    for img_info in images_to_process:
        image_name = img_info["name"]
        fixed_threshold = img_info["fixed_threshold"]
        image_filename = f"{image_name}.png"
        image_path = config_reconstructor.IMAGE_DIR / image_filename

        print(f"\n--- Processing Image: {image_name} ---")

        # --- RECONSTRUCTION PHASE ---
        # Set config_reconstructor's current image
        config_reconstructor.IMAGE_NAME = image_name
        config_reconstructor.IMAGE_FILENAME = image_filename
        config_reconstructor.IMAGE_PATH = image_path

        reco_params_suffix = generate_reconstruction_params_suffix(config_reconstructor)
        
        # Use the RECONSTRUCTION_RESULTS_BASE_DIR from config_reconstructor
        reco_results_dir_for_data = config_reconstructor.RECONSTRUCTION_RESULTS_BASE_DIR / image_name
        reco_data_filepath = reco_results_dir_for_data / f"{image_name}_segment_data_{reco_params_suffix}.pkl"

        original_image_shape = None
        upscale_factor = config_reconstructor.UPSCALE_FACTOR # Get this from config for the current run

        if not reco_data_filepath.exists():
            print(f"Reconstruction data NOT found for {image_name}. Running image reconstruction...")
            reconstructor = AdaptivePolynomialReconstructor(config_reconstructor)
            reco_suffix_returned, img_shape_returned, upscale_returned = reconstructor.run_reconstruction()
            
            if not reco_suffix_returned: # Check if reconstruction failed
                print(f"Skipping edge detection for {image_name} as reconstruction failed.")
                continue
            
            original_image_shape = img_shape_returned
            upscale_factor = upscale_returned
            # Verify the returned suffix matches the expected one
            if reco_suffix_returned != reco_params_suffix:
                print(f"Warning: Returned suffix '{reco_suffix_returned}' does not match expected '{reco_params_suffix}'.")
                # Use the returned suffix for edge detection to ensure data can be found
                reco_params_suffix = reco_suffix_returned
        else:
            print(f"Reconstruction data found for {image_name}. Skipping reconstruction.")
            try:
                original_image_shape = get_image_shape(image_path)
            except FileNotFoundError as e:
                print(f"Error: {e}. Cannot determine original image shape. Skipping {image_name}.")
                continue
            # Upscale factor remains as per config_reconstructor for this run

        # Ensure original_image_shape is valid before proceeding
        if original_image_shape is None or original_image_shape[0] == 0 or original_image_shape[1] == 0:
            print(f"Error: Could not determine valid original image shape for {image_name}. Skipping edge detection.")
            continue

        # --- EDGE DETECTION PHASE ---
        print(f"\n--- Starting Edge Detection for {image_name} ---")
        
        # Instantiate EdgeDetector (it uses config_detector for its defaults)
        detector = EdgeDetector(config_detector, image_name, image_path)
        
        # Run edge detection with the main strategy and the specific fixed threshold
        print(f"Running edge detection with strategy '{config_detector.ERROR_COMBINATION_STRATEGY}' and fixed threshold {fixed_threshold}...")
        detector.run_detection(
            params_suffix=reco_params_suffix,
            original_image_shape=original_image_shape,
            upscale_factor=upscale_factor,
            fixed_threshold_override=fixed_threshold
        )
        print(f"Edge detection for {image_name} completed.")

    print("\nAll experiments finished.")


if __name__ == "__main__":
    # Ensure multiprocessing works correctly on all OS (especially Windows)
    multiprocessing.freeze_support() 
    run_all_experiments()

