import os
import shutil
import re
from pathlib import Path
from typing import Dict

def shorten_filename_for_overleaf(original_filename: str) -> str:
    """
    Shortens a complex filename from the adaptive image reconstruction/edge detection
    results into a more concise, Overleaf-friendly format.

    Args:
        original_filename: The full original filename (e.g., "Shepp_Logan_phantom_original_image_deg5_full_mesh_...pdf")

    Returns:
        The shortened filename.
    """
    name_parts = original_filename.split('_')
    image_name = name_parts[0] # Assuming image name is the first part

    # Define mappings for common, verbose parts to shorter equivalents
    plot_type_map = {
        # Reconstruction plots
        "original_image": "orig",
        "reconstructed_image": "reco",
        "actual_error_heatmap": "err",
        "reconstruction_segmentation": "seg",
        "segment_error_heatmap": "segerr",
        "error_distribution": "edist",
        "depth_segment_count": "dcount",
        "segment_size_distribution": "ssdist",
        "segment_data": "data", # For .pkl files

        # Edge detection plots
        "chosen_error_map": "chosen",
        "final_binary_edge_map": "final", # This one is preferred
        # "final_adaptive_binary_edge_map": "final_adapt", # This one is being removed as per discussion
        "edge_strategy_comparison": "comp",
        "reassembled_raw_error_error_original": "rawEo",
        "reassembled_raw_error_error_smoothed": "rawEs",
        "reassembled_raw_error_diff_original_poly_smoothed": "rawDo",
        "reassembled_raw_error_diff_smoothed_poly_original": "rawDs",
    }

    # Extract file extension
    ext = Path(original_filename).suffix
    base_name = original_filename[:len(original_filename) - len(ext)]

    # --- Step 1: Replace verbose plot type descriptions ---
    shortened_base = base_name
    for long_key, short_val in plot_type_map.items():
        # Use regex to find the exact plot type preceded by underscore and followed by underscore
        # This prevents accidental partial matches
        pattern = r'_' + re.escape(long_key) + r'_'
        if re.search(pattern, shortened_base):
            shortened_base = re.sub(pattern, f'_{short_val}_', shortened_base)
            break # Assume only one plot type per filename


    # --- Step 2: Condense reconstruction parameters (deg, measure, errthresh, upscale) ---
    # Example: deg5_full_mesh_measure-rmse_errthresh0p0001_depth15_min5_basis1_procs16_upscale3
    reco_param_match = re.search(
        r'_deg(\d+)_full_mesh_measure-rmse_errthresh([\dpe-]+)_depth\d+_min\d+_basis\d+_procs\d+'
        r'(?:_upscale(\d+))?', # Optional upscale factor
        shortened_base
    )
    if reco_param_match:
        deg = reco_param_match.group(1)
        err_thresh_val = reco_param_match.group(2).replace('p', '') # Remove 'p' for brevity
        upscale_val = reco_param_match.group(3) if reco_param_match.group(3) else ''

        short_reco_params = f"d{deg}e{err_thresh_val}"
        if upscale_val:
            short_reco_params += f"u{upscale_val}"
        
        # Replace the long parameter string with the shortened one
        shortened_base = re.sub(reco_param_match.re, f'_{short_reco_params}', shortened_base)


    # --- Step 3: Condense edge detection parameters (strategy, threshold) ---
    # Example: _edge_strat-weighted_sum_edge_thresh-fixed-0p08
    edge_param_match = re.search(
        r'_edge_strat-([a-zA-Z_]+)_edge_thresh-fixed-([\dpe-]+)',
        shortened_base
    )
    if edge_param_match:
        strategy = edge_param_match.group(1)
        threshold_val = edge_param_match.group(2).replace('p', '') # Remove 'p' for brevity

        # Map strategy names to shorter versions
        strategy_short_map = {
            'max': 'Max',
            'weighted_sum': 'WS',
            'error_original': 'Eo',
            'error_smoothed': 'Es',
            'diff_original_poly_smoothed': 'Do',
            'diff_smoothed_poly_original': 'Ds',
        }
        short_strategy = strategy_short_map.get(strategy, strategy) # Use original if not in map

        short_edge_params = f"s{short_strategy}t{threshold_val}"
        shortened_base = re.sub(edge_param_match.re, f'_{short_edge_params}', shortened_base)
    
    # Final cleanup: remove any remaining double underscores or trailing underscores
    shortened_base = shortened_base.replace('__', '_')
    if shortened_base.endswith('_'):
        shortened_base = shortened_base[:-1]

    # Reassemble with image name (which was kept in place) and extension
    final_short_name = f"{shortened_base}{ext}"

    # Ensure image name is at the start (to handle cases where plot type is first in the original string)
    if not final_short_name.startswith(image_name):
        # This could happen if image_name was not the first component parsed correctly
        # Fallback to a simpler naming for safety if complex parsing failed, or prepend.
        # For simplicity, let's assume original filenames are structured like "ImageName_PlotType_Params.ext"
        # If it's still long, try a very basic shortening:
        if len(final_short_name) > 60: # Arbitrary length for "too long"
            # Fallback will use the image name and the last detected short plot type
            last_short_val = "plot" # Default if no plot type was matched
            for long_key, short_val in plot_type_map.items():
                if long_key in base_name: # Check in the full base_name to get the most relevant
                    last_short_val = short_val
            return f"{image_name}_{last_short_val}{ext}"
        
    return final_short_name.lower() # Convert to lowercase for good measure


def export_results_for_overleaf(
    project_root: Path,
    overleaf_output_base_dir_name: str = "overleaf_exports",
    reconstruction_results_source_dir_name: str = "image_reconstruction_results",
    edge_detection_results_source_dir_name: str = "edge_detection_results"
):
    """
    Creates Overleaf-compatible copies of results folders with shortened filenames.

    Args:
        project_root: The root directory of your project.
        overleaf_output_base_dir_name: The name of the directory where shortened
                                       files will be saved.
        reconstruction_results_source_dir_name: Name of the source directory for reconstruction results.
        edge_detection_results_source_dir_name: Name of the source directory for edge detection results.
    """
    overleaf_output_dir = project_root / overleaf_output_base_dir_name
    
    reco_source_dir = project_root / "results" / reconstruction_results_source_dir_name
    edge_source_dir = project_root / "results" / edge_detection_results_source_dir_name

    source_dirs = {
        reco_source_dir: overleaf_output_dir / reconstruction_results_source_dir_name,
        edge_source_dir: overleaf_output_dir / edge_detection_results_source_dir_name,
    }

    print(f"Exporting shortened files to: {overleaf_output_dir}")

    for source_base, dest_base in source_dirs.items():
        if not source_base.exists():
            print(f"Source directory not found: {source_base}. Skipping.")
            continue

        print(f"\nProcessing files in: {source_base}")
        # Iterate through image-specific subdirectories
        for image_subdir in source_base.iterdir():
            if image_subdir.is_dir():
                print(f"  Processing image directory: {image_subdir.name}")
                dest_image_subdir = dest_base / image_subdir.name
                os.makedirs(dest_image_subdir, exist_ok=True)

                for file_path in image_subdir.iterdir():
                    if file_path.is_file():
                        original_filename = file_path.name
                        shortened_filename = shorten_filename_for_overleaf(original_filename)
                        
                        dest_file_path = dest_image_subdir / shortened_filename
                        
                        try:
                            shutil.copy2(file_path, dest_file_path)
                            print(f"    Copied and shortened: {original_filename} -> {shortened_filename}")
                        except Exception as e:
                            print(f"    Error copying {original_filename}: {e}")

    print("\nExport process completed.")

if __name__ == "__main__":
    # If this script is in `your_project_root/scripts/`, then the project root
    # is two levels up from the script's current location.
    PROJECT_ROOT_FOR_EXPORT = Path(__file__).resolve().parent.parent

    # Create the scripts directory if it doesn't exist
    scripts_dir = PROJECT_ROOT_FOR_EXPORT / "scripts"
    os.makedirs(scripts_dir, exist_ok=True)

    export_results_for_overleaf(PROJECT_ROOT_FOR_EXPORT)

