# computational_performance_analyzer.py

import sys
import os
import time
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Union, Optional
import multiprocessing
import json # For saving structured data

# Determine the project root dynamically
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import necessary modules for full reconstruction
try:
    import config.config_reconstructor as config_reconstructor # Import general config for parameters
    from utils.image_reconstructor import AdaptivePolynomialReconstructor # Import the reconstructor
    
    _poly_approx_available = True
except ImportError as e:
    print(f"Could not import necessary modules for performance analysis: {e}")
    print("Please ensure your project structure is correct and all config and core files are in place.")
    print("Computational performance analysis will be skipped.")
    _poly_approx_available = False


class ComputationalPerformanceAnalyzer:
    """
    Analyzes the computational performance and approximation quality of different
    polynomial approximation methods by running full adaptive image reconstructions
    for a set of benchmark images.
    """

    def __init__(self, config_obj, benchmark_image_names: List[str]):
        """
        Initializes the analyzer.

        Args:
            config_obj: The config_reconstructor module.
            benchmark_image_names: A list of names of the images to use for benchmarking
                                   the full reconstruction process.
        """
        self.config_template = config_obj # Store the original config_reconstructor
        self.benchmark_image_names = benchmark_image_names
        
        # Results directory for performance data
        self.performance_results_dir = self.config_template.BASE_RESULTS_DIR / "computational_performance_results"
        os.makedirs(self.performance_results_dir, exist_ok=True)


    def measure_method_performance_for_image(self, method: str, image_name: str) -> Dict[str, Union[float, int]]:
        """
        Measures the average computational time and approximation quality metrics
        across all terminal segments during a full adaptive image reconstruction
        for a *single image* using the specified node selection method.

        Args:
            method: The node selection method to test (e.g., 'full_mesh', 'leja').
            image_name: The name of the image to benchmark.

        Returns:
            A dictionary containing 'total_reconstruction_time_s', 'avg_segment_time_ms',
            'num_nodes', 'avg_max_absolute_error', and 'avg_std_dev_error'.
        """
        if not _poly_approx_available:
            print(f"Skipping performance measurement for '{method}' on '{image_name}' due to missing modules.")
            return {
                'total_reconstruction_time_s': -1.0,
                'avg_segment_time_ms': -1.0,
                'num_nodes': -1, 
                'avg_max_absolute_error': -1.0,
                'avg_std_dev_error': -1.0,
                'num_terminal_segments': -1
            }

        print(f"\n--- Running full reconstruction for method: '{method}' on image: '{image_name}' ---")

        # Temporarily modify the config_reconstructor for this run
        current_image_path = self.config_template.IMAGE_DIR / f"{image_name}.png"
        
        # Store original values to restore them later (important as config is shared)
        original_nodes_method = self.config_template.NODES_METHOD
        original_image_name_cfg = self.config_template.IMAGE_NAME
        original_image_filename_cfg = self.config_template.IMAGE_FILENAME
        original_image_path_cfg = self.config_template.IMAGE_PATH

        self.config_template.NODES_METHOD = method
        self.config_template.IMAGE_NAME = image_name
        self.config_template.IMAGE_FILENAME = f"{image_name}.png"
        self.config_template.IMAGE_PATH = current_image_path

        # Instantiate the reconstructor with the (temporarily) updated config
        reconstructor = AdaptivePolynomialReconstructor(self.config_template)

        start_reconstruction_time = time.time()
        # Run the full reconstruction
        # The run_reconstruction populates reconstructor.final_segment_data
        params_suffix, _, _ = reconstructor.run_reconstruction()
        total_reconstruction_time_s = time.time() - start_reconstruction_time

        # Restore original config values
        self.config_template.NODES_METHOD = original_nodes_method
        self.config_template.IMAGE_NAME = original_image_name_cfg
        self.config_template.IMAGE_FILENAME = original_image_filename_cfg
        self.config_template.IMAGE_PATH = original_image_path_cfg


        # --- Aggregate Metrics from all terminal segments ---
        segment_times = []
        segment_max_abs_errors = []
        segment_std_dev_errors = []
        num_terminal_segments = 0

        for segment_id, data in reconstructor.final_segment_data.items():
            if data.get('status') in ['terminated_by_error', 'terminated_by_depth', 'terminated_by_size']:
                segment_times.append(data.get('computation_time', 0.0))
                segment_max_abs_errors.append(data.get('max_absolute_error', 0.0))
                segment_std_dev_errors.append(data.get('std_dev_error', 0.0))
                num_terminal_segments += 1
        
        avg_segment_time_ms = (np.mean(segment_times) * 1000) if segment_times else 0.0
        avg_max_absolute_error = np.mean(segment_max_abs_errors) if segment_max_abs_errors else 0.0
        avg_std_dev_error = np.mean(segment_std_dev_errors) if segment_std_dev_errors else 0.0

        # Node count is specific to the method and polynomial degree, not averaged over segments.
        # It's always (deg+1)*(deg+2)/2 for Leja/Fekete/Padua and (m*deg+1)^2 for Full Mesh.
        poly_degree = self.config_template.POLY_DEGREE
        if method == 'full_mesh':
            num_points_per_dim = self.config_template.M_CHEB * poly_degree + 1
            num_nodes = num_points_per_dim * num_points_per_dim
        else: # Leja, Fekete, Padua
            num_nodes = int((poly_degree + 1) * (poly_degree + 2) / 2)

        print(f"  Result for '{image_name}' with '{method}': Total Time: {total_reconstruction_time_s:.4f}s, Avg Seg Time: {avg_segment_time_ms:.4f}ms, Num Nodes: {num_nodes}, Avg Max Abs Error: {avg_max_absolute_error:.6f}, Avg Std Dev Error: {avg_std_dev_error:.6f}, Segments: {num_terminal_segments}")

        return {
            'total_reconstruction_time_s': total_reconstruction_time_s,
            'avg_segment_time_ms': avg_segment_time_ms,
            'num_nodes': num_nodes, # This represents the *per-segment* node count for approximation
            'avg_max_absolute_error': avg_max_absolute_error,
            'avg_std_dev_error': avg_std_dev_error,
            'num_terminal_segments': num_terminal_segments
        }

    def run_analysis(self):
        """
        Runs the performance analysis for all specified node selection methods
        by performing full image reconstructions across all benchmark images.
        """
        print(f"\nStarting comprehensive computational performance analysis for full image reconstruction across {len(self.benchmark_image_names)} images...")
        
        methods_to_test = [
            'full_mesh', 
            'leja',      
            'fekete',    
            'padua'      
        ]

        # Structure to store results: {method: {image_name: {metrics}}}
        all_results: Dict[str, Dict[str, Dict[str, Union[float, int]]]] = {method: {} for method in methods_to_test}

        for image_name in self.benchmark_image_names:
            for method in methods_to_test:
                image_method_results = self.measure_method_performance_for_image(method, image_name)
                all_results[method][image_name] = image_method_results

        # Aggregate results for overall summary (e.g., average across images)
        aggregated_results: Dict[str, Dict[str, Union[float, int]]] = {}
        for method in methods_to_test:
            method_results_across_images = [
                res for res in all_results[method].values()
                if res['total_reconstruction_time_s'] != -1.0 # Exclude failed runs
            ]
            
            if method_results_across_images:
                avg_total_time = np.mean([res['total_reconstruction_time_s'] for res in method_results_across_images])
                avg_avg_segment_time = np.mean([res['avg_segment_time_ms'] for res in method_results_across_images])
                # Num nodes is constant per method and degree, so just take from the first successful run
                num_nodes = method_results_across_images[0]['num_nodes'] 
                avg_avg_max_abs_error = np.mean([res['avg_max_absolute_error'] for res in method_results_across_images])
                avg_avg_std_dev_error = np.mean([res['avg_std_dev_error'] for res in method_results_across_images])
                avg_num_terminal_segments = np.mean([res['num_terminal_segments'] for res in method_results_across_images])
            else:
                avg_total_time = -1.0
                avg_avg_segment_time = -1.0
                num_nodes = -1
                avg_avg_max_abs_error = -1.0
                avg_avg_std_dev_error = -1.0
                avg_num_terminal_segments = -1

            aggregated_results[method] = {
                'avg_total_reconstruction_time_s': avg_total_time,
                'avg_avg_segment_time_ms': avg_avg_segment_time,
                'num_nodes': num_nodes, # This is the per-segment node count
                'avg_avg_max_absolute_error': avg_avg_max_abs_error,
                'avg_avg_std_dev_error': avg_avg_std_dev_error,
                'avg_num_terminal_segments': avg_num_terminal_segments
            }

        # Save all detailed results (per image, per method) to a JSON file
        detailed_results_filename = f"poly_approx_full_reco_performance_deg{self.config_template.POLY_DEGREE}_detailed.json"
        detailed_results_filepath = self.performance_results_dir / detailed_results_filename
        try:
            with open(detailed_results_filepath, 'w') as f:
                json.dump(all_results, f, indent=4)
            print(f"\nSaved detailed full reconstruction performance results to {detailed_results_filepath}")
        except Exception as e:
            print(f"Error saving detailed performance results to JSON: {e}")

        # Save aggregated results (averaged across images) to a separate JSON file
        aggregated_results_filename = f"poly_approx_full_reco_performance_deg{self.config_template.POLY_DEGREE}_aggregated.json"
        aggregated_results_filepath = self.performance_results_dir / aggregated_results_filename
        try:
            with open(aggregated_results_filepath, 'w') as f:
                json.dump(aggregated_results, f, indent=4)
            print(f"Saved aggregated full reconstruction performance results to {aggregated_results_filepath}")
        except Exception as e:
            print(f"Error saving aggregated performance results to JSON: {e}")


        print("\nComputational performance analysis finished.")
        print("\nSummary of Full Reconstruction Results (Averaged Across Benchmarked Images):")
        print(f"{'Method':<20} | {'Avg Total Time (s)':<19} | {'Avg Seg Time (ms)':<19} | {'Num Nodes':<10} | {'Avg Max Abs Error':<19} | {'Avg Std Dev Error':<19} | {'Avg Segments':<12}")
        print("-" * 140) # Adjusted separator length
        for method, data in aggregated_results.items():
            print(f"{method:<20} | {data['avg_total_reconstruction_time_s']:<19.4f} | {data['avg_avg_segment_time_ms']:<19.4f} | {data['num_nodes']:<10} | {data['avg_avg_max_absolute_error']:<19.6f} | {data['avg_avg_std_dev_error']:<19.6f} | {data['avg_num_terminal_segments']:<12.1f}")

if __name__ == "__main__":
    if not _poly_approx_available:
        print("Required modules not available. Exiting performance analyzer.")
        sys.exit(1)

    # Define the list of images to use for benchmarking
    # Ensure these images exist in your config_reconstructor.IMAGE_DIR
    benchmark_images = [
        "Shepp_Logan_phantom",
        "spiral",
        "spiral_and_zigzag"
    ]

    analyzer = ComputationalPerformanceAnalyzer(config_reconstructor, benchmark_image_names=benchmark_images) 
    analyzer.run_analysis()
