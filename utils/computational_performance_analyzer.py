# computational_performance_analyzer.py

import sys
import os
import time
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Union, Optional
import multiprocessing
import json # For saving structured data
from PIL import Image # For loading real image

# Determine the project root dynamically
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import necessary functions from poly_approx package
try:
    # We need image_poly_approximation_segment for benchmarking
    from poly_approx.image_poly_approximation import image_poly_approximation_segment
    
    # Correct import: use the unified extremal_points function
    from poly_approx.interpolation_nodes import extremal_points
    
    # Import config_reconstructor for default poly degree, basis, etc.
    import config.config_reconstructor as config_reconstructor 

    _poly_approx_available = True
except ImportError as e:
    print(f"Could not import modules from poly_approx or config: {e}")
    print("Please ensure your project structure is correct and all required modules are present.")
    print("Computational performance analysis will be skipped.")
    _poly_approx_available = False

    # Define dummy functions for robustness if imports fail
    def image_poly_approximation_segment(*args, **kwargs):
        print("Dummy image_poly_approximation_segment called for performance analysis.")
        time.sleep(0.01) # Simulate 10ms
        dummy_poly_degree = kwargs.get('poly_degree', 5)
        num_coeffs = int((dummy_poly_degree + 1) * (dummy_poly_degree + 2) / 2)
        dummy_coeffs = np.zeros(num_coeffs)
        # Dummy error map, ensure it has some variance for std dev calculation
        dummy_error_map = np.random.rand(100, 100) * 0.1 # Small random errors
        return {
            'computation_time': 0.01,
            'coefficients_original': dummy_coeffs,
            'error_original': dummy_error_map # Return an error map for metric calculation
        }
    
    # Dummy extremal_points function
    def extremal_points(*args, **kwargs): return np.zeros((1, 2))


class ComputationalPerformanceAnalyzer:
    """
    Analyzes the computational performance of different polynomial approximation methods
    by measuring the time taken for a fixed-size segment from a real image.
    """

    def __init__(self, config_obj, image_name: str = "spiral_and_zigzag"):
        self.config = config_obj
        self.image_name = image_name
        self.image_path = self.config.IMAGE_DIR / f"{self.image_name}.png"
        
        # Define a fixed size segment for benchmarking from the real image
        # We will load the full image and take a crop to ensure consistency
        self.segment_size = (100, 100) # Example: 100x100 pixels
        self.dummy_image_segment = self._load_and_crop_image_segment(self.image_path, self.segment_size)
        
        # A rectangle representing the scaled domain for the dummy segment
        self.dummy_rectangle = (0.0, 0.0, 1.0, 1.0) # Assume it maps to [0,1]x[0,1] for simplicity

        # Results directory for performance data
        self.performance_results_dir = self.config.BASE_RESULTS_DIR / "computational_performance_results"
        os.makedirs(self.performance_results_dir, exist_ok=True)

    def _load_and_crop_image_segment(self, image_path: Path, crop_size: Tuple[int, int]) -> np.ndarray:
        """Loads a grayscale image, normalizes it, and crops a segment for analysis."""
        if not image_path.exists():
            print(f"Error: Image not found at {image_path}. Cannot perform analysis. Using dummy random segment.")
            return np.random.rand(*crop_size).astype(np.float32) # Fallback to random
        
        try:
            img = np.array(Image.open(image_path).convert('L'), dtype=np.float32) / 255.0
            
            # Take a crop from the center or top-left, ensuring it's within bounds
            height, width = img.shape
            crop_h, crop_w = crop_size
            
            if height < crop_h or width < crop_w:
                print(f"Warning: Image {image_path.name} is too small ({height}x{width}) for desired crop ({crop_h}x{crop_w}). Using full image scaled or padding.")
                # Simple scaling down if image is smaller than target crop
                # Or for simplicity, if smaller, just return what's available (might not be exactly crop_size)
                # For consistency, it's better to ensure a fixed size.
                # Let's take top-left corner or scale if needed
                if height < crop_h: crop_h = height
                if width < crop_w: crop_w = width
                return img[:crop_h, :crop_w]
            
            # Take a crop from the top-left corner (0,0) for simplicity
            cropped_segment = img[0:crop_h, 0:crop_w]
            print(f"Loaded and cropped {crop_h}x{crop_w} segment from {image_path.name} for performance analysis.")
            return cropped_segment

        except Exception as e:
            print(f"Error loading or cropping image {image_path}: {e}. Using dummy random segment.")
            return np.random.rand(*crop_size).astype(np.float32) # Fallback to random


    def measure_method_performance(self, method: str, num_runs: int = 100) -> Dict[str, Union[float, int]]:
        """
        Measures the average computational time, number of nodes, max absolute error,
        and standard deviation of error for a given method.

        Args:
            method: The node selection method to test.
            num_runs: Number of times to run the approximation for averaging.

        Returns:
            A dictionary containing 'average_time_ms', 'num_nodes', 'max_absolute_error',
            and 'std_dev_error'.
        """
        if not _poly_approx_available:
            print(f"Skipping performance measurement for '{method}' due to missing modules.")
            return {'average_time_ms': -1.0, 'num_nodes': -1, 'max_absolute_error': -1.0, 'std_dev_error': -1.0}

        total_time = 0.0
        total_max_abs_error = 0.0
        total_std_dev_error = 0.0
        num_nodes = 0 

        print(f"Benchmarking '{method}' for {num_runs} runs...")

        # Determine the number of nodes using the extremal_points function once
        try:
            nodes = extremal_points(
                deg=self.config.POLY_DEGREE,
                method=method,
                rectangle=self.dummy_rectangle,
                admissible_mesh_type=self.config.ADMISSIBLE_MESH_TYPE,
                m_cheb=self.config.M_CHEB,
                poly_basis=self.config.POLY_BASIS_USED
            )
            num_nodes = len(nodes)
        except Exception as e:
            print(f"Error determining num_nodes for '{method}' using extremal_points: {e}. Using default count.")
            num_nodes = int((self.config.POLY_DEGREE + 1) * (self.config.POLY_DEGREE + 2) / 2)


        for i in range(num_runs):
            try:
                start_time = time.perf_counter()
                results = image_poly_approximation_segment(
                    image_segment=self.dummy_image_segment,
                    rectangle=self.dummy_rectangle,
                    poly_degree=self.config.POLY_DEGREE,
                    nodes_method=method, 
                    admissible_mesh_type=self.config.ADMISSIBLE_MESH_TYPE,
                    m_cheb=self.config.M_CHEB,
                    poly_basis=self.config.POLY_BASIS_USED
                )
                end_time = time.perf_counter()
                total_time += (end_time - start_time) * 1000 # Convert to milliseconds
                
                # Calculate error metrics
                if 'error_original' in results and results['error_original'] is not None and results['error_original'].size > 0:
                    total_max_abs_error += np.max(np.abs(results['error_original']))
                    total_std_dev_error += np.std(results['error_original'])
                else:
                    print(f"Warning: No 'error_original' in results for '{method}' run {i}. Using 0 for error metrics.")

            except Exception as e:
                print(f"Error during benchmark for method '{method}' run {i}: {e}. Skipping this run.")
                # Return current averages for completed runs, or -1 if no runs completed
                if i == 0: # If first run failed
                    return {'average_time_ms': -1.0, 'num_nodes': num_nodes, 'max_absolute_error': -1.0, 'std_dev_error': -1.0}
                break # Break loop if an error occurs

        runs_completed = num_runs if (num_runs - i) == 0 else i # Number of actual successful runs
        
        avg_time_ms = total_time / runs_completed
        avg_max_abs_error = total_max_abs_error / runs_completed
        avg_std_dev_error = total_std_dev_error / runs_completed

        print(f"'{method}' - Avg Time: {avg_time_ms:.4f} ms, Num Nodes: {num_nodes}, Max Abs Error: {avg_max_abs_error:.6f}, Std Dev Error: {avg_std_dev_error:.6f}")
        return {
            'average_time_ms': avg_time_ms,
            'num_nodes': num_nodes,
            'max_absolute_error': avg_max_abs_error,
            'std_dev_error': avg_std_dev_error
        }

    def run_analysis(self):
        """
        Runs the performance analysis for all specified node selection methods.
        """
        print(f"\nStarting computational performance analysis using segment from {self.image_name}...")
        
        methods_to_test = [
            'full_mesh', 
            'leja',      
            'fekete',    
            'padua'      
        ]

        performance_results = {}
        for method in methods_to_test:
            results = self.measure_method_performance(method)
            performance_results[method] = results

        # Save results to a JSON file
        results_filename = f"poly_approx_performance_deg{self.config.POLY_DEGREE}_{self.image_name}_segment.json"
        results_filepath = self.performance_results_dir / results_filename
        try:
            with open(results_filepath, 'w') as f:
                json.dump(performance_results, f, indent=4)
            print(f"\nSaved performance results to {results_filepath}")
        except Exception as e:
            print(f"Error saving performance results to JSON: {e}")

        print("\nComputational performance analysis finished.")
        print("\nSummary of Results:")
        print(f"{'Method':<25} | {'Avg Time (ms)':<15} | {'Num Nodes':<10} | {'Max Abs Error':<15} | {'Std Dev Error':<15}")
        print("-" * 85) # Adjusted separator length
        for method, data in performance_results.items():
            print(f"{method:<25} | {data['average_time_ms']:<15.4f} | {data['num_nodes']:<10} | {data['max_absolute_error']:<15.6f} | {data['std_dev_error']:<15.6f}")

if __name__ == "__main__":
    if not _poly_approx_available:
        print("Required modules not available. Exiting performance analyzer.")
        sys.exit(1)

    analyzer = ComputationalPerformanceAnalyzer(config_reconstructor, image_name="spiral_and_zigzag") 
    analyzer.run_analysis()
