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
        dummy_error_map = np.zeros((100, 100)) 
        return {
            'computation_time': 0.01,
            'coefficients_original': dummy_coeffs,
            'error_original': dummy_error_map
        }
    
    # Dummy extremal_points function
    def extremal_points(*args, **kwargs): return np.zeros((1, 2))


class ComputationalPerformanceAnalyzer:
    """
    Analyzes the computational performance of different polynomial approximation methods
    by measuring the time taken for a fixed-size segment.
    """

    def __init__(self, config_obj):
        self.config = config_obj
        # Define a fixed size segment for benchmarking
        self.segment_size = (100, 100) # Example: 100x100 pixels
        self.dummy_image_segment = np.random.rand(*self.segment_size).astype(np.float32) # Random dummy image segment
        # A rectangle representing the scaled domain for the dummy segment
        self.dummy_rectangle = (0.0, 0.0, 1.0, 1.0) # Assume it maps to [0,1]x[0,1] for simplicity

        # Results directory for performance data
        self.performance_results_dir = self.config.BASE_RESULTS_DIR / "computational_performance_results"
        os.makedirs(self.performance_results_dir, exist_ok=True)


    def measure_method_performance(self, method: str, num_runs: int = 100) -> Dict[str, Union[float, int]]:
        """
        Measures the average computational time and number of nodes for a given method.

        Args:
            method: The node selection method to test (e.g., 'full_mesh', 'leja', 'fekete', 'padua').
            num_runs: Number of times to run the approximation for averaging.

        Returns:
            A dictionary containing 'average_time_ms' and 'num_nodes'.
        """
        if not _poly_approx_available:
            print(f"Skipping performance measurement for '{method}' due to missing modules.")
            return {'average_time_ms': -1.0, 'num_nodes': -1}

        total_time = 0.0
        num_nodes = 0 

        print(f"Benchmarking '{method}' for {num_runs} runs...")

        # Determine the number of nodes using the extremal_points function
        # This will be consistent with how nodes are generated for approximation.
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
            # Fallback to number of coefficients if extremal_points fails to return nodes
            num_nodes = int((self.config.POLY_DEGREE + 1) * (self.config.POLY_DEGREE + 2) / 2)


        for i in range(num_runs):
            start_time = time.perf_counter()
            try:
                # Call the core approximation function
                # The image_poly_approximation_segment internally calls the appropriate
                # node generation function based on 'method', which should be extremal_points
                # or a wrapper that calls it.
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
                
            except Exception as e:
                print(f"Error during benchmark for method '{method}' run {i}: {e}. Skipping this run.")
                return {'average_time_ms': -1.0, 'num_nodes': num_nodes}


        avg_time_ms = total_time / num_runs
        print(f"'{method}' - Avg Time: {avg_time_ms:.4f} ms, Num Nodes: {num_nodes}")
        return {'average_time_ms': avg_time_ms, 'num_nodes': num_nodes}

    def run_analysis(self):
        """
        Runs the performance analysis for all specified node selection methods.
        """
        print("\nStarting computational performance analysis...")
        
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
        results_filename = f"poly_approx_performance_deg{self.config.POLY_DEGREE}.json"
        results_filepath = self.performance_results_dir / results_filename
        try:
            with open(results_filepath, 'w') as f:
                json.dump(performance_results, f, indent=4)
            print(f"\nSaved performance results to {results_filepath}")
        except Exception as e:
            print(f"Error saving performance results to JSON: {e}")

        print("\nComputational performance analysis finished.")
        print("\nSummary of Results:")
        print(f"{'Method':<25} | {'Avg Time (ms)':<15} | {'Num Nodes':<10}")
        print("-" * 55)
        for method, data in performance_results.items():
            print(f"{method:<25} | {data['average_time_ms']:<15.4f} | {data['num_nodes']:<10}")

if __name__ == "__main__":
    if not _poly_approx_available:
        print("Required modules not available. Exiting performance analyzer.")
        sys.exit(1)

    analyzer = ComputationalPerformanceAnalyzer(config_reconstructor) 
    analyzer.run_analysis()
