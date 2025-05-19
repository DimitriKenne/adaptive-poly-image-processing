import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D # Required for 3D plotting
# Removed import math as it's not directly used
import os # Import the os module for creating directories
from typing import List, Tuple # Import necessary types

# Add project root to Python path
# This allows importing modules from the project's root directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Define the base path for the results folder
BASE_RESULTS_DIR = PROJECT_ROOT / "results"

# Define the subfolder for this specific script's results
SCRIPT_RESULTS_DIR = BASE_RESULTS_DIR / "poly_projector_tests"

# Ensure the results directory and the script-specific subfolder exist
os.makedirs(SCRIPT_RESULTS_DIR, exist_ok=True)


# Assuming your project structure is something like:
# your_project/
# ├── poly_approx/
# │   ├── __init__.py  # This empty file is required to make 'poly_approx' a Python package
# │   ├── interpolation_nodes.py
# │   ├── polynomial_bases.py
# │   ├── poly_projector.py
# │   └── admissible_meshes.py
# └── scripts/
#     └── test_poly_projector.py

# Flag to check if dummy functions are being used
using_dummy_functions = False

try:
    # Import functions from your poly_approx package
    from poly_approx.poly_projector import poly_projector2d # Only need poly_projector2d here
    from poly_approx.interpolation_nodes import extremal_points # Need extremal_points to get nodes
    # Removed imports for evaluate_polynomial_from_coeffs, right_division, padua_points_theoretical,
    # gen_vanderm2d, graded_lexicographic_multi_indices, admissible_mesh as they are not
    # directly called in the main part of this script.

except ImportError as e:
    print(f"Could not import modules from poly_approx: {e}")
    print("Please ensure your project structure is correct or adjust the import paths in test_poly_projector.py.")
    print("Expected structure: your_project/poly_approx/ and your_project/scripts/")
    print("Using dummy functions. Testing and plotting will be skipped.")
    using_dummy_functions = True
    # Provide dummy functions if imports fail
    def poly_projector2d(*args, **kwargs):
        print("Dummy poly_projector2d called.")
        # Return a dummy function that always returns 0 or an array of 0s
        return lambda eval_set: np.zeros(np.atleast_2d(eval_set).shape[0])
    # Dummy functions for those removed from imports but used in dummy block
    def evaluate_polynomial_from_coeffs(*args, **kwargs):
         print("Dummy evaluate_polynomial_from_coeffs called.")
         return np.array([])
    def right_division(*args, **kwargs):
         print("Dummy right_division called.")
         return np.array([])
    def extremal_points(*args, **kwargs):
        print("Dummy extremal_points called.")
        return np.array([])
    def padua_points_theoretical(*args, **kwargs):
        print("Dummy padua_points_theoretical called.")
        return np.array([])
    def gen_vanderm2d(*args, **kwargs):
        print("Dummy gen_vanderm2d called.")
        return np.eye(1), lambda x: [1] # Return dummy values
    def graded_lexicographic_multi_indices(*args, **kwargs):
        print("Dummy graded_lexicographic_multi_indices called.")
        return [(0,0)]
    def admissible_mesh(*args, **kwargs):
        print("Dummy admissible_mesh called.")
        return np.array([]) # Return empty array


# --- Sample 2D function to approximate ---
def sample_function(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """A sample function to test polynomial approximation."""
    # Example: A simple peak function with some noise
    # Ensure x and y are treated as arrays for element-wise operations
    x_arr = np.asarray(x)
    y_arr = np.asarray(y)
    return np.sin(np.pi * x_arr / 5.0) * np.cos(np.pi * y_arr / 5.0) + 0.05 * np.random.rand(*x_arr.shape) # Reduced noise


if __name__ == "__main__":
    if using_dummy_functions:
        print("Skipping test due to import errors.")
        sys.exit(1) # Exit the script if imports failed

    # --- Configuration ---
    polynomial_degree = 5 # Degree of the polynomial approximation
    eval_grid_density = 50 # Density for the evaluation grid (for plotting)

    # Define the rectangle of interest
    test_rectangle: Tuple[float, float, float, float] = (0.0, 0.0, 5.0, 5.0)
    xmin, ymin, xmax, ymax = test_rectangle # Extract bounds

    # Choose a method for selecting nodes.
    # Options: 'full_mesh', 'leja', 'fekete', 'padua'
    # 'leja' and 'fekete' require an admissible mesh (generated internally by extremal_points).
    # 'padua' does not require a mesh.
    nodes_method = 'full_mesh' # Default node selection method
    # Default admissible mesh type (only relevant for 'leja' and 'fekete' if mesh is not explicitly provided)
    # Admissible mesh type is handled internally by extremal_points based on the rectangle
    # admissible_mesh_type = 'cheb' # Options: 'cheb', 'uni' - Removed as it's now internal to extremal_points
    m_cheb = 2 # Parameter 'm' for Chebyshev mesh construction (m > 1)


    # --- 1. Generate Nodes (Interpolation or Least Squares) ---
    print(f"\nGenerating nodes using '{nodes_method}' method...")

    try:
        # extremal_points now takes the rectangle directly and handles mesh generation internally if needed.
        # It returns points within the specified rectangle.
        interpolation_nodes = extremal_points(
            polynomial_degree,
            method=nodes_method,
            rectangle=test_rectangle, # Pass the rectangle directly
            # admissible_mesh_type is handled internally by extremal_points
            m_cheb=m_cheb # Pass m_cheb for internal mesh generation if needed
            # Zc and Zr are not needed here, extremal_points uses the rectangle
        )
        print(f"Generated {len(interpolation_nodes)} nodes for polynomial degree {polynomial_degree}.")
    except ValueError as e:
        print(f"Error generating nodes: {e}")
        print("Cannot proceed with approximation.")
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred while generating nodes: {e}")
        print("Cannot proceed with approximation.")
        sys.exit(1)


    # Expected number of coefficients/basis functions for degree n in 2D is h_n = (n+1)(n+2)/2
    expected_dim = int((polynomial_degree + 1) * (polynomial_degree + 2) / 2)
    print(f"Expected dimension of polynomial space for degree {polynomial_degree}: {expected_dim}")

    # Check if we have enough nodes for the chosen degree and method
    if nodes_method != 'full_mesh' and len(interpolation_nodes) < expected_dim:
        print(f"Warning: Number of generated nodes ({len(interpolation_nodes)}) is less than the expected polynomial dimension ({expected_dim}).")
        print("Using discrete least squares approximation with available nodes.")
        # For least squares, dim can be <= number of nodes.
        # For interpolation, number of nodes must equal dim.
        if len(interpolation_nodes) > 0:
             # Adjust polynomial dimension if not enough nodes for interpolation
            polynomial_dimension = len(interpolation_nodes)
            print(f"Adjusting polynomial dimension to {polynomial_dimension} for approximation.")
        else:
            print("No nodes were generated. Cannot proceed with approximation.")
            sys.exit(1)
    else:
        polynomial_dimension = expected_dim


    # --- 2. Get Function Values at the Nodes ---
    # Evaluate the sample function at the generated interpolation nodes
    function_values_at_nodes = sample_function(interpolation_nodes[:, 0], interpolation_nodes[:, 1])

    # --- 3. Compute the Polynomial Projector (get the evaluation function) ---
    # This step computes the coefficients internally and returns a function to evaluate.
    try:
        # poly_projector2d now takes the rectangle directly
        polynomial_eval_func, _ = poly_projector2d(
            dim=polynomial_dimension, # Use the adjusted dimension
            nodes_set=interpolation_nodes,
            func_values=function_values_at_nodes,
            poly_basis=1, # Use shifted-normalized monomials (adjust if needed)
            rectangle=test_rectangle # Pass the rectangle for basis scaling
            # Zc and Zr are not needed here if rectangle is provided
        )
        print("Polynomial projector computed successfully.")
    except ValueError as e:
        print(f"Error computing polynomial projector: {e}")
        print("Cannot proceed with polynomial evaluation and plotting.")
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred while computing polynomial projector: {e}")
        print("Cannot proceed with polynomial evaluation and plotting.")
        sys.exit(1)


    # --- 4. Evaluate the Polynomial on a Grid for Plotting ---
    # Create a dense grid of points over the rectangle of interest for evaluation
    # Ensure grid dimensions are integers, at least 2 for linspace
    num_eval_points_x = max(2, int((xmax - xmin) * eval_grid_density))
    num_eval_points_y = max(2, int((ymax - ymin) * eval_grid_density))

    x_eval = np.linspace(xmin, xmax, num_eval_points_x)
    y_eval = np.linspace(ymin, ymax, num_eval_points_y)
    X_eval, Y_eval = np.meshgrid(x_eval, y_eval)
    evaluation_points = np.vstack([X_eval.ravel(), Y_eval.ravel()]).T

    print(f"\nEvaluating polynomial on a {num_eval_points_x}x{num_eval_points_y} grid ({len(evaluation_points)} points)...")

    # Evaluate the polynomial at all points in the evaluation grid
    # The polynomial_eval_func handles evaluating at multiple points
    reconstructed_values = polynomial_eval_func(evaluation_points)

    # Reshape the reconstructed values back to a grid for plotting
    Reconstructed_Surface = reconstructed_values.reshape(X_eval.shape)

    # Also evaluate the original function on the same grid for comparison
    Original_Surface = sample_function(X_eval, Y_eval)

    # --- Calculate the absolute error ---
    Absolute_Error = np.abs(Original_Surface - Reconstructed_Surface)


    # --- 5. Visualize the Results ---
    # Increased figure size to accommodate the third plot (heatmap)
    fig = plt.figure(figsize=(18, 6))

    # Plot the original function surface
    ax1 = fig.add_subplot(131, projection='3d') # Changed subplot to 1 row, 3 columns
    ax1.plot_surface(X_eval, Y_eval, Original_Surface, cmap='viridis', rstride=1, cstride=1, alpha=0.8)
    ax1.scatter(interpolation_nodes[:, 0], interpolation_nodes[:, 1], function_values_at_nodes, color='red', s=20, label='Nodes')
    ax1.set_title('Original Function and Nodes')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_zlabel('f(x,y)')
    ax1.legend()

    # Plot the reconstructed polynomial surface
    ax2 = fig.add_subplot(132, projection='3d') # Changed subplot to 1 row, 3 columns
    ax2.plot_surface(X_eval, Y_eval, Reconstructed_Surface, cmap='plasma', rstride=1, cstride=1, alpha=0.8)
    ax2.set_title('Reconstructed Polynomial Surface')
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.set_zlabel('P(x,y)')

    # Plot the heatmap of the absolute error
    ax3 = fig.add_subplot(133) # Added a new subplot for the heatmap
    # Use imshow for heatmap. extent sets the coordinates for the axes.
    # origin='lower' ensures the origin [xmin,ymin] is at the bottom left.
    im = ax3.imshow(Absolute_Error, origin='lower', extent=[xmin, xmax, ymin, ymax],
                    cmap='hot', aspect='auto') # Using 'hot' colormap, aspect='auto' to fit the region dimensions
    fig.colorbar(im, ax=ax3, label='Absolute Error') # Add a colorbar
    ax3.set_title('Absolute Error Heatmap')
    ax3.set_xlabel('x')
    ax3.set_ylabel('y')
    ax3.set_aspect('equal', adjustable='box') # Ensure equal aspect ratio for the region

    plt.tight_layout()

    # --- Save the figure to the script-specific results subfolder ---
    # Include admissible mesh type in the filename if applicable (extremal_points handles this internally now)
    # We can simplify the filename as the mesh type is implicitly Chebyshev
    filename = f"poly_projector_deg{polynomial_degree}_{nodes_method}_nodes_error_heatmap.png"

    filepath = os.path.join(str(SCRIPT_RESULTS_DIR), filename)

    try:
        plt.savefig(filepath)
        print(f"\nSaved plot to {filepath}")
    except Exception as e:
        print(f"\nError saving plot to {filepath}: {e}")


    # Close the plot figure to free up memory
    plt.close(fig)
