import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D # Required for 3D plotting
import os # Import the os module for creating directories
from typing import List, Tuple, Dict, Union # Import necessary types

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

# --- Global Matplotlib Plotting Parameters for High Quality Output ---
MATPLOTLIB_PARAMS: Dict[str, Union[str, int, float, bool, List[str]]] = {
    # "text.usetex": True, # Uncomment if you have LaTeX installed
    # "font.family": "serif",
    # "font.serif": ["Computer Modern Roman"],

    "font.family": "serif",
    "font.serif": ["DejaVu Serif"],

    "font.size": 10,
    "axes.labelsize": 10,
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "figure.titlesize": 12,

    "lines.linewidth": 1.5,
    "lines.markersize": 6,

    "figure.autolayout": True,
    "savefig.dpi": 600, # High DPI for raster images
    "savefig.format": "pdf", # Prefer vector format
    "figure.figsize": (18, 6), # Adjusted for 1x3 subplots

    "axes.grid": True,
    "grid.linestyle": ':',
    "grid.alpha": 0.6,
}

# Specific DPI for raster images (PNG) when saving
PLOT_DPI = 600


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
except ImportError as e:
    print(f"Could not import modules from poly_approx: {e}")
    print("Please ensure your project structure is correct or adjust the import paths in test_poly_projector.py.")
    print("Expected structure: your_project/poly_approx/ and your_project/scripts/")
    print("Using dummy functions. Testing and plotting will be skipped.")
    using_dummy_functions = True
    # Provide dummy functions if imports fail
    def poly_projector2d(*args, **kwargs):
        print("Dummy poly_projector2d called.")
        return lambda eval_set: np.zeros(np.atleast_2d(eval_set).shape[0])
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
        return np.eye(1), lambda x: [1]
    def graded_lexicographic_multi_indices(*args, **kwargs):
        print("Dummy graded_lexicographic_multi_indices called.")
        return [(0,0)]
    def admissible_mesh(*args, **kwargs):
        print("Dummy admissible_mesh called.")
        return np.array([])


# --- Sample 2D function to approximate ---
def sample_function(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """A sample function to test polynomial approximation."""
    x_arr = np.asarray(x)
    y_arr = np.asarray(y)
    return np.sin(np.pi * x_arr / 5.0) * np.cos(np.pi * y_arr / 5.0) + 0.05 * np.random.rand(*x_arr.shape)

if __name__ == "__main__":
    # Apply global Matplotlib settings
    plt.rcParams.update(MATPLOTLIB_PARAMS)

    if using_dummy_functions:
        print("Skipping test due to import errors.")
        sys.exit(1)

    # --- Configuration ---
    polynomial_degree = 5
    eval_grid_density = 50

    test_rectangle: Tuple[float, float, float, float] = (0.0, 0.0, 5.0, 5.0)
    xmin, ymin, xmax, ymax = test_rectangle

    nodes_method = 'full_mesh'
    m_cheb = 2

    # --- 1. Generate Nodes (Interpolation or Least Squares) ---
    print(f"\nGenerating nodes using '{nodes_method}' method...")
    try:
        interpolation_nodes = extremal_points(
            polynomial_degree,
            method=nodes_method,
            rectangle=test_rectangle,
            m_cheb=m_cheb
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

    expected_dim = int((polynomial_degree + 1) * (polynomial_degree + 2) / 2)
    print(f"Expected dimension of polynomial space for degree {polynomial_degree}: {expected_dim}")

    if nodes_method != 'full_mesh' and len(interpolation_nodes) < expected_dim:
        print(f"Warning: Number of generated nodes ({len(interpolation_nodes)}) is less than the expected polynomial dimension ({expected_dim}).")
        print("Using discrete least squares approximation with available nodes.")
        if len(interpolation_nodes) > 0:
            polynomial_dimension = len(interpolation_nodes)
            print(f"Adjusting polynomial dimension to {polynomial_dimension} for approximation.")
        else:
            print("No nodes were generated. Cannot proceed with approximation.")
            sys.exit(1)
    else:
        polynomial_dimension = expected_dim

    # --- 2. Get Function Values at the Nodes ---
    function_values_at_nodes = sample_function(interpolation_nodes[:, 0], interpolation_nodes[:, 1])

    # --- 3. Compute the Polynomial Projector (get the evaluation function) ---
    try:
        polynomial_eval_func, _ = poly_projector2d(
            dim=polynomial_dimension,
            nodes_set=interpolation_nodes,
            func_values=function_values_at_nodes,
            poly_basis=1,
            rectangle=test_rectangle
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
    num_eval_points_x = max(2, int((xmax - xmin) * eval_grid_density))
    num_eval_points_y = max(2, int((ymax - ymin) * eval_grid_density))

    x_eval = np.linspace(xmin, xmax, num_eval_points_x)
    y_eval = np.linspace(ymin, ymax, num_eval_points_y)
    X_eval, Y_eval = np.meshgrid(x_eval, y_eval)
    # Corrected typo: YY_eval -> Y_eval
    evaluation_points = np.vstack([X_eval.ravel(), Y_eval.ravel()]).T

    print(f"\nEvaluating polynomial on a {num_eval_points_x}x{num_eval_points_y} grid ({len(evaluation_points)} points)...")

    reconstructed_values = polynomial_eval_func(evaluation_points)
    Reconstructed_Surface = reconstructed_values.reshape(X_eval.shape)
    Original_Surface = sample_function(X_eval, Y_eval)
    Absolute_Error = np.abs(Original_Surface - Reconstructed_Surface)

    # --- 5. Visualize the Results ---
    fig = plt.figure(figsize=(MATPLOTLIB_PARAMS["figure.figsize"][0], MATPLOTLIB_PARAMS["figure.figsize"][1]))

    ax1 = fig.add_subplot(131, projection='3d')
    ax1.plot_surface(X_eval, Y_eval, Original_Surface, cmap='viridis', rstride=1, cstride=1, alpha=0.8)
    ax1.scatter(interpolation_nodes[:, 0], interpolation_nodes[:, 1], function_values_at_nodes, color='red', s=20, label='Nodes')
    ax1.set_title('Original Function and Nodes')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_zlabel('f(x,y)')
    ax1.legend()

    ax2 = fig.add_subplot(132, projection='3d')
    ax2.plot_surface(X_eval, Y_eval, Reconstructed_Surface, cmap='plasma', rstride=1, cstride=1, alpha=0.8)
    ax2.set_title('Reconstructed Polynomial Surface')
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.set_zlabel('P(x,y)')

    ax3 = fig.add_subplot(133)
    im = ax3.imshow(Absolute_Error, origin='lower', extent=[xmin, xmax, ymin, ymax],
                    cmap='hot', aspect='auto')
    fig.colorbar(im, ax=ax3, label='Absolute Error')
    ax3.set_title('Absolute Error Heatmap')
    ax3.set_xlabel('x')
    ax3.set_ylabel('y')
    ax3.set_aspect('equal', adjustable='box')

    plt.tight_layout()

    # --- Save the figure to the script-specific results subfolder ---
    filename = f"poly_projector_deg{polynomial_degree}_{nodes_method}_nodes_error_heatmap.{MATPLOTLIB_PARAMS['savefig.format']}"
    filepath = os.path.join(str(SCRIPT_RESULTS_DIR), filename)

    try:
        plt.savefig(filepath, dpi=PLOT_DPI, format=MATPLOTLIB_PARAMS['savefig.format'])
        print(f"\nSaved plot to {filepath}")
    except Exception as e:
        print(f"\nError saving plot to {filepath}: {e}")

    plt.close(fig)
