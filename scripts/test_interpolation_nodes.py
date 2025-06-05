import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import os # Import the os module for creating directories
from typing import Dict, Tuple, List, Union # Import necessary types

# Add project root to Python path
# This allows importing modules from the project's root directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Define the base path for the results folder
BASE_RESULTS_DIR = PROJECT_ROOT / "results"

# Define the subfolder for this specific script's results
SCRIPT_RESULTS_DIR = BASE_RESULTS_DIR / "interpolation_nodes_tests"

# Ensure the results directory and the script-specific subfolder exist
os.makedirs(SCRIPT_RESULTS_DIR, exist_ok=True)


# --- Global Matplotlib Plotting Parameters for High Quality Output ---
MATPLOTLIB_PARAMS: Dict[str, Union[str, int, float, bool, List[str]]] = {
    # "text.usetex": True, # Uncomment if you have LaTeX installed
    # "font.family": "serif",
    # "font.serif": ["Computer Modern Roman"],

    "font.family": "serif",
    "font.serif": ["DejaVu Serif"],

    "font.size": 12, # Increased base font size
    "axes.labelsize": 14, # Increased axis label size
    "legend.fontsize": 12, # Increased legend font size
    "xtick.labelsize": 12, # Increased x-tick label size
    "ytick.labelsize": 12, # Increased y-tick label size
    "figure.titlesize": 16, # Increased main figure title size
    "axes.titlesize": 14, # Increased subplot title size

    "lines.linewidth": 1.5,
    "lines.markersize": 6,

    "figure.autolayout": True,
    "savefig.dpi": 600, # High DPI for raster images
    "savefig.format": "pdf", # Prefer vector format
    "figure.figsize": (12, 10), # Adjusted for 2x2 subplots to be larger

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
# │   └── admissible_meshes.py
# └── scripts/
#     └── test_interpolation_nodes.py

# Flag to check if dummy functions are being used
using_dummy_functions = False

# Try importing necessary modules. If it fails, set using_dummy_functions to True.
try:
    # Import necessary functions
    from poly_approx.interpolation_nodes import extremal_points
    # Import the admissible_mesh function
    from poly_approx.admissible_meshes import compute_admissible_mesh as admissible_mesh_function # Import with alias

    _imports_successful = True # Flag to indicate successful import
except ImportError as e:
    print(f"Could not import core modules from poly_approx: {e}")
    print("Please ensure your project structure is correct or adjust the import paths in test_interpolation_nodes.py.")
    print("Expected structure: your_project/poly_approx/ and your_project/scripts/")
    print("Using dummy functions. Plotting will be skipped.")
    using_dummy_functions = True
    # Provide dummy functions that return empty arrays if imports fail
    def extremal_points(*args, **kwargs):
        print("Dummy extremal_points called.")
        return np.array([])
    # Dummy functions for potentially imported but not directly used functions in the main script
    # Define dummy functions for those that were removed from imports but used in the dummy block
    def padua_points_theoretical(*args, **kwargs):
        print("Dummy padua_points_theoretical called.")
        return np.array([])
    def gen_vanderm2d(*args, **kwargs):
        print("Dummy gen_vanderm2d called.")
        return np.eye(1), lambda x: [1] # Return dummy values
    def graded_lexicographic_multi_indices(*args, **kwargs):
        print("Dummy graded_lexicographic_multi_indices called.")
        return [(0,0)]
    def admissible_mesh_function(*args, **kwargs):
        print("Dummy compute_admissible_mesh called.")
        return np.array([]) # Return empty array


if __name__ == "__main__":
    # Apply global Matplotlib settings
    plt.rcParams.update(MATPLOTLIB_PARAMS)

    if using_dummy_functions:
        print("Skipping test due to import errors.")
        sys.exit(1) # Exit the script if imports failed

    # --- Configuration ---
    polynomial_degree = 10 # Degree of the polynomial
    # Define the rectangle for the nodes
    test_rectangle: List[float] = [0.0, 0.0, 5.0, 5.0]
    xmin, ymin, xmax, ymax = test_rectangle # Extract bounds for plotting limits

    # Admissible mesh type is now fixed to 'cheb'
    admissible_mesh_type = 'cheb'
    m_cheb = 2 # Parameter 'm' for Chebyshev mesh construction (m > 1)
    poly_basis_for_leja_fekete = 1 # Polynomial basis to use in Leja/Fekete selection (1: Shifted Monomials)


    # --- Generate Points for each method ---
    print("\nGenerating points using 'full_mesh' method...")
    try:
        full_mesh_points = extremal_points(
            polynomial_degree,
            method='full_mesh',
            rectangle=test_rectangle, # Pass the rectangle
            admissible_mesh_type=admissible_mesh_type, # This is now fixed to 'cheb'
            m_cheb=m_cheb
        )
        print(f"Generated {len(full_mesh_points)} points using 'full_mesh'.")
        admissible_mesh_for_plotting = full_mesh_points
    except Exception as e:
        print(f"Error generating 'full_mesh' points: {e}")
        full_mesh_points = np.array([]) # Assign empty array if error occurs
        admissible_mesh_for_plotting = np.array([]) # Set to empty if generation fails


    print("\nGenerating points using 'leja' method...")
    try:
        leja_points = extremal_points(
            polynomial_degree,
            method='leja',
            rectangle=test_rectangle, # Pass the rectangle
            admissible_mesh_type=admissible_mesh_type, # This is now fixed to 'cheb'
            m_cheb=m_cheb,
            poly_basis=poly_basis_for_leja_fekete # Pass basis parameter
        )
        print(f"Generated {len(leja_points)} points using 'leja'.")
    except ValueError as e:
        print(f"Error generating Leja points: {e}")
        leja_points = np.array([]) # Assign empty array if error occurs
    except Exception as e:
        print(f"An unexpected error occurred while generating Leja points: {e}")
        leja_points = np.array([])


    print("\nGenerating points using 'fekete' method...")
    try:
        fekete_points = extremal_points(
            polynomial_degree,
            method='fekete',
            rectangle=test_rectangle, # Pass the rectangle
            admissible_mesh_type=admissible_mesh_type, # This is now fixed to 'cheb'
            m_cheb=m_cheb,
            poly_basis=poly_basis_for_leja_fekete # Pass basis parameter
        )
        print(f"Generated {len(fekete_points)} points using 'fekete'.")
    except ValueError as e:
        print(f"Error generating Fekete points: {e}")
        fekete_points = np.array([]) # Assign empty array if error occurs
    except Exception as e:
        print(f"An unexpected error occurred while generating Fekete points: {e}")
        fekete_points = np.array([])


    print("\nGenerating points using 'padua' method...")
    try:
        padua_points = extremal_points(
            polynomial_degree,
            method='padua',
            rectangle=test_rectangle # Pass the rectangle
        )
        print(f"Generated {len(padua_points)} points using 'padua'.")
    except ValueError as e:
        print(f"Error generating Padua points: {e}")
        padua_points = np.array([]) # Assign empty array if error occurs
    except Exception as e:
        print(f"An unexpected error occurred while generating Padua points: {e}")
        padua_points = np.array([])


    # --- Visualize the generated points in a single figure with subplots ---
    if not using_dummy_functions and any(len(pts) > 0 for pts in [admissible_mesh_for_plotting, leja_points, fekete_points, padua_points]):
        fig, axes = plt.subplots(2, 2, figsize=(MATPLOTLIB_PARAMS["figure.figsize"][0], MATPLOTLIB_PARAMS["figure.figsize"][1]))
        axes = axes.flatten() # Flatten the 2x2 array of axes for easy iteration

        # Plot 1: Admissible Mesh (using points from the 'full_mesh' method)
        ax1 = axes[0]
        if admissible_mesh_for_plotting.shape[0] > 0:
             ax1.scatter(admissible_mesh_for_plotting[:, 0], admissible_mesh_for_plotting[:, 1], s=5, label=f'{admissible_mesh_type.capitalize()} Admissible Mesh', color='blue')
             # ax1.set_title('(a)') # Commented out simple label
             ax1.set_title(f'{admissible_mesh_type.capitalize()} Admissible Mesh (deg={polynomial_degree}, m={m_cheb})') # Uncommented descriptive title
        else:
             ax1.set_title('Admissible Mesh (Generation Failed)')
        ax1.set_xlabel('x')
        ax1.set_ylabel('y')
        ax1.set_aspect('equal', adjustable='box')
        ax1.grid(True)
        # ax1.legend() # Legend might be redundant with title, remove if desired
        ax1.set_xlim(xmin, xmax)
        ax1.set_ylim(ymin, ymax)


        # Plot 2: Discrete Leja Points
        ax2 = axes[1]
        if leja_points.shape[0] > 0:
            ax2.scatter(leja_points[:, 0], leja_points[:, 1], s=20, color='red', label='Discrete Leja Points')
            # ax2.set_title('(b)') # Commented out simple label
            ax2.set_title(f'Discrete Leja Points (deg={polynomial_degree})') # Uncommented descriptive title
        else:
            ax2.set_title('Discrete Leja Points (Generation Failed)')
        ax2.set_xlabel('x')
        ax2.set_ylabel('y')
        ax2.set_aspect('equal', adjustable='box')
        ax2.grid(True)
        # ax2.legend()
        ax2.set_xlim(xmin, xmax)
        ax2.set_ylim(ymin, ymax)


        # Plot 3: Approximate Fekete Points
        ax3 = axes[2]
        if fekete_points.shape[0] > 0:
            ax3.scatter(fekete_points[:, 0], fekete_points[:, 1], s=20, color='green', label='Approx. Fekete Points')
            # ax3.set_title('(c)') # Commented out simple label
            ax3.set_title(f'Approximate Fekete Points (deg={polynomial_degree})') # Uncommented descriptive title
        else:
            ax3.set_title('Approximate Fekete Points (Generation Failed)')
        ax3.set_xlabel('x')
        ax3.set_ylabel('y')
        ax3.set_aspect('equal', adjustable='box')
        ax3.grid(True)
        # ax3.legend()
        ax3.set_xlim(xmin, xmax)
        ax3.set_ylim(ymin, ymax)


        # Plot 4: Theoretical Padua Points
        ax4 = axes[3]
        if padua_points.shape[0] > 0:
            ax4.scatter(padua_points[:, 0], padua_points[:, 1], s=20, color='purple', label='Padua Points')
            # ax4.set_title('(d)') # Commented out simple label
            ax4.set_title(f'Padua Points (deg={polynomial_degree})') # Uncommented descriptive title
        else:
            ax4.set_title('Padua Points (Generation Failed)')
        ax4.set_xlabel('x')
        ax4.set_ylabel('y')
        ax4.set_aspect('equal', adjustable='box')
        ax4.grid(True)
        # ax4.legend()
        ax4.set_xlim(xmin, xmax)
        ax4.set_ylim(ymin, ymax)


        plt.tight_layout()

        # --- Save the figure to the script-specific results subfolder ---
        # Updated filename to reflect it's a combined plot
        filename = f"node_sets_deg{polynomial_degree}_m{m_cheb}_all_methods.{MATPLOTLIB_PARAMS['savefig.format']}"
        filepath = os.path.join(str(SCRIPT_RESULTS_DIR), filename)

        try:
            plt.savefig(filepath, dpi=PLOT_DPI, format=MATPLOTLIB_PARAMS['savefig.format'])
            print(f"\nSaved combined plot to {filepath}")
        except Exception as e:
            print(f"Error saving combined plot to {filepath}: {e}")

        plt.close(fig) # Close the plot figure to free up memory

    else:
        print("\nSkipping plotting due to import errors or no points generated.")

