import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
# Removed import math as it's not directly used
import os # Import the os module for creating directories
from typing import Tuple, List, Union # Import necessary types

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

    # Removed imports for padua_points_theoretical, gen_vanderm2d, graded_lexicographic_multi_indices
    # as they are only needed in the dummy block or not used in the main script.

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
    if using_dummy_functions:
        print("Skipping test due to import errors.")
        sys.exit(1) # Exit the script if imports failed

    # --- Configuration ---
    polynomial_degree = 10 # Degree of the polynomial
    # Define the rectangle for the nodes
    # Using a list for rectangle bounds
    test_rectangle: List[float] = [0.0, 0.0, 5.0, 5.0]
    xmin, ymin, xmax, ymax = test_rectangle # Extract bounds for plotting limits

    # Admissible mesh type is now fixed to 'cheb'
    admissible_mesh_type = 'cheb'
    m_cheb = 2 # Parameter 'm' for Chebyshev mesh construction (m > 1)
    poly_basis_for_leja_fekete = 1 # Polynomial basis to use in Leja/Fekete selection (1: Shifted Monomials)


    # --- Generate Points for each method ---
    # The 'full_mesh' method will now provide the admissible mesh points for plotting.

    # 1. Full Mesh (uses the generated admissible mesh internally via extremal_points)
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
        # Use the full_mesh_points as the admissible mesh for plotting
        admissible_mesh_for_plotting = full_mesh_points
    except Exception as e:
        print(f"Error generating 'full_mesh' points: {e}")
        full_mesh_points = np.array([]) # Assign empty array if error occurs
        admissible_mesh_for_plotting = np.array([]) # Set to empty if generation fails


    # 2. Discrete Leja Points (uses the generated admissible mesh internally via extremal_points)
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


    # 3. Approximate Fekete Points (uses the generated admissible mesh internally via extremal_points)
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


    # 4. Theoretical Padua Points (generated internally by extremal_points)
    print("\nGenerating points using 'padua' method...")
    # Padua points don't require a mesh, but we specify the rectangle
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


    # --- Visualize the generated points ---
    # Only attempt to plot if imports were successful and at least one set of points was generated
    if not using_dummy_functions and any(len(pts) > 0 for pts in [admissible_mesh_for_plotting, leja_points, fekete_points, padua_points]):
        fig = plt.figure(figsize=(12, 10))

        # Plot Admissible Mesh (using points from the 'full_mesh' method)
        plt.subplot(2, 2, 1)
        if admissible_mesh_for_plotting.shape[0] > 0:
             plt.scatter(admissible_mesh_for_plotting[:, 0], admissible_mesh_for_plotting[:, 1], s=5, label=f'{admissible_mesh_type.capitalize()} Admissible Mesh')
             plt.title(f'{admissible_mesh_type.capitalize()} Admissible Mesh (deg={polynomial_degree})')
        else:
             plt.title('Admissible Mesh (Generation Failed)')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.axis('equal')
        plt.grid(True)
        plt.legend()
        plt.xlim(xmin, xmax) # Use rectangle bounds for plot limits
        plt.ylim(ymin, ymax)


        # Plot Discrete Leja Points
        plt.subplot(2, 2, 2)
        if leja_points.shape[0] > 0:
            plt.scatter(leja_points[:, 0], leja_points[:, 1], s=20, color='red', label='Discrete Leja Points')
            plt.title(f'Discrete Leja Points (deg={polynomial_degree})')
            plt.legend()
        else:
            plt.title('Discrete Leja Points (Generation Failed)')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.axis('equal')
        plt.grid(True)
        plt.xlim(xmin, xmax) # Use rectangle bounds for plot limits
        plt.ylim(ymin, ymax)


        # Plot Approximate Fekete Points
        plt.subplot(2, 2, 3)
        if fekete_points.shape[0] > 0:
            plt.scatter(fekete_points[:, 0], fekete_points[:, 1], s=20, color='green', label='Approx. Fekete Points')
            plt.title(f'Approximate Fekete Points (deg={polynomial_degree})')
            plt.legend()
        else:
            plt.title('Approximate Fekete Points (Generation Failed)')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.axis('equal')
        plt.grid(True)
        plt.xlim(xmin, xmax) # Use rectangle bounds for plot limits
        plt.ylim(ymin, ymax)


        # Plot Theoretical Padua Points
        plt.subplot(2, 2, 4)
        if padua_points.shape[0] > 0:
            plt.scatter(padua_points[:, 0], padua_points[:, 1], s=20, color='purple', label='Theoretical Padua Points')
            plt.title(f'Theoretical Padua Points (deg={polynomial_degree})')
            plt.legend()
        else:
            plt.title('Theoretical Padua Points (Generation Failed)')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.axis('equal')
        plt.grid(True)
        plt.xlim(xmin, xmax) # Use rectangle bounds for plot limits
        plt.ylim(ymin, ymax)


        plt.tight_layout()

        # --- Save the figure to the script-specific results subfolder ---
        filename = f"interpolation_nodes_deg{polynomial_degree}_admesh-{admissible_mesh_type}_m{m_cheb}.png"
        filepath = os.path.join(str(SCRIPT_RESULTS_DIR), filename)

        try:
            plt.savefig(filepath)
            print(f"\nSaved plot to {filepath}")
        except Exception as e:
            print(f"Error saving plot to {filepath}: {e}")


        # Close the plot figure to free up memory
        plt.close(fig)

    else:
        print("\nSkipping plotting due to import errors or no points generated.")
