import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import os # Import the os module for creating directories
from typing import List, Dict, Union # Import List for type hinting

# Add project root to Python path
# This allows importing modules from the project's root directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Define the base path for the results folder
BASE_RESULTS_DIR = PROJECT_ROOT / "results"

# Define the subfolder for this specific script's results
SCRIPT_RESULTS_DIR = BASE_RESULTS_DIR / "admissible_mesh_tests"

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
    "figure.figsize": (8, 8), # Default figure size for single plots

    "axes.grid": True,
    "grid.linestyle": ':',
    "grid.alpha": 0.6,
}

# Specific DPI for raster images (PNG) when saving
PLOT_DPI = 600


# Assuming your project structure is something like:
# your_project/
# ├── poly_approx/
# │   ├── __init__.py
# │   └── admissible_meshes.py # Your updated file
# └── scripts/
#     └── test_admissible_meshes.py # This script

# Flag to check if dummy functions are being used (though less likely for this simple script)
using_dummy_functions = False

try:
    # Import the compute_admissible_mesh function
    from poly_approx.admissible_meshes import compute_admissible_mesh
except ImportError as e:
    print(f"Could not import compute_admissible_mesh module: {e}")
    print("Please ensure your project structure is correct and admissible_meshes.py is in poly_approx.")
    print("Skipping tests.")
    using_dummy_functions = True
    # Provide a dummy function if import fails
    def compute_admissible_mesh(*args, **kwargs):
        print("Dummy compute_admissible_mesh called.")
        return np.array([]) # Return empty array


if __name__ == "__main__":
    # Apply global Matplotlib settings
    plt.rcParams.update(MATPLOTLIB_PARAMS)

    if using_dummy_functions:
        sys.exit(1) # Exit if the necessary module could not be imported

    # --- Configuration ---
    # Define the degrees you want to test
    degrees_to_test: List[int] = [1, 2, 3, 4, 5] # Example degrees

    # Define the rectangle bounds for the meshes
    test_rectangle: List[float] = [0.0, 0.0, 1.0, 1.0]
    x_min, y_min, x_max, y_max = test_rectangle # Extract bounds for plotting limits

    # Parameter 'm' for Chebyshev mesh (must be > 1)
    m_cheb = 2 # Example value for m


    # --- Generate and Plot Chebyshev Meshes for each Degree ---
    print(f"Testing Chebyshev admissible mesh generation for degrees: {degrees_to_test}")

    for deg in degrees_to_test:
        print(f"\n--- Generating Chebyshev mesh for degree {deg} ---")

        try:
            # Generate Chebyshev Admissible Mesh
            # mesh_type is implicitly 'cheb' as it's the only supported type now
            cheby_ad_mesh = compute_admissible_mesh(
                deg=deg,
                mesh_type="cheb", # Explicitly pass "cheb"
                m=m_cheb,
                x_min=x_min,
                x_max=x_max,
                y_min=y_min,
                y_max=y_max
            )
            num_cheby_points = len(cheby_ad_mesh)
            print(f"Generated Chebyshev mesh with {num_cheby_points} points.")

            # --- Visualize the Mesh ---
            fig, ax = plt.subplots(1, 1, figsize=(MATPLOTLIB_PARAMS["figure.figsize"][0], MATPLOTLIB_PARAMS["figure.figsize"][1])) # Use configured figsize

            # Plot Chebyshev Mesh
            if num_cheby_points > 0:
                ax.scatter(cheby_ad_mesh[:, 0], cheby_ad_mesh[:, 1], s=10)
            ax.set_title(f"Chebyshev Admissible Mesh (deg={deg}, m={m_cheb})\nPoints: {num_cheby_points}")
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.set_aspect('equal', adjustable='box') # Ensure equal aspect ratio
            ax.grid(True)
            ax.set_xlim(x_min, x_max) # Set plot limits to rectangle bounds
            ax.set_ylim(y_min, y_max)


            plt.tight_layout()

            # --- Save the figure to the script-specific results subfolder ---
            plot_filename = f"admissible_mesh_deg{deg}_cheb_m{m_cheb}.{MATPLOTLIB_PARAMS['savefig.format']}"
            # Use os.path.join for robust path creation
            plot_filepath = os.path.join(str(SCRIPT_RESULTS_DIR), plot_filename)

            try:
                plt.savefig(plot_filepath, dpi=PLOT_DPI, format=MATPLOTLIB_PARAMS['savefig.format'])
                print(f"Saved plot to {plot_filepath}")
            except Exception as e:
                print(f"Error saving plot to {plot_filepath}: {e}")

            # Close the plot figure
            plt.close(fig)

        except ValueError as e:
            print(f"Error generating Chebyshev mesh for degree {deg}: {e}")
        except Exception as e:
            print(f"An unexpected error occurred for degree {deg}: {e}")

    print("\nAdmissible mesh testing complete.")
