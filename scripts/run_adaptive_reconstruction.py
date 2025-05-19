# scripts/run_adaptive_reconstruction.py

import sys
from pathlib import Path
import multiprocessing # Import multiprocessing

# Determine the project root dynamically based on the location of this script
# If this script is in 'scripts/', the project root is one level up.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
# Add project root to Python path to allow importing 'config' and 'utils'
sys.path.insert(0, str(PROJECT_ROOT))

# Import the configuration and the reconstructor class
try:
    import config.config as config
    from utils.adaptive_image_reconstruction_class import AdaptivePolynomialReconstructor
except ImportError as e:
    print(f"Error importing modules: {e}")
    print("Please ensure 'config' and 'utils' folders are in your project root")
    print("and contain config.py and adaptive_image_reconstruction_class.py respectively.")
    sys.exit(1)

if __name__ == "__main__":
    # Ensure the script is run as the main program when using multiprocessing
    # This is crucial on Windows for the 'spawn' start method
    multiprocessing.freeze_support()

    # Instantiate the reconstructor with the loaded configuration
    reconstructor = AdaptivePolynomialReconstructor(config)

    # Run the reconstruction process
    reconstructor.run_reconstruction()
