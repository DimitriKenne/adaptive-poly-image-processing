# scripts/adaptive_edge_detection.py

import sys
from pathlib import Path
import time
import multiprocessing

# Add project root to Python path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import the AdaptiveEdgeDetector class from utils
try:
    from utils.edge_detectors import AdaptiveEdgeDetector
except ImportError:
    print("Error: Could not import AdaptiveEdgeDetector from utils.edge_detectors.")
    print("Please ensure utils/edge_detectors.py exists and is in your Python path.")
    sys.exit(1)

# Import the edge configuration
try:
    import config.edge_config as edge_config_module
except ImportError:
    print("Error: config/edge_config.py not found.")
    print("Please ensure your project structure is correct.")
    sys.exit(1)

# Access the edge_config instance from the imported module
edge_config = edge_config_module.EdgeConfig()


if __name__ == "__main__":
    # Ensure the script is run as the main program when using multiprocessing
    # This is crucial on Windows for the 'spawn' start method
    multiprocessing.freeze_support()

    print("Running Adaptive Edge Detection using the AdaptiveEdgeDetector class.")
    start_time = time.time()

    # Create an instance of the AdaptiveEdgeDetector, passing the configuration
    detector = AdaptiveEdgeDetector(config=edge_config)

    # Run the edge detection process
    detector.run_detection()

    end_time = time.time()
    print(f"\nAdaptive edge detection script finished in {end_time - start_time:.4f} seconds.")

