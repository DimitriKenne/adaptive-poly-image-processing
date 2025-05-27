# scripts/simple_edge_detection.py

import sys
from pathlib import Path
import time

# Add project root to Python path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import the SimpleEdgeDetector class from utils
try:
    from utils.edge_detector import SimpleEdgeDetector
except ImportError:
    print("Error: Could not import SimpleEdgeDetector from utils.edge_detectors.")
    print("Please ensure utils/edge_detectors.py exists and is in your Python path.")
    sys.exit(1)

# Import the edge configuration module
try:
    import config.edge_config as edge_config_module
except ImportError:
    print("Error: config/edge_config.py not found.")
    print("Please ensure your project structure is correct.")
    sys.exit(1)

# Access the edge_config instance from the imported module
edge_config = edge_config_module.edge_config


if __name__ == "__main__":
    print("Running Simple Edge Detection using the SimpleEdgeDetector class.")
    start_time = time.time()

    # Create an instance of the SimpleEdgeDetector, passing the configuration
    detector = SimpleEdgeDetector(config=edge_config)

    # Run the edge detection process
    detector.run_detection()

    end_time = time.time()
    print(f"\nSimple edge detection script finished in {end_time - start_time:.4f} seconds.")
