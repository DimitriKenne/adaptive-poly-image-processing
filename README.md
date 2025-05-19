# Adaptive Polynomial Image Reconstruction and Edge Detection

This project explores adaptive methods for image processing using polynomial approximation. It includes scripts for reconstructing images based on polynomial fits to adaptively determined segments and for detecting edges based on the reconstruction error.

The core idea is to use the error of a low-degree polynomial approximation within image segments to guide further subdivision (adaptive refinement). Segments where the approximation error is high are subdivided, and the process is repeated. This allows for higher-resolution processing in areas with complex features (like edges) and lower-resolution processing in smoother areas. Edge detection is then performed on the resulting segments based on various error measures.

## Features

- **Adaptive Image Reconstruction**: Reconstructs an image by performing polynomial approximation on segments determined by an adaptive subdivision process based on approximation error.
- **Adaptive Edge Detection**: Detects edges by analyzing the reconstruction error within the adaptively determined segments.
- **Polynomial Approximation**: Supports different polynomial degrees and node selection methods (Padua, Fekete, Leja, Full Mesh).
- **Error Measures**: Various metrics (MSE, MAE, RMSE) to quantify reconstruction error within segments.
- **Error Map Combination**: Strategies to combine different types of error maps (e.g., error on original vs. smoothed image) for robust analysis.
- **Adaptive Subdivision Criteria**: Uses a calculated edge quality measure based on error characteristics near potential edges within a segment to decide whether to subdivide.
- **Dynamic Edge Quality Thresholding**: Automatically calculates a threshold for the edge quality measure based on a quantile of measure values from the full image, reducing the need for manual tuning.
- **Results Visualization**: Generates plots showing the original image, reconstructed image, error heatmaps, and final detected edges.
- **Modular Design**: Code is organized into modules (`poly_approx`) and scripts (`scripts`) for clarity and reusability.

## Project Structure

```
your_project/
├── poly_approx/
│   ├── __init__.py             # Makes poly_approx a Python package
│   ├── admissible_meshes.py    # Functions for generating admissible meshes
│   ├── error_edge_detection.py # Functions for error processing, combination, thresholding, and edge quality measures
│   ├── image_poly_approximation.py # Core polynomial approximation function for image segments
│   ├── interpolation_nodes.py  # Functions for generating interpolation nodes
│   ├── polynomial_bases.py     # Functions for polynomial basis calculation
│   └── poly_projector.py       # Functions for projecting onto polynomial space
├── scripts/
│   ├── adaptive_image_reconstruction_script.py # Script for adaptive image reconstruction
│   └── adaptive_edge_detection_script.py     # Script for adaptive edge detection
├── images/                     # Directory to place input images
│   └── your_image.png          # Example input image
└── results/                    # Directory where output images, data, and plots are saved
    ├── adaptive_image_reconstruction_tests/ # Results from reconstruction script
    │   └── image_name/
    │       └── ... output files ...
    └── adaptive_edge_detection_tests/     # Results from edge detection script
        └── image_name/
            └── ... output files ...
```

## Setup

1. **Clone the repository**

   ```bash
   git clone <repository_url>
   cd your_project
   ```
   _Replace `<repository_url>` with the actual URL if hosted on a platform like GitHub._

2. **Install dependencies**

   This project requires Python 3.x and several libraries. It's recommended to use a virtual environment:

   ```bash
   python -m venv venv
   source venv/bin/activate   # On Windows use `venv\Scripts\activate`

   pip install numpy matplotlib pillow scipy scikit-image
   ```

   - **numpy**: For numerical operations.
   - **matplotlib**: For plotting and visualization.
   - **pillow (PIL Fork)**: For image loading and saving.
   - **scipy**: Required for Gaussian filtering and gradient calculations.
   - **scikit-image**: Required for Otsu's thresholding and morphology operations (like dilation for the edge band).

## Usage

### 1. Adaptive Image Reconstruction

Run the reconstruction script with configurable parameters in the `if __name__ == "__main__":` block:

```bash
python scripts/adaptive_image_reconstruction_script.py
```

**Key Parameters:**

- `image_name`: Name of the input image file (without extension) in the `images/` folder.
- `poly_degree`: Degree of the polynomial approximation.
- `nodes_method`: Node selection method (`full_mesh`, `leja`, `fekete`, `padua`).
- `admissible_mesh_type`: Mesh type for node selection (`cheb`, `uni`).
- `m_cheb`: Parameter for Chebyshev mesh construction.
- `sigma`: Standard deviation for Gaussian smoothing (if applied).
- `error_measure_type`: Error metric for segment quality (`mse`, `mae`, `rmse`).
- `error_threshold`: Tolerance below which segments stop subdividing.
- `max_depth`: Maximum recursion depth.

**Output:**
Results are saved in `results/adaptive_image_reconstruction_tests/<image_name>/`:

- Reassembled approximation image (`*_approx_*.png`).
- Raw error maps per segment (`*.npy`).
- Normalized reconstruction error heatmap (`*_actual_error_viz_*.png`).
- Comparison plot of original, reconstructed, and error heatmap.

### 2. Adaptive Edge Detection

Run the edge detection script with parameters set in the `if __name__ == "__main__":` block:

```bash
python scripts/adaptive_edge_detection_script.py
```

**Key Parameters:**

- `sample_image_path`: Full path to the input image.
- `max_depth`: Maximum recursion depth.
- `min_segment_size`: Minimum segment dimension to stop subdividing.
- `poly_degree`, `nodes_method`, `admissible_mesh_type`, `m_cheb`: Polynomial approximation settings.
- `error_combination_strategy`: How to combine segment error maps (`max`, `weighted_sum`, `logical_and`, `logical_or`).
- `error_combination_weights`: Weights for the `weighted_sum` strategy.
- `edge_threshold_type`: Thresholding method (`fixed`, `otsu`).
- `fixed_edge_threshold`: Value if `edge_threshold_type` is `fixed`.
- `edge_quality_measure_type`: Metric for edge quality subdivision (`gradient_magnitude_near_edges`, `variance_near_edges`, `mean_abs_error_near_edges`).
- `edge_quality_band_width`: Band width around potential edges for measure calculation.
- `quality_threshold_quantile`: Quantile to set dynamic subdivision threshold.

**Output:**
Results are saved in `results/adaptive_edge_detection_tests/<image_name>/`:

- Final reassembled binary edge map (`*_adaptive_edge_map_*.png`).
- Visualization plot of original image vs. detected edges.

## Future Work

- A user-friendly `main.py` or `demo.py` to provide an integrated interface for running reconstruction and edge detection, and visualizing results in a single workflow.

## Contributing

Contributions are welcome! Please open issues or submit pull requests for new features, bug fixes, or documentation improvements.

## License

Add license information here (e.g., MIT, GPL, etc.).
