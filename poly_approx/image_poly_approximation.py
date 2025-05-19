import numpy as np
import time
# Re-adding gaussian_filter as it's used for smoothing the segment
from scipy.ndimage import gaussian_filter
from PIL import Image # Still needed for save_images
import os # Still needed for save_images and path handling
from typing import Optional, Callable, Tuple # Keep necessary types
from pathlib import Path # Still needed for save_images path handling
import math # Import math for comb (needed for expected_dim)


# Corrected relative imports for modules within the same package (poly_approx)
# poly_projector2d is now expected to return coefficients
from .poly_projector import poly_projector2d
from .interpolation_nodes import extremal_points


def save_images(image_dict, save_folder):
    """
    Save images to the specified folder.

    Parameters:
    -----------
    image_dict : dict
        Dictionary containing image names as keys and image arrays as values.
        Image arrays are expected to be in the range [0, 1] for standard image formats.
        Error maps should be normalized to [0, 1] before being passed to this function.
    save_folder : str
        Folder path where images will be saved.
    """
    os.makedirs(save_folder, exist_ok=True)

    for filename, img in image_dict.items():
        # print(f"Saving {filename}: Min={img.min()}, Max={img.max()}") # Optional: uncomment for debugging

        # --- Apply min-max normalization to ALL images before saving ---
        # This ensures the full [0, 1] range is used for visualization in the saved image.
        min_val = np.min(img)
        max_val = np.max(img)

        # Handle the case where the image is constant (min == max)
        epsilon = 1e-8 # Use a small epsilon for floating point comparison
        if max_val - min_val < epsilon:
            # If all values are the same, save as a uniform grayscale image
            # If the constant value is close to 0, save as black (0).
            # If close to 1, save as white (255).
            # Otherwise, save as a mid-gray value scaled by the constant value.
            if np.abs(min_val) < epsilon: # Close to black
                img_to_save = np.zeros_like(img)
            elif np.abs(min_val - 1.0) < epsilon: # Close to white
                 img_to_save = np.ones_like(img)
            else: # Constant but not black or white
                 # Scale the constant value to [0, 1] range for saving
                 img_to_save = np.full_like(img, fill_value=min_val) # Use the constant value directly

        else:
            # Min-max normalization to scale to [0, 1]
            img_to_save = (img - min_val) / (max_val - min_val)

        # Clip to [0, 1] as a safeguard after normalization
        img_to_save = np.clip(img_to_save, 0, 1)

        # Convert to uint8 [0, 255] for standard grayscale PNG
        img_uint8 = (img_to_save * 255).astype(np.uint8)

        filepath = os.path.join(save_folder, filename)
        try:
            Image.fromarray(img_uint8, 'L').save(filepath)
            # print(f"Saved image to {filepath}") # Optional: uncomment for debugging
        except Exception as e:
            print(f"Error saving image {filename} to {filepath}: {e}")


def compute_polynomial_approx(image_segment: np.ndarray, poly_degree: int, points: np.ndarray, rectangle: Tuple[float, float, float, float]) -> Tuple[Callable, np.ndarray]:
    """
    Compute the polynomial interpolation (or least-squares fit) for given points
    on an image segment and return the evaluation function and coefficients.

    Parameters:
    -----------
    image_segment : ndarray
        The 2D numpy array representing the image segment (grayscale).
    poly_degree : int
        The degree of the polynomial approximation.
    points : ndarray
        The 2D numpy array of interpolation/approximation points. These points
        are expected to be within the bounds of the 'rectangle'.
    rectangle : tuple
        The bounds of the rectangular domain [xmin, ymin, xmax, ymax] that the
        'points' belong to. This rectangle's coordinates should be relative to
        the original image scaled to [0,1]x[0,1].

    Returns:
    --------
    Tuple[Callable, ndarray]
        A tuple containing:
        - A function that evaluates the computed polynomial at given 2D points.
          The evaluation function expects input points in the same domain as the
          original 'points' (i.e., within the 'rectangle' bounds relative to [0,1]x[0,1]).
        - The numpy array of polynomial coefficients.
    """
    height, width = image_segment.shape

    # --- Sampling: Map points from rectangle domain to pixel coordinates ---
    # The points are in the rectangle [xmin, ymin, xmax, ymax] relative to the original image [0,1]x[0,1].
    # To get the corresponding pixel coordinates in the *segment*, we need to map from
    # [xmin, ymin] to [0,0] and [xmax, ymax] to [width-1, height-1] of the segment.
    # This assumes the segment data corresponds exactly to the pixel region defined by the rectangle.
    # Let's assume the rectangle [xmin, ymin, xmax, ymax] maps to the segment's pixel bounds [0, width-1] x [0, height-1].
    xmin, ymin, xmax, ymax = rectangle
    range_x = xmax - xmin
    range_y = ymax - ymin

    # Handle degenerate rectangle case (zero range)
    epsilon = 1e-9
    if range_x < epsilon:
        range_x = epsilon
    if range_y < epsilon:
        range_y = epsilon

    # Map points from [xmin, xmax] to [0, width-1] and [ymin, ymax] to [0, height-1]
    x_coords_pixel = (points[:, 0] - xmin) / range_x * (width - 1)
    y_coords_pixel = (points[:, 1] - ymin) / range_y * (height - 1)


    # Use nearest neighbor interpolation to get pixel values at potentially non-integer coordinates
    # Ensure indices are within bounds after scaling
    x_coords_int = np.clip(np.round(x_coords_pixel).astype(int), 0, width - 1)
    y_coords_int = np.clip(np.round(y_coords_pixel).astype(int), 0, height - 1)

    # Get function values (image intensities) at the sampled pixel locations
    func_values = image_segment[y_coords_int, x_coords_int]

    # The dimension of the polynomial space for degree 'poly_degree' in 2D
    # This is needed by poly_projector2d to know the number of basis functions
    poly_dimension = int((poly_degree + 1) * (poly_degree + 2) / 2) # Using formula instead of math.comb for simplicity

    # Compute the polynomial using the selected points and their function values
    # Pass the original 'points' (in the rectangle domain) and the rectangle bounds
    # poly_projector2d now returns both the function and the coefficients
    poly_func, coefficients = poly_projector2d(poly_dimension, points, func_values, poly_basis=1, rectangle=rectangle) # Pass rectangle

    # The returned function 'poly_func' expects evaluation points in the same domain as the input 'points'
    return poly_func, coefficients # Return both the function and coefficients


def image_poly_approximation_segment(
    image_segment: np.ndarray,
    rectangle: Tuple[float, float, float, float], # Rectangle parameter
    poly_degree: int = 5,
    nodes_method: str = 'leja',
    admissible_mesh_type: str = 'cheb', # Parameter for mesh generation
    m_cheb: int = 2, # Parameter for Chebyshev mesh
    poly_basis: int = 1 # Parameter for basis in Leja/Fekete (passed to gen_vanderm2d)
) -> dict:
    """
    Perform polynomial approximation on a single image segment within a specified rectangle.

    This function takes a preprocessed image segment (e.g., grayscale, normalized)
    and computes its polynomial approximation based on selected nodes within the
    given rectangle. It returns the raw error maps, approximations, and coefficients.

    Parameters:
    -----------
    image_segment : ndarray
        The 2D numpy array representing the image segment (grayscale, normalized).
    rectangle : tuple
        The bounds of the rectangular domain [xmin, ymin, xmax, ymax] that this
        segment corresponds to relative to the original image scaled to [0,1]x[0,1].
        Nodes will be generated within this rectangle.
    poly_degree : int, optional
        The degree of the polynomial approximation. Default is 5.
    nodes_method : str, optional
        The method for selecting interpolation/approximation nodes.
        Options: 'full_mesh', 'leja', 'fekete', 'padua'. Default is 'leja'.
    admissible_mesh_type : str, optional
        The type of admissible mesh to generate for 'leja', 'fekete', and 'full_mesh'
        methods ('cheb' only supported now). Default is 'cheb'.
    m_cheb : int, optional
        Parameter 'm' for Chebyshev mesh construction (m > 1). Default is 2.
    poly_basis : int, optional
        Indicate the polynomial basis to be used for the Vandermonde matrix
        within discrete Leja and Fekete point selection (1: Shifted Monomials,
        2: Monomial, 3: Chebyshev). Default is 1.


    Returns:
    --------
    dict
        A dictionary containing the original segment, smoothed segment,
        polynomial approximations, raw error maps, the nodes used, and the
        polynomial coefficients. Returns an empty dictionary if an error occurs.
    """
    height, width = image_segment.shape
    print(f"Processing segment with shape: ({height}, {width}) within rectangle {rectangle}")

    xmin, ymin, xmax, ymax = rectangle

    # Apply Gaussian smoothing to the segment
    # Sigma should be relative to the segment size, or a fixed small value
    # Using a small fixed sigma for now, adjust as needed.
    smoothed_segment = gaussian_filter(image_segment, sigma=1.0) # Adjusted sigma


    # --- Generate Nodes (Interpolation or Least Squares) ---
    start_time = time.time()
    print(f"Selecting interpolation points using '{nodes_method}' method for degree {poly_degree}...")

    try:
        # extremal_points now directly takes the rectangle and handles mesh generation internally if needed.
        # It returns points within the specified rectangle.
        points = extremal_points(
            poly_degree,
            method=nodes_method,
            rectangle=rectangle, # Pass the rectangle directly
            admissible_mesh_type=admissible_mesh_type,
            m_cheb=m_cheb,
            poly_basis=poly_basis # Pass basis parameter for Leja/Fekete
        )
        print(f"Generated {len(points)} nodes.")
    except ValueError as e:
        print(f"Error generating nodes: {e}")
        return {} # Return empty dict on error
    except Exception as e:
        print(f"An unexpected error occurred while generating nodes: {e}")
        return {}


    # Expected number of coefficients/basis functions for degree n in 2D is h_n = (n+1)(n+2)/2
    expected_dim = int((poly_degree + 1) * (poly_degree + 2) / 2)
    # print(f"Expected dimension of polynomial space for degree {poly_degree}: {expected_dim}") # Optional print

    # Determine the dimension for the polynomial projector
    # If the number of generated nodes is less than the expected dimension for interpolation,
    # we fall back to least squares with the available nodes.
    # For 'full_mesh', we use the expected dimension if the mesh is large enough, otherwise the mesh size.
    if nodes_method != 'full_mesh' and len(points) < expected_dim:
         print(f"Warning: Number of generated nodes ({len(points)}) is less than the expected polynomial dimension ({expected_dim}).")
         print("Using discrete least squares approximation with available nodes.")
         # polynomial_dimension = len(points) # Use number of nodes as dimension for least squares - Not needed for poly_projector2d call
    elif nodes_method == 'full_mesh' and len(points) < expected_dim:
         print(f"Warning: Full mesh size ({len(points)}) is less than the expected polynomial dimension ({expected_dim}).")
         print("Using discrete least squares approximation with the full mesh.")
         # polynomial_dimension = len(points) # Use mesh size as dimension for least squares - Not needed for poly_projector2d call
    # else:
        # polynomial_dimension = expected_dim # Use expected dimension for interpolation or full mesh LS (if large enough) - Not needed for poly_projector2d call


    # --- Compute Polynomial Approximations and Get Coefficients ---
    print("Computing polynomial approximations and extracting coefficients...")

    try:
        # Compute polynomial for the original segment
        # compute_polynomial_approx now returns (poly_func, coefficients)
        poly_original_func, coeffs_original = compute_polynomial_approx(image_segment, poly_degree, points, rectangle)

        # Compute polynomial for the smoothed segment
        # The same points and rectangle are used
        poly_smoothed_func, coeffs_smoothed = compute_polynomial_approx(smoothed_segment, poly_degree, points, rectangle)

    except ValueError as e:
        print(f"Error computing polynomial approximations or getting coefficients: {e}")
        return {} # Return empty dict on error
    except Exception as e:
        print(f"An unexpected error occurred while computing polynomial approximations or getting coefficients: {e}")
        return {}


    # --- Evaluate Polynomials over the Segment Grid ---
    # Create a grid of points over the segment's pixel coordinates [0, width-1] x [0, height-1]
    # scaled to the rectangle [xmin, ymin] to [xmax, ymax] for evaluation.
    # The evaluation grid points should be in the same domain as the input 'points' for the polynomial function.
    x_eval_scaled = np.linspace(xmin, xmax, width)
    y_eval_scaled = np.linspace(ymin, ymax, height)
    XX_eval_scaled, YY_eval_scaled = np.meshgrid(x_eval_scaled, y_eval_scaled)
    evaluation_points_scaled = np.vstack([XX_eval_scaled.ravel(), YY_eval_scaled.ravel()]).T

    # Evaluate the polynomials at all points in the evaluation grid
    approx_original_flat = poly_original_func(evaluation_points_scaled).real
    approx_smoothed_flat = poly_smoothed_func(evaluation_points_scaled).real

    # Reshape the approximated values back to the segment shape
    # Clip to [0, 1] to ensure valid image intensity range
    approx_original = np.clip(approx_original_flat.reshape(height, width), 0, 1)
    approx_smoothed = np.clip(approx_smoothed_flat.reshape(height, width), 0, 1)

    # --- Compute Raw Error Maps ---
    # Errors are computed on the original scale, not normalized yet.
    error_original = np.abs(image_segment - approx_original)
    error_smoothed = np.abs(smoothed_segment - approx_smoothed)
    diff_original_poly_smoothed = np.abs(image_segment - approx_smoothed)
    diff_smoothed_poly_original = np.abs(smoothed_segment - approx_original)

    elapsed_time = time.time() - start_time
    print(f"Approximation and error computation completed in {elapsed_time:.4f} seconds")

    # Return results as a dictionary, now including coefficients
    # Raw error maps are returned, normalization happens later if needed.
    return {
        'original_segment': image_segment,
        'smoothed_segment': smoothed_segment,
        'approx_original': approx_original,
        'approx_smoothed': approx_smoothed,
        'error_original': error_original, # Raw error
        'error_smoothed': error_smoothed, # Raw error
        'diff_original_poly_smoothed': diff_original_poly_smoothed, # Raw error
        'diff_smoothed_poly_original': diff_smoothed_poly_original, # Raw error
        'computation_time': elapsed_time,
        'nodes': points, # Include the nodes used
        'coefficients_original': coeffs_original, # Include coefficients for original
        'coefficients_smoothed': coeffs_smoothed # Include coefficients for smoothed
    }

