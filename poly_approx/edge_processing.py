# poly_approx/edge_processing.py

import numpy as np
from typing import Dict, Optional, Tuple
# Import skimage for Otsu's method and morphology operations if available
try:
    from skimage.filters import threshold_otsu
    from skimage.morphology import disk, binary_dilation
    _skimage_available = True
except ImportError:
    print("Warning: scikit-image not found. Some functions (Otsu, morphology) will not be available.")
    _skimage_available = False

# Import scipy for gradient calculation
try:
    from scipy.ndimage import gaussian_filter
    _scipy_available = True
except ImportError:
    print("Warning: scipy not found. Gradient calculation will not be available.")
    _scipy_available = False

# Import normalize_error_image from image_reconstruction_metrics
from poly_approx.image_reconstruction_metrics import normalize_error_image


def combine_errors(
    error_dict: Dict[str, np.ndarray],
    strategy: str = 'max',
    weights: Optional[Dict[str, float]] = None
) -> np.ndarray:
    """
    Combine multiple normalized error images into a single composite.

    Parameters
    ----------
    error_dict : Dict[str, np.ndarray]
        Normalized error images ([0, 1]). All must have the same shape.
    strategy : str, optional
        Combination method: 'max', 'weighted_sum'.
        Default is 'max'.
    weights : Optional[Dict[str, float]], optional
        Weights for 'weighted_sum'. Keys must match error_dict keys.

    Returns
    -------
    np.ndarray
        The composite error image in the range [0, 1].

    Raises
    ------
    ValueError
        If invalid strategy, or weights mismatch/missing for 'weighted_sum'.
    """
    if not error_dict:
        print("Warning: combine_errors received an empty error_dict. Returning empty array.")
        return np.array([], dtype=np.float32).reshape(0, 0)


    shapes = [img.shape for img in error_dict.values()]
    if not all(s == shapes[0] for s in shapes):
        raise ValueError("All error maps must have the same shape for combination.")

    img_shape = shapes[0]
    normalized_errors = list(error_dict.values())

    if strategy == 'max':
        # Stack and take max along the last axis
        return np.max(np.stack(normalized_errors, axis=-1), axis=-1)

    elif strategy == 'weighted_sum':
        if weights is None:
            num_errors = len(normalized_errors)
            if num_errors == 0:
                 return np.zeros(img_shape, dtype=np.float32) # Return zero if no errors to sum
            # Default to equal weights if none provided
            weights = {key: 1.0 / num_errors for key in error_dict.keys()}

        if set(weights.keys()) != set(error_dict.keys()):
            raise ValueError("Keys in 'weights' dictionary must match keys in 'error_dict'.")

        composite_error = np.sum([weights[key] * error_img for key, error_img in error_dict.items()], axis=0)
        return np.clip(composite_error, 0, 1)

    # Removed 'logical_and' and 'logical_or' strategies
    else:
        raise ValueError(f"Invalid combination strategy: {strategy}. Expected 'max' or 'weighted_sum'.")


def simple_threshold(image: np.ndarray, threshold: float) -> np.ndarray:
    """
    Apply a fixed threshold to an image to create a binary map.

    Parameters
    ----------
    image : np.ndarray
        The input image (2D numpy array).
    threshold : float
        The threshold value.

    Returns
    -------
    np.ndarray
        A binary map (0 or 1, uint8 dtype).
    """
    if image.size == 0:
        return np.zeros_like(image, dtype=np.uint8)
    return (image > threshold).astype(np.uint8)


def otsu_threshold(image: np.ndarray) -> np.ndarray:
    """
    Apply Otsu's global thresholding to an image.

    Parameters:
    -----------
    image : ndarray
        The input image (2D numpy array).

    Returns:
    --------
    ndarray
        A binary map (0 or 1, uint8 dtype).

    Raises:
    -------
    ImportError
        If scikit-image is not available.
    ValueError
        If the input image is constant or empty.
    """
    if not _skimage_available:
        raise ImportError("Otsu's method requires scikit-image.")

    if image.size == 0:
        # Return an empty binary array with uint8 dtype
        return np.array([], dtype=np.uint8).reshape(image.shape)

    # Scale image to uint8 [0, 255] for Otsu if it's float in [0, 1]
    img_scaled = image
    if np.max(image) <= 1.0 + 1e-8 and np.min(image) >= 0.0 - 1e-8:
         img_scaled = (image * 255).astype(np.uint8)
    elif np.max(image) <= 255.0 + 1e-8 and np.min(image) >= 0.0 - 1e-8:
         img_scaled = image.astype(np.uint8)
    else:
         # Handle other potential ranges by normalizing to [0, 255]
         min_val = np.min(image)
         max_val = np.max(image)
         if max_val - min_val < 1e-8:
             # Handle constant image case before scaling
             print("Warning: Input image is constant for Otsu. Returning zero map.")
             return np.zeros_like(image, dtype=np.uint8)
         img_scaled = ((image - min_val) / (max_val - min_val + 1e-8) * 255).astype(np.uint8)


    if np.all(img_scaled == img_scaled.flat[0]): # Check if all elements are the same
         print("Warning: Input image is constant for Otsu. Returning zero map.")
         return np.zeros_like(img_scaled, dtype=np.uint8)

    try:
        thresh = threshold_otsu(img_scaled)
        return (img_scaled > thresh).astype(np.uint8)
    except ValueError as e:
         print(f"Error during Otsu thresholding: {e}. Returning zero map.")
         return np.zeros_like(img_scaled, dtype=np.uint8)
    except Exception as e:
         print(f"An unexpected error occurred during Otsu thresholding: {e}. Returning zero map.")
         return np.zeros_like(img_scaled, dtype=np.uint8)


def apply_threshold(
    image: np.ndarray, # Renamed from error_image to image as it can be any image now
    threshold_type: str = 'fixed',
    fixed_threshold: float = 0.5
) -> np.ndarray:
    """
    Apply thresholding to an image to extract a binary map.
    Assumes input image is in the range [0, 1] for 'fixed' thresholding,
    but handles scaling for 'otsu'.

    Parameters
    ----------
    image : np.ndarray
        The input image (2D numpy array).
    threshold_type : str, optional
        'fixed' or 'otsu'. Default is 'fixed'.
    fixed_threshold : float, optional
        Fixed threshold value for 'fixed' type. Should be between 0.0 and 1.0.
        Default is 0.5.

    Returns
    -------
    np.ndarray
        A binary image (0 or 1, uint8 dtype).

    Raises
    ------
    ValueError
        If invalid threshold_type.
    ImportError
        If 'otsu' chosen but scikit-image not available.
    """
    if image.size == 0:
        return np.array([], dtype=np.uint8).reshape(image.shape)

    if threshold_type == 'fixed':
        # For fixed threshold, assume input is [0, 1] and threshold is also [0, 1]
        return simple_threshold(image, fixed_threshold)
    elif threshold_type == 'otsu':
        # Otsu handles internal scaling, but we should ensure skimage is available
        if not _skimage_available:
             raise ImportError("Otsu's method requires scikit-image.")
        return otsu_threshold(image)
    else:
        raise ValueError(f"Invalid threshold_type: {threshold_type}.")


def get_band_around_edges(binary_edge_map: np.ndarray, band_width: int) -> np.ndarray:
    """
    Creates a binary mask representing a band around detected edges.

    Parameters:
    -----------
    binary_edge_map : ndarray
        A binary image (0 or 1) where 1s are edges. Must be uint8 or boolean.
    band_width : int
        The width of the band in pixels.

    Returns:
    --------
    ndarray
        A binary mask (uint8) for the band.
    """
    if not _skimage_available:
        print("Warning: scikit-image not available for band creation. Returning zero mask.")
        return np.zeros_like(binary_edge_map, dtype=np.uint8)

    if binary_edge_map.size == 0 or np.sum(binary_edge_map) == 0:
         return np.zeros_like(binary_edge_map, dtype=np.uint8)

    # Ensure binary_edge_map is boolean for binary_dilation
    binary_edge_map_bool = binary_edge_map.astype(bool)

    selem = disk(band_width)
    band_mask = binary_dilation(binary_edge_map_bool, selem)
    return band_mask.astype(np.uint8)


def calculate_gradient_magnitude(image: np.ndarray) -> np.ndarray:
    """
    Calculates the magnitude of the gradient of an image using Gaussian derivatives.

    Parameters:
    -----------
    image : ndarray
        The input image (2D numpy array). Expected to be normalized [0, 1] or [0, 255].

    Returns:
    --------
    ndarray
        The gradient magnitude image (float32). Returns zero array for empty input.
    """
    if not _scipy_available:
        print("Warning: scipy not available for gradient calculation. Returning zero array.")
        return np.zeros_like(image, dtype=np.float32)

    if image.size == 0:
        return np.zeros_like(image, dtype=np.float32)

    # Ensure image is float for gradient calculation
    image_float = image.astype(np.float64)
    # Added smoothing before gradient calculation - sigma=0.5 is a common choice
    smoothed_image = gaussian_filter(image_float, sigma=0.5)
    grad_x = gaussian_filter(smoothed_image, sigma=0.5, order=(0, 1))
    grad_y = gaussian_filter(smoothed_image, sigma=0.5, order=(1, 0))

    gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
    return gradient_magnitude.astype(np.float32)


# The following function is currently used for the *initial* edge quality measure M(S)
# in the adaptive method. As we develop new edge-driven measures, this function
# might be modified or replaced by more specialized functions that operate on
# the *detected binary edge map* within a segment.
def calculate_edge_quality_measure(
    image_for_quality: np.ndarray,
    image_source_for_band: np.ndarray, # Image used to generate the binary map for the band
    band_threshold_type: str = 'otsu',
    fixed_band_threshold: float = 0.5,
    measure_type: str = 'gradient_magnitude_near_edges',
    band_width: int = 3,
    quantile: Optional[float] = None # Quantile to return instead of the mean
) -> float:
    """
    Calculates an edge quality measure M(S) for a segment based on a band around edges.
    The edges for the band are detected in 'image_source_for_band'.
    The measure is calculated on 'image_for_quality' within that band.

    Parameters:
    -----------
    image_for_quality : ndarray
        Image on which the measure is calculated (e.g., error map).
    image_source_for_band : ndarray
        Image used to generate the binary map for the band (e.g., original segment,
        or a composite error map).
    band_threshold_type : str, optional
        Thresholding type for `image_source_for_band` ('fixed', 'otsu'). Default 'otsu'.
    fixed_band_threshold : float, optional
        Fixed threshold for `band_threshold_type='fixed'`. Default 0.5.
    measure_type : str, optional
        Measure type: 'gradient_magnitude_near_edges', 'variance_near_edges',
        'mean_abs_error_near_edges'. Default 'gradient_magnitude_near_edges'.
        NOTE: 'gradient_magnitude_near_edges' requires SciPy.
    band_width : int, optional
        The width of the band in pixels around the edges. Default 3.
    quantile : Optional[float], optional
        If provided, returns the specified quantile (0.0-1.0) of the measure values
        within the band instead of the mean.

    Returns:
    --------
    float
        The calculated edge quality measure M(S). Returns 0.0 if not possible
        (e.g., no edges detected, libraries not available, empty band).

    Raises:
    -------
    ValueError
        If invalid measure_type or quantile out of range.
    ImportError
        If required libraries for measure_type or band creation are not available.
    """
    if image_for_quality.size == 0 or image_source_for_band.size == 0:
        print("DEBUG: Input image(s) empty for quality measure. Returning 0.0.")
        return 0.0

    if quantile is not None and not (0.0 <= quantile <= 1.0):
         raise ValueError(f"Quantile must be between 0.0 and 1.0, got {quantile}.")

    try:
        # Generate a binary map from the source image to define the band
        binary_edge_map_for_band = apply_threshold(
            image_source_for_band,
            threshold_type=band_threshold_type,
            fixed_threshold=fixed_band_threshold
        )
    except ImportError as e:
         print(f"Error generating binary map for band (ImportError): {e}. Returning 0.0.")
         return 0.0
    except ValueError as e:
        print(f"Error generating binary map for band (ValueError): {e}. Returning 0.0.")
        return 0.0
    except Exception as e:
        print(f"An unexpected error occurred generating binary map for band: {e}. Returning 0.0.")
        return 0.0


    if np.sum(binary_edge_map_for_band) == 0:
        return 0.0

    # Ensure get_band_around_edges is available
    if not _skimage_available:
        print(f"Error: scikit-image is required for get_band_around_edges for '{measure_type}'. Returning 0.0.")
        return 0.0

    band_mask = get_band_around_edges(binary_edge_map_for_band, band_width)

    if np.sum(band_mask) == 0:
        return 0.0

    measure_values_in_band = None

    if measure_type == 'gradient_magnitude_near_edges':
        if not _scipy_available:
             raise ImportError(f"SciPy is required for '{measure_type}'.")
        gradient_magnitude = calculate_gradient_magnitude(image_for_quality)
        if gradient_magnitude is None or gradient_magnitude.size == 0:
             print(f"Warning: Gradient magnitude calculation failed for '{measure_type}'. Returning 0.0.")
             return 0.0
        measure_values_in_band = gradient_magnitude[band_mask > 0]

    elif measure_type == 'variance_near_edges':
        # This measure type primarily uses scikit-image for band creation
        if not _skimage_available:
             raise ImportError(f"scikit-image is required for '{measure_type}' (for band creation).")
        measure_values_in_band = image_for_quality[band_mask > 0]
        if measure_values_in_band.size < 2:
             return 0.0
        variance_value = np.var(measure_values_in_band)
        return float(variance_value) # Return variance directly

    elif measure_type == 'mean_abs_error_near_edges':
        # This measure type primarily uses scikit-image for band creation
        if not _skimage_available:
             raise ImportError(f"scikit-image is required for '{measure_type}' (for band creation).")
        measure_values_in_band = np.abs(image_for_quality[band_mask > 0])

    else:
        raise ValueError(f"Invalid edge quality measure type: {measure_type}.")

    if measure_values_in_band is None or measure_values_in_band.size == 0:
         return 0.0

    if quantile is not None:
        # Calculate the specified quantile
        quality_measure = np.quantile(measure_values_in_band, quantile)
    elif measure_type != 'variance_near_edges': # Variance is handled above
        # Calculate the mean if no quantile is specified (default behavior)
        quality_measure = np.mean(measure_values_in_band)
    else:
        # This case should not be reached if variance is handled above, but as a safeguard
        print(f"DEBUG: Warning: Unhandled case for measure type '{measure_type}' and quantile status. Returning 0.0.")
        return 0.0


    return float(quality_measure)

def calculate_edge_density(binary_edge_map: np.ndarray) -> float:
    """
    Calculates the density of edge pixels in a binary edge map.

    Parameters:
    -----------
    binary_edge_map : ndarray
        A binary image (0 or 1) where 1s are edge pixels.

    Returns:
    --------
    float
        The proportion of edge pixels (between 0.0 and 1.0). Returns 0.0 for empty map.
    """
    if binary_edge_map.size == 0:
        return 0.0
    return np.sum(binary_edge_map) / binary_edge_map.size
