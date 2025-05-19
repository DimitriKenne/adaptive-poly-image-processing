import numpy as np
from typing import Optional # Import Optional for type hints

def calculate_mse(error_map: np.ndarray) -> float:
    """
    Calculate the Mean Squared Error (MSE) of an error map.

    Parameters:
    -----------
    error_map : ndarray
        The input error map (2D numpy array).

    Returns:
    --------
    float
        The calculated MSE. Returns 0.0 for an empty error map.
    """
    if error_map.size == 0:
        return 0.0
    # Ensure the error map is treated as floating point for calculation
    return np.mean(error_map.astype(float)**2)

def calculate_mae(error_map: np.ndarray) -> float:
    """
    Calculate the Mean Absolute Error (MAE) of an error map.

    Parameters:
    -----------
    error_map : ndarray
        The input error map (2D numpy array).

    Returns:
    --------
    float
        The calculated MAE. Returns 0.0 for an empty error map.
    """
    if error_map.size == 0:
        return 0.0
    # Ensure the error map is treated as floating point for calculation
    return np.mean(np.abs(error_map.astype(float)))

def calculate_rmse(error_map: np.ndarray) -> float:
    """
    Calculate the Root Mean Squared Error (RMSE) of an error map.

    Parameters:
    -----------
    error_map : ndarray
        The input error map (2D numpy array).

    Returns:
    --------
    float
        The calculated RMSE. Returns 0.0 for an empty error map.
    """
    if error_map.size == 0:
        return 0.0
    # Calculate MSE first, then take the square root
    return np.sqrt(calculate_mse(error_map))

def normalize_error_image(error_map: np.ndarray, epsilon: float = 1e-8) -> np.ndarray:
    """
    Normalize an error map to the range [0, 1] for visualization.

    Applies min-max normalization. Handles cases where the error map is constant.

    Parameters:
    -----------
    error_map : ndarray
        The input error map (2D numpy array).
    epsilon : float, optional
        A small value added to the denominator to prevent division by zero
        in case of a constant error map. Default is 1e-8.

    Returns:
    --------
    np.ndarray
        The normalized error map in the range [0, 1]. Returns an array of zeros
        if the input error map is empty or constant.
    """
    if error_map.size == 0:
        return np.zeros_like(error_map, dtype=np.float32)

    min_val = np.min(error_map)
    max_val = np.max(error_map)

    # Handle the case where the error map is constant
    if max_val - min_val < epsilon:
        # If constant, return a zero map (indicating uniform error, which is lowest visually)
        return np.zeros_like(error_map, dtype=np.float32)
    else:
        # Apply min-max normalization
        normalized_error = (error_map - min_val) / (max_val - min_val + epsilon)
        # Clip to [0, 1] just in case of floating point inaccuracies
        return np.clip(normalized_error, 0, 1).astype(np.float32)


def calculate_error_measure(error_map: np.ndarray, measure_type: str = 'mse') -> float:
    """
    Calculate a specified error measure for an error map.

    Parameters:
    -----------
    error_map : ndarray
        The input error map (2D numpy array).
    measure_type : str, optional
        The type of error measure to calculate. Options:
        - 'mse': Mean Squared Error (default)
        - 'mae': Mean Absolute Error
        - 'rmse': Root Mean Squared Error

    Returns:
    --------
    float
        The calculated error measure.

    Raises:
    -------\
    ValueError
        If an invalid measure_type is provided.
    """
    if error_map.size == 0:
        return 0.0 # Return 0 for empty maps regardless of measure type

    if measure_type == 'mse':
        return calculate_mse(error_map)
    elif measure_type == 'mae':
        return calculate_mae(error_map)
    elif measure_type == 'rmse':
        return calculate_rmse(error_map)
    else:
        raise ValueError(f"Invalid error measure type: {measure_type}. Choose from 'mse', 'mae', 'rmse'.")

