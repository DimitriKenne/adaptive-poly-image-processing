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
    -------
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
        raise ValueError(f"Invalid error measure type: {measure_type}. Supported types: 'mse', 'mae', 'rmse'.")
