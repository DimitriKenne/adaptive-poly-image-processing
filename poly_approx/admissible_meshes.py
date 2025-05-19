import numpy as np
# Removed import math as it's not directly used
from typing import Tuple, Optional

def chebyshev_lobatto_points(n: int, min_val: float = -1.0, max_val: float = 1.0) -> np.ndarray:
    """
    Generate n Chebyshev-Lobatto points in the 1D interval [min_val, max_val].

    Chebyshev-Lobatto points are the extrema of the Chebyshev polynomial of degree n-1,
    including the endpoints. They are given by cos(j*pi/(n-1)) for j = 0, 1, ..., n-1.

    Parameters
    ----------
    n : int
        The number of points to generate (n >= 1).
    min_val : float, optional
        The minimum value of the interval. Default is -1.0.
    max_val : float, optional
        The maximum value of the interval. Default is 1.0.

    Returns
    -------
    np.ndarray
        A 1D array of n Chebyshev-Lobatto points in the specified interval, sorted.

    Raises
    ------
    ValueError
        If n is less than 1.
    """
    if n < 1:
        raise ValueError("Number of points 'n' must be at least 1.")

    # Handle degenerate interval case
    if max_val <= min_val:
        print(f"Warning: Degenerate interval [{min_val}, {max_val}] for Chebyshev-Lobatto points. Returning {n} points at the midpoint.")
        return np.full(n, (min_val + max_val) / 2.0)


    # Generate points in [-1, 1] using the formula cos(j*pi/(n-1))
    # Handle n=1 case separately to avoid division by zero
    if n == 1:
        points_1d = np.array([0.0]) # Midpoint for n=1 in [-1, 1]
    else:
        j = np.arange(n)
        points_1d = np.cos(j * np.pi / (n - 1))

    # Scale and shift points to the desired interval [min_val, max_val]
    # The points are currently in [-1, 1]. We map this to [min_val, max_val].
    # Formula: scaled_point = min_val + (point - (-1)) * (max_val - min_val) / (1 - (-1))
    # scaled_point = min_val + (point + 1) * (max_val - min_val) / 2
    scaled_points_1d = min_val + (points_1d + 1) * (max_val - min_val) / 2.0

    # Sort the points in ascending order
    return np.sort(scaled_points_1d)


def generate_chebyshev_product_grid(
    num_points_x: int,
    num_points_y: int,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float
) -> np.ndarray:
    """
    Generate a 2D admissible mesh as a product of 1D Chebyshev-Lobatto points.

    The mesh is constructed by taking the Cartesian product of Chebyshev-Lobatto
    points in the x-interval [x_min, x_max] and the y-interval [y_min, y_max].
    The generated points are within the specified rectangle.

    Parameters
    ----------
    num_points_x : int
        The number of Chebyshev-Lobatto points in the x-direction (>= 1).
    num_points_y : int
        The number of Chebyshev-Lobatto points in the y-direction (>= 1).
    x_min : float
        The minimum value of the x-interval.
    x_max : float
        The maximum value of the x-interval.
    y_min : float
        The minimum value of the y-interval.
    y_max : float
        The maximum value of the y-interval.

    Returns
    -------
    np.ndarray
        A 2D array of points with shape (num_points_x * num_points_y, 2),
        representing the Chebyshev product grid within the specified rectangle.

    Raises
    ------
    ValueError
        If num_points_x or num_points_y is less than 1.
    """
    if num_points_x < 1 or num_points_y < 1:
        raise ValueError("Number of points in each direction must be at least 1.")

    # Generate 1D Chebyshev-Lobatto points for x and y intervals
    cheby_x = chebyshev_lobatto_points(num_points_x, x_min, x_max)
    cheby_y = chebyshev_lobatto_points(num_points_y, y_min, y_max)

    # Create the 2D product grid using meshgrid
    XX, YY = np.meshgrid(cheby_x, cheby_y)

    # Flatten the meshgrid into a list of 2D points
    admissible_mesh = np.vstack([XX.ravel(), YY.ravel()]).T

    return admissible_mesh


# Removed generate_uniform_grid function

def compute_admissible_mesh(
    deg: int,
    mesh_type: str = "cheb", # Default is still cheb
    m: int = 2,
    x_min: float = 0.0,
    x_max: float = 1.0,
    y_min: float = 0.0,
    y_max: float = 1.0
) -> np.ndarray:
    """
    Generate a Chebyshev admissible mesh for polynomial interpolation of a given degree
    on a rectangle [x_min, x_max] x [y_min, y_max].

    The generated mesh points are guaranteed to be within the specified rectangle.

    Generates a Chebyshev product grid using (m*deg + 1) Chebyshev-Lobatto points
    in each dimension.

    Parameters
    ----------
    deg : int
        The degree of the polynomial for which the mesh should be admissible.
    mesh_type : str, optional
        The type of admissible mesh to generate. Currently only "cheb" is supported.
        Default is "cheb".
    m : int, optional
        Parameter for Chebyshev mesh construction (m > 1). Used to determine
        the number of 1D Chebyshev points (m*deg + 1 points per dimension).
        Default is 2.
    x_min : float, optional
        The minimum value of the x-interval for the rectangle. Default is 0.0.
    x_max : float, optional
        The maximum value of the x-interval for the rectangle. Default is 1.0.
    y_min : float, optional
        The minimum value of the y-interval for the rectangle. Default is 0.0.
    y_max : float, optional
        The maximum value of the y-interval for the rectangle. Default is 1.0.

    Returns
    -------
    np.ndarray
        A 2D array of points representing the admissible mesh within the specified rectangle.

    Raises
    ------
    ValueError
        If deg is less than 0, mesh_type is not "cheb", or m is not > 1.
    """
    if deg < 0:
        raise ValueError("Polynomial degree 'deg' must be non-negative.")

    # Only support Chebyshev mesh type
    if mesh_type != "cheb":
        raise ValueError(f"Invalid mesh_type: {mesh_type}. Currently only 'cheb' is supported.")

    # Add check for degenerate rectangle bounds
    if x_min > x_max or y_min > y_max:
        raise ValueError(f"Invalid rectangle bounds: [{x_min}, {x_max}] x [{y_min}, {y_max}]. x_min must be <= x_max and y_min must be <= y_max.")


    if m <= 1:
         raise ValueError("Parameter 'm' for Chebyshev mesh must be > 1.")

    # For Chebyshev product grid (Lemma 2.1), use (m*deg + 1) points per dimension
    # based on Chebyshev-Lobatto points.
    num_points_per_dim = m * deg + 1

    # Ensure at least 1 point per dimension, especially for deg=0
    if num_points_per_dim < 1:
         num_points_per_dim = 1

    admissible_mesh = generate_chebyshev_product_grid(
        num_points_per_dim,
        num_points_per_dim,
        x_min, x_max, y_min, y_max
    )
    print(f"Generated {mesh_type} admissible mesh of degree {deg} (m={m}) with {len(admissible_mesh)} points.")

    return admissible_mesh
