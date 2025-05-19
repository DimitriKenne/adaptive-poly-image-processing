from scipy.linalg import lu, qr
# Corrected relative import for modules within the same package (poly_approx)
from .polynomial_bases import gen_vanderm2d, graded_lexicographic_multi_indices
# Import the admissible_mesh function
from .admissible_meshes import compute_admissible_mesh
import matplotlib.pyplot as plt # Keep import for potential visualization/debugging functions (if any)
import numpy as np
import math
from typing import Tuple, Optional, Union, List # Import necessary types


def discrete_leja_pts2d(deg: int, A: np.ndarray, poly_basis: int = 1) -> np.ndarray:
    """
    Generate discrete Leja points from an admissible mesh A.

    Selects points using LU factorization after two orthogonalization steps
    of the Vandermonde matrix.

    Parameters
    ----------
    deg : int
        The degree of interpolation.
    A : (...,2) array
        An admissible mesh (AM) of degree deg. Leja points are a subset of this mesh.
    poly_basis : int, optional
        Polynomial basis to use for the Vandermonde matrix (1: Shifted-normalized monomials,
        2: Monomial, 3: Chebyshev). Default is 1.

    Returns
    -------
    X : (math.comb(2+deg, deg), 2) ndarray
        Selected Leja points from the mesh A.
    """
    N = math.comb(2 + deg, deg) # Number of points required for degree deg
    M = len(A) # Number of points in the admissible mesh

    if M < N:
        raise ValueError(f"The mesh A must contain at least {N} points for degree {deg}.")

    A = np.asarray(A)

    # Generate the Vandermonde matrix for the entire mesh
    # Pass the bounding box of the mesh A as the rectangle for basis scaling
    xmin, ymin = np.min(A, axis=0)
    xmax, ymax = np.max(A, axis=0)
    mesh_rectangle = (xmin, ymin, xmax, ymax)

    V, _ = gen_vanderm2d(A, col=N, poly_basis=poly_basis, rectangle=mesh_rectangle) # Removed Zc, Zr


    # Orthogonalize the Vandermonde matrix twice using QR factorization
    Q1, R1 = qr(V, mode='economic')
    V1 = V @ np.linalg.pinv(R1) # Using pinv for robustness
    Q, R2 = qr(V1, mode='economic')

    # Perform LU decomposition with pivoting on the orthogonalized matrix Q
    P_matrix, L, U = lu(Q)

    # Extract Leja indices using the transpose of the permutation matrix
    I = P_matrix.T @ np.arange(M)
    leja_indices = I[:N].astype(int) # Take the first N indices

    # Select the Leja points from the original mesh A
    X = A[leja_indices]

    return X


def approx_fekete_pts2d(deg: int, A: np.ndarray, poly_basis: int = 1) -> np.ndarray:
    """
    Generate approximate Fekete Points (AFP) from an admissible mesh A.

    Selects points using QR factorization with pivoting after two
    orthogonalization steps of the Vandermonde matrix.

    Parameters
    ----------
    deg : int
        The degree of interpolation.
    A : array
        An admissible mesh (AM) of degree d, from which the AFP are extracted.
    poly_basis : int, optional
        Polynomial basis to use for the Vandermonde matrix (1: Shifted-normalized monomials,
        2: Monomial, 3: Chebyshev). Default is 1.

    Returns
    -------
    X : (math.comb(2+deg, deg), 2) array
        The deg-th set of AFP from the mesh A.
    """
    N = math.comb(2+deg, deg) # Number of points required for degree deg
    M = len(A) # Number of points in the admissible mesh

    if M < N:
        raise ValueError(f"The mesh A must contain at least {N} points for degree {deg}.")

    A = np.asarray(A)

    # Compute the Vandermonde matrix
    # Pass the bounding box of the mesh A as the rectangle for basis scaling
    xmin, ymin = np.min(A, axis=0)
    xmax, ymax = np.max(A, axis=0)
    mesh_rectangle = (xmin, ymin, xmax, ymax)

    V, _ = gen_vanderm2d(A, col=N, poly_basis=poly_basis, rectangle=mesh_rectangle) # Removed Zc, Zr

    # Orthogonalize the Vandermonde matrix twice using QR factorization
    Q1, R1 = qr(V, mode='economic')
    V1 = V @ np.linalg.pinv(R1) # Using pinv for robustness
    Q2, R2 = qr(V1, mode='economic')

    # QR factorization with pivoting on the transpose of Q2
    W = np.transpose(Q2)
    Q, R, P = qr(W, pivoting=True)

    # The permutation vector P gives the indices of the approximate Fekete points.
    afp_indices = P[:N].astype(int) # Take the first N indices

    # Extraction of AFP from the original mesh A
    X = A[afp_indices]

    return X

#-------------------------------------------------------------------------------------------

def padua_points_theoretical(deg: int, rectangle: Union[Tuple[float, float, float, float], List[float]]) -> np.ndarray:
    """
    Compute unique theoretical Padua points scaled to a specific rectangular region.

    Generates base points on [-1,1]^2 using a specific formula and scales them.

    Parameters
    ----------
    deg : int
        Degree of Padua points.
    rectangle : tuple or list
        The bounds of the rectangular domain [xmin, ymin, xmax, ymax] to scale the points to.

    Returns
    -------
    X : ndarray
        Unique Padua points scaled and shifted to the specified rectangle.

    Raises
    ------
    ValueError
        If the rectangle format is invalid.
    """
    if len(rectangle) != 4:
         raise ValueError("Rectangle must be a tuple or list of 4 elements: [xmin, ymin, xmax, ymax].")

    xmin, ymin, xmax, ymax = rectangle
    center_x = (xmin + xmax) / 2.0
    center_y = (ymin + ymax) / 2.0

    # Handle degree 0 case separately
    if deg == 0:
        return np.array([[center_x, center_y]])

    # Compute base Padua points on [-1,1]^2 using the original user formula
    # Use the first (deg+1)(deg+2)/2 multi-indices in graded lexicographic order.
    N_expected = int((deg+1)*(deg+2)/2)
    indices = graded_lexicographic_multi_indices(N_expected)

    all_points = []
    for i in indices:
        m, j = i[0], i[1]
        # Apply the specific formula from the user's original code
        if deg == 0: # Safeguard, though handled above
             continue
        x_coord = (-1)**m * np.cos(j * deg * np.pi / (deg + 1))
        y_coord = (-1)**j * np.cos(m * (deg + 1) * np.pi / deg)
        all_points.append([x_coord, y_coord])

    all_points = np.array(all_points)

    # Remove duplicate points using rounding for floating point comparison
    if all_points.shape[0] > 0:
        unique_points = np.unique(np.round(all_points, decimals=8), axis=0)
    else:
        unique_points = np.array([])

    # Scale and shift unique points from [-1, 1] to the specified rectangle
    if unique_points.shape[0] > 0:
        X_scaled = unique_points.copy()
        X_scaled[:, 0] = xmin + (unique_points[:, 0] + 1) * (xmax - xmin) / 2.0
        X_scaled[:, 1] = ymin + (unique_points[:, 1] + 1) * (ymax - ymin) / 2.0
    else:
        X_scaled = np.array([])

    return X_scaled


def extremal_points(
    deg: int,
    method: str = 'leja',
    rectangle: Union[Tuple[float, float, float, float], List[float]] = (0.0, 0.0, 1.0, 1.0), # Rectangle parameter
    admissible_mesh_type: str = 'cheb', # Parameter for mesh generation
    m_cheb: int = 2, # Parameter for Chebyshev mesh
    poly_basis: int = 1 # Parameter for basis in Leja/Fekete (passed to gen_vanderm2d)
) -> np.ndarray:
    """
    Generate extremal points (Leja, Fekete, Padua, or full mesh) for polynomial
    interpolation within a specified rectangular domain.

    Generates an admissible mesh internally for mesh-based methods if needed.

    Parameters
    ----------
    deg : int
        The degree of interpolation.
    method : str, optional
        Method to generate points: 'leja', 'fekete', 'padua', 'full_mesh'.
        Default is 'leja'.
    rectangle : tuple or list, optional
        Bounds of the rectangular domain [xmin, ymin, xmax, ymax].
        Default is (0.0, 0.0, 1.0, 1.0).
    admissible_mesh_type : str, optional
        Type of admissible mesh ('cheb' or 'uni') for mesh-based methods. Default is 'cheb'.
    m_cheb : int, optional
        Parameter 'm' for Chebyshev mesh construction (m > 1). Default is 2.
    poly_basis : int, optional
        Polynomial basis to use for Vandermonde matrix in Leja/Fekete selection. Default is 1.

    Returns
    -------
    X : (N, 2) ndarray
        Selected extremal points within the specified rectangle.

    Raises
    ------
    ValueError
        If method is unknown or rectangle format is invalid.
    """
    if len(rectangle) != 4:
         raise ValueError("Rectangle must be a tuple or list of 4 elements: [xmin, ymin, xmax, ymax].")

    xmin, ymin, xmax, ymax = rectangle

    if method in ['leja', 'fekete', 'full_mesh']:
        # Generate Admissible Mesh for the specified rectangle
        print(f"Generating admissible mesh ('{admissible_mesh_type}') for rectangle {rectangle}...")
        try:
            admissible_mesh_for_nodes = compute_admissible_mesh(
                deg=deg,
                mesh_type=admissible_mesh_type,
                m=m_cheb,
                x_min=xmin,
                x_max=xmax,
                y_min=ymin,
                y_max=ymax
            )
            print(f"Generated admissible mesh with {len(admissible_mesh_for_nodes)} points.")
        except Exception as e:
             print(f"Error generating admissible mesh: {e}. Returning empty array.")
             return np.array([])

        # Select points from the generated mesh
        if method == 'leja':
            points = discrete_leja_pts2d(deg, admissible_mesh_for_nodes, poly_basis=poly_basis) # Removed Zc, Zr
        elif method == 'fekete':
            points = approx_fekete_pts2d(deg, admissible_mesh_for_nodes, poly_basis=poly_basis) # Removed Zc, Zr
        elif method == 'full_mesh':
             points = admissible_mesh_for_nodes

    elif method == 'padua':
        # Generate Theoretical Padua Points scaled to the rectangle
        print(f"Generating theoretical Padua points for rectangle {rectangle}...")
        points = padua_points_theoretical(deg, rectangle=rectangle)
        print(f"Generated {len(points)} points using 'padua'.")

    else:
         raise ValueError(f"Unknown method: {method}. Choose from 'leja', 'fekete', 'padua', 'full_mesh'.")

    return points
