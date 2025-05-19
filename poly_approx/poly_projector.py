import numpy as np
# Import Callable, Union, Optional, and Tuple from typing
from typing import Callable, Union, Optional, Tuple
# Corrected relative import for modules within the same package (poly_approx)
from .polynomial_bases import gen_vanderm2d # Only need gen_vanderm2d here
import numpy.typing as npt
from scipy.linalg import lstsq, qr # Explicitly import lstsq and qr

def right_division(B: npt.NDArray, A: npt.NDArray) -> npt.NDArray:
    """
    Solve the linear system XA = B using the least squares method.

    Parameters
    ----------
    B : npt.NDArray
        Right-hand side matrix.
    A : npt.NDArray
        Coefficient matrix.

    Returns
    -------
    npt.NDArray
        Solution matrix X such that XA = B.
    """
    try:
        # Solve the transposed system A.T @ X_T = B.T
        X_T, residuals, rank, s = lstsq(A.T, B.T)
        return X_T.T
    except np.linalg.LinAlgError as e:
        raise ValueError(f"Error in right division: {e}")

def evaluate_polynomial_from_coeffs(coeffs: npt.NDArray, basis_func_generator: Callable, eval_points: Union[npt.NDArray, list]) -> Union[float, npt.NDArray]:
    """
    Evaluate a polynomial at given points using its coefficients and basis function generator.

    Parameters
    ----------
    coeffs : npt.NDArray
        Array of polynomial coefficients.
    basis_func_generator : Callable
        A function that takes a 2D point and returns a list or array
        of basis function values at that point.
    eval_points : array-like
        A single 2D point or an array of 2D points at which to evaluate the polynomial.

    Returns
    -------
    float or npt.NDArray
        The polynomial's value at the evaluation point(s).
    """
    # Ensure eval_points is a 2D array.
    eval_points = np.atleast_2d(eval_points)
    if eval_points.shape[1] != 2:
        raise ValueError("Evaluation points must be 2D")

    # Generate the Vandermonde matrix for the evaluation points.
    V0 = np.array([basis_func_generator(point) for point in eval_points])

    # Evaluate the polynomial: P(x,y) = sum(coeffs[j] * basis_j(x,y))
    # This is a matrix multiplication: V0 @ coeffs
    L = V0 @ coeffs

    # Return a scalar if only one point was evaluated, otherwise return the array of values.
    return L[0] if L.size == 1 else L


def poly_projector2d(
    dim: int,
    nodes_set: npt.NDArray,
    func_values: npt.NDArray,
    poly_basis: int = 1,
    # Removed Zc and Zr parameters as they are now handled via the rectangle in gen_vanderm2d
    # Zc: Optional[npt.NDArray] = None,
    # Zr: Optional[float] = None,
    rectangle: Optional[Tuple[float, float, float, float]] = None # Rectangle parameter is still needed
) -> Callable:
    """
    Compute a polynomial interpolation (or least-squares fit) for 2D points
    using an orthogonalization method to find coefficients, and return a
    function to evaluate the polynomial.

    Parameters
    ----------
    dim : int
        Dimension of the polynomial space. Must be <= number of nodes.
        If equal to the number of nodes, an interpolation polynomial is computed.
        If less than the number of nodes, a least-squares fit is performed.
    nodes_set : npt.NDArray
        Array of interpolation points with shape (n, 2).
    func_values : npt.NDArray
        Function values at the interpolation points with shape (n,).
    poly_basis : int, optional
        Specifies the type of polynomial basis to use:
            1: Shifted-normalized monomials,
            2: Standard monomial basis (default),
            3: Chebyshev polynomial products.
    rectangle : tuple, optional
        The bounds of the rectangular domain [xmin, ymin, xmax, ymax] that the
        nodes and evaluation points belong to. Used for basis scaling in gen_vanderm2d.

    Returns
    -------
    Callable
        A function that evaluates the polynomial at given 2D points.

    Raises
    ------
    ValueError
        If input dimensions are incompatible or invalid.
    """
    # Convert inputs to arrays.
    nodes_set = np.asarray(nodes_set)
    func_values = np.asarray(func_values)

    # Validate input dimensions.
    if nodes_set.ndim != 2 or nodes_set.shape[1] != 2:
        raise ValueError("nodes_set must be a 2D array with shape (n, 2)")
    if nodes_set.shape[0] != func_values.shape[0]:
        raise ValueError("The number of nodes must match the number of function values")
    if dim > nodes_set.shape[0]:
        raise ValueError("`dim` must be less than or equal to the number of nodes in nodes_set")

    # Determine basis scaling parameters if rectangle is provided and Zc/Zr are not
    # Removed this block as Zc/Zr are no longer parameters of poly_projector2d
    # basis_Zc = Zc
    # basis_Zr = Zr
    # if rectangle is not None and (Zc is None or Zr is None):
    #     xmin, ymin, xmax, ymax = rectangle
    #     basis_Zc = np.array([(xmin + xmax) / 2.0, (ymin + ymax) / 2.0])
    #     basis_Zr = max(xmax - xmin, ymax - ymin) / 2.0 # Use max dimension for scaling


    # Generate the Vandermonde matrix and the polynomial basis function.
    # `p` is a function that takes a 2D point and returns the basis values at that point.
    # Pass only the rectangle to gen_vanderm2d for scaling
    V, p = gen_vanderm2d(
        nodes_set,
        col=dim,
        poly_basis=poly_basis,
        # Removed Zc=basis_Zc, Zr=basis_Zr from the call
        rectangle=rectangle # Pass the rectangle
    )

    # --- Compute the polynomial coefficients using the orthogonalization method ---
    # This involves two steps of QR factorization and then solving for coefficients.

    # First QR factorization: V = Q1 @ R1
    Q1, R1 = qr(V, mode='economic')

    # Compute V1 = V @ inv(R1)
    # Use pinv for numerical stability in case R1 is close to singular
    V1 = V @ np.linalg.pinv(R1)

    # Second QR factorization on V1: V1 = Q2 @ R2
    # Q here corresponds to Q2 in the derivation
    Q, R2 = qr(V1, mode='economic')

    # Compute the coefficients using the formula derived from the orthogonalization steps:
    # coeffs = (R2 @ R1)^-1 @ Q.conj().T @ func_values
    # Using right_division to compute (R2 @ R1)^-1
    W = right_division(np.eye(R1.shape[0]), R2 @ R1)
    coeffs = W @ Q.conj().T @ func_values


    # Return a callable function that evaluates the polynomial using the computed coefficients
    # and the basis function generator.
    def polynomial_evaluation_function(eval_set: Union[npt.NDArray, list]) -> Union[float, npt.NDArray]:
        """
        Evaluate the computed polynomial at given points using the pre-calculated coefficients.
        """
        # Use the helper function to perform the evaluation
        return evaluate_polynomial_from_coeffs(coeffs, p, eval_set)

    return polynomial_evaluation_function
