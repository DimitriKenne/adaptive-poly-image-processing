import numpy as np
from typing import Optional, Callable, Tuple, Union, List # Import List

def graded_lexicographic_multi_indices(total_terms):
        """Generate multi-indices in graded lexicographic order"""
        indices = []
        for total_deg in range(total_terms):
            for deg_x in range(total_deg + 1):
                deg_y = total_deg - deg_x
                indices.append((deg_x, deg_y))
        return indices[:total_terms]


def gen_vanderm2d(X: np.ndarray,
                  col: Optional[int] = None,
                  poly_basis: int = 1,
                  rectangle: Optional[Union[Tuple[float, float, float, float], List[float]]] = None # Rectangle parameter
                 ) -> Tuple[np.ndarray, Callable]:
    """
    Generate the Vandermonde matrix V at the points in X.

    Parameters
    ----------
    X : ndarray
        Input points with shape (n, 2), each row a 2D complex vector.
        These points are assumed to be within the 'rectangle' domain if provided.
    col : int, optional
        Number of columns in Vandermonde matrix. If None, defaults to the number of points.
    poly_basis : int, optional
        1: Shifted-normalized monomials
        2: Normal monomial basis (default)
        3: Chebyshev polynomial products
    rectangle : tuple or list, optional
        The bounds of the rectangular domain [xmin, ymin, xmax, ymax] that the points X belong to.
        Required for basis functions that scale points (e.g., shifted monomials, Chebyshev)
        to their standard domain.

    Returns
    -------
    V : ndarray
        Vandermonde matrix V[i,j] = e_j(a_i)
    p : function
        Polynomial basis function
    """

    X = np.asarray(X)
    n_points = len(X)

    # Check if X has 2 dimensions
    if X.ndim != 2:
        raise ValueError("Invalid: X is not a 2D array.")

    # Check if the second dimension is 2
    if X.shape[1] != 2:
        raise ValueError("Invalid: Each row of X must have exactly 2 elements.")


    if col is None:
        col = n_points

    if rectangle is None and poly_basis in [1, 3]:
         # Rectangle is required for shifted monomials and Chebyshev bases
         raise ValueError(f"Rectangle bounds must be provided for poly_basis {poly_basis}.")
    elif rectangle is not None and len(rectangle) != 4:
         raise ValueError("Rectangle must be a tuple or list of 4 elements: [xmin, ymin, xmax, ymax].")


    multi_indices = graded_lexicographic_multi_indices(col)

    # Calculate Zc and Zr from the rectangle for shifted monomials
    # These are calculated here once for the rectangle and used by the basis function
    if rectangle is not None:
        xmin, ymin, xmax, ymax = rectangle
        rect_center = np.array([(xmin + xmax) / 2.0, (ymin + ymax) / 2.0])
        rect_dimensions = np.array([xmax - xmin, ymax - ymin])
        # Zr maps the max dimension / 2 to 1
        rect_Zr = np.max(rect_dimensions) / 2.0
        # Handle zero dimension case
        if rect_Zr < 1e-9:
            rect_Zr = 1.0 # Default to 1 if dimensions are zero

    def shifted_monomial_basis(x, I= multi_indices, rect=None): # Added rect parameter
        """Shifted-normalized monomial basis: e_j(z) = prod(((z - Zc) / Zr)^{a_1, a_2})"""
        # Ensure x is a numpy array
        x = np.asarray(x)

        if rect is None:
             raise ValueError("Rectangle bounds must be provided for shifted monomial basis.")

        xmin, ymin, xmax, ymax = rect
        # Calculate Zc and Zr internally based on the rectangle
        Zc_arr = np.array([(xmin + xmax) / 2.0, (ymin + ymax) / 2.0])
        dimensions = np.array([xmax - xmin, ymax - ymin])
        Zr_val = np.max(dimensions) / 2.0
        # Handle zero dimension case
        if Zr_val < 1e-9:
            Zr_val = 1.0 # Default to 1 if dimensions are zero


        # Apply the shift and scale to map from rectangle domain to basis domain ([-1, 1] if Zr is half max dim)
        # Ensure Zr_val is not zero before division
        if Zr_val < 1e-9:
            scaled_shifted_x = x - Zc_arr # Just shift if Zr is zero (degenerate case)
        else:
            scaled_shifted_x = (x - Zc_arr) / Zr_val

        # Compute the basis function values
        basis_values = [np.prod(scaled_shifted_x ** np.array(idx)) for idx in multi_indices]
        return basis_values


    def monomial_basis(x, I= multi_indices):
        """Monomial basis: e_j(z) = z_1^{a_1}...z_n^{a_n}"""
        # Ensure x is a numpy array
        x = np.asarray(x)
        # This basis uses the input point x directly, defined over the domain of X
        return [x[0]**idx[0] * x[1]**idx[1] for idx in multi_indices]

    def chebyshev_basis(x, I=multi_indices, rect=None): # Added rect parameter
        """Chebyshev polynomial products, scaled to a specific rectangle."""
        # Ensure x is a numpy array
        x = np.asarray(x)

        if rect is None:
             raise ValueError("Rectangle bounds [xmin, ymin, xmax, ymax] must be provided for Chebyshev basis.")

        xmin, ymin, xmax, ymax = rect
        range_x = xmax - xmin
        range_y = ymax - ymin

        # Handle cases where the range is zero (e.g., a degenerate rectangle)
        # Add a small epsilon to avoid division by zero
        range_epsilon = 1e-9
        if range_x < range_epsilon:
            range_x = range_epsilon
        if range_y < range_epsilon:
            range_y = range_epsilon


        # Evaluate the 2D Chebyshev basis functions
        # The basis functions are products of 1D Chebyshev polynomials T_n(x) evaluated at scaled coordinates.
        # The scaled coordinates map the rectangle [xmin, xmax] x [ymin, ymax] to [-1, 1] x [-1, 1].
        basis_values = []
        for idx in multi_indices:
            # Scale the x and y coordinates of the point from the rectangle domain to [-1, 1]
            # Formula: scaled_coord = 2 * (coord - min_coord) / (max_coord - min_coord) - 1
            scaled_x_val = 2 * (x[0] - xmin) / range_x - 1
            scaled_y_val = 2 * (x[1] - ymin) / range_y - 1

            # Ensure scaled values are within [-1, 1] due to floating point arithmetic
            scaled_x_val = np.clip(scaled_x_val, -1.0, 1.0)
            scaled_y_val = np.clip(scaled_y_val, -1.0, 1.0)

            # Evaluate the product of 1D Chebyshev polynomials T_{alpha1}(scaled_x) * T_{alpha2}(scaled_y)
            # T_n(x) = cos(n * arccos(x))
            basis_value = np.cos(idx[0] * np.arccos(scaled_x_val)) * np.cos(idx[1] * np.arccos(scaled_y_val))
            basis_values.append(basis_value)

        return basis_values


    poly_basis_funcs = {
        1: shifted_monomial_basis,
        2: monomial_basis,
        3: chebyshev_basis
    }

    # Select the basis function generator.
    # If a basis requires the rectangle for scaling, pass it.
    if poly_basis == 1:
         if rectangle is None:
              raise ValueError("Rectangle bounds must be provided for shifted monomial basis.")
         p = lambda x: shifted_monomial_basis(x, I=multi_indices, rect=rectangle)
    elif poly_basis == 3:
        if rectangle is None:
            raise ValueError("Rectangle bounds must be provided for Chebyshev basis.")
        p = lambda x: chebyshev_basis(x, I=multi_indices, rect=rectangle)
    else: # Monomial basis doesn't need rectangle for internal scaling
        p = poly_basis_funcs.get(poly_basis, shifted_monomial_basis)


    # Generate Vandermonde matrix V[i,j] = p_j(a_i)
    V = np.array([p(x) for x in X])

    return V, p

def gen_vanderdet2d(X, poly_basis=1, rectangle=None): # Removed Zc, Zr
    """Compute the determinant of the Vandermonde matrix"""
    # Pass the rectangle argument to gen_vanderm2d
    A = gen_vanderm2d(X, poly_basis=poly_basis, rectangle=rectangle)[0]
    return np.linalg.det(A)
