"""
Implementation based on
"Fixed Point Algorithms for Estimating Power Means of Positive Definite Matrices" by M Congedo, A Barachant, and E K Koopaei (2017)
"Riemannian geometry for EEG-based brain-computer interfaces; a primer and a review" by M Congedo and A Barachant and R Bhatia (2017)
"""

import numpy as np
from scipy.linalg import sqrtm, inv, fractional_matrix_power
from sklearn.covariance import ledoit_wolf


def is_spd(matrix):
    """Check if a matrix is symmetric positive definite (SPD)."""
    return np.allclose(matrix, matrix.T) and np.all(np.linalg.eigvals(matrix) > 0)


def matrix_power(M, p):
    """Compute matrix power """
    assert np.allclose(M, M.T), "Matrix is not symmetric"
    return fractional_matrix_power(M, p)


def power_mean(covariances, p, weights=None, tol=1e-12, max_iter=100, eps=1e-11, reg=1e-10):
    """
    Compute power mean of SPD matrices using MPM algorithm.
    
    Parameters:
    covariances (list): List of SPD matrices (N x N)
    p (float): Power parameter in (-1, 1)\{0}
    weights (np.ndarray): Optional weights (default: uniform)
    tol (float): Convergence tolerance
    max_iter (int): Maximum iterations
    eps (float): Small value to check for small eigenvalues
    reg (float): matrix regularization parameter
    
    Returns:
    np.ndarray: Power mean matrix
    """
    assert p != 0, "Power parameter cannot be zero"
    assert -1 < p < 1, "Power parameter must be in (-1, 1)"
    if weights is not None:
        assert np.sum(weights) == 1, "Weights must sum to 1"
    for mat in covariances:
        assert is_spd(mat), "Input matrices must be SPD"
    
    # check for small eigenvalues
    eigv = np.array([np.linalg.eigvals(C) for C in covariances])
    
    min_ev = np.min(eigv, axis=1)
    if np.any(min_ev < eps):
        print("WARNING: Small eigenvalues detected in input matrices. Applying regularization...")
        covariances = [C + reg * np.eye(C.shape[0]) for C in covariances]

    if p < 0:
        # regularized inverse
        covariances = [inv(C) for C in covariances]
    
    K = len(covariances)
    N = covariances[0].shape[0]
    weights = np.ones(K)/K if weights is None else weights
    
    # Initialization using commuting case solution
    p_abs = abs(p)
    powered_covs = [matrix_power(C, p_abs) for C in covariances]
    G_initial = sum(w * C for w, C in zip(weights, powered_covs))
    G_initial = matrix_power(G_initial, 1/p_abs)
    
    # Initialize X as inverse sqrt of G_initial
    X = sqrtm(G_initial) if p > 0 else inv(sqrtm(G_initial))
    
    # Heuristic phi value from paper, phi must be in [0.25, 0.5]
    phi = 0.375 / p_abs
    
    for _ in range(max_iter):
        # Compute H
        H = np.zeros((N, N))
        for C, w in zip(covariances, weights):
            term = X @ C @ X.T
            H += w * matrix_power(term, p_abs)
            if np.any(np.isnan(matrix_power(term, p_abs))):
                print(np.linalg.eigh(term)[0])
        
        # Check convergence
        conv = np.linalg.norm(H - np.eye(N)) / np.sqrt(N)
        if conv < tol:
            break
        
        # Update X using H^{-phi}
        H_power = matrix_power(H, -phi)
        X = H_power @ X
        
    # Reconstruct P from X
    P = inv(X.T @ X) if p > 0 else X.T @ X

    return 0.5 * (P + P.T)  # Ensure symmetry


def riemann_mean(covariances, weights=None, tol=1e-8, max_iter=100, eps=1e-10, reg=1e-9):
    """
    Compute Riemannian geometric mean using midpoint of p=±0.01 power means.
    
    Parameters:
    covariances (list): List of SPD matrices (N x N)
    weights (np.ndarray): Optional weights
    tol (float): Convergence tolerance
    max_iter (int): Maximum iterations
    eps (float): Small value to check for small eigenvalues
    reg (float): matrix regularization parameter

    Returns:
    np.ndarray: Estimated geometric mean matrix
    """
    # Compute power means at p=±0.01
    G_p = power_mean(covariances, 0.01, weights, tol, max_iter, eps, reg)
    G_n = power_mean(covariances, -0.01, weights, tol, max_iter, eps, reg)
    
    # Compute geodesic midpoint (G_p #_{0.5} G_n)
    sqrt_Gp = sqrtm(G_p)
    inv_sqrt_Gp = inv(sqrt_Gp)
    mid_operator = sqrtm(inv_sqrt_Gp @ G_n @ inv_sqrt_Gp)
    midpoint = sqrt_Gp @ mid_operator @ sqrt_Gp
    
    return 0.5 * (midpoint + midpoint.T)  # Ensure symmetry


def riemann_distance(C1, C2, reg=1e-10):
    """Compute Riemannian distance between two SPD matrices"""

    # Check if the input matrices are SPD
    if not is_spd(C1):
        print("WARNING: C1 is not SPD. Applying regularization...")
        C1 = C1 + reg * np.eye(C1.shape[0])
    if not is_spd(C2):
        print("WARNING: C2 is not SPD. Applying regularization...")
        C2 = C2 + reg * np.eye(C2.shape[0])

    # Compute similarity transformation
    inv_sqrt_C1 = np.linalg.inv(sqrtm(C1))
    M = inv_sqrt_C1 @ C2 @ inv_sqrt_C1
    
    eigvals, eigvecs = np.linalg.eigh(M)
    eigvals = np.maximum(eigvals, reg)
    dist = np.sqrt(np.sum(np.log(eigvals)**2))
    return dist


def riemann_geodesic(C1, C2, t=0.5):
    """Compute a point along the geodesic path at t between two SPD matrices"""
    C1_sqrt = sqrtm(C1)
    C1_inv_sqrt = inv(C1_sqrt)
    M = C1_inv_sqrt @ C2 @ C1_inv_sqrt
    M = sqrtm(M) if t == 0.5 else matrix_power(M, t)
    return C1_sqrt @ M @ C1_sqrt


class RiemannMDM:
        
        """
        RiemanMDM: A Minimum Distance to Mean classifier based on the Riemannian manifold of SPD matrices.
        This class implements a classifier that operates on covariance matrices derived 
        from multi-channel time-series data. It uses Riemannian geometry to compute 
        class means and classify new samples based on their Riemannian distance to these means.
        """

        def __init__(self):
            """ Initializes an instance of the class. """
            self.classes = None
            self.means = None

        def store(self, file_path: str) -> None:
            """
            Save the means to a file.

            Parameters:
                file_path (str): Path to the file where the means will be saved.
            """
            np.save(file_path, self.means)

        def load(self, file_path: str) -> None:
            """
            Load the means from a file.

            Parameters:
                file_path (str): Path to the file from which the means will be loaded.
            """
            self.means = np.load(file_path, allow_pickle=True).item()
            self.classes = list(self.means.keys())


        def covariance(self, x: np.ndarray) -> np.ndarray:
            """
            Computes the covariance matrix of the input data.

            Parameters:
                x (array-like of shape (n_channels, n_timesteps), dtype float):
                    The input data, where each sample is a multi-channel time-series data matrix.

            Returns:
                ndarray: Covariance matrix of shape (n_channels, n_channels).
            """
            # Remove the mean from each channel
            x -= np.mean(x, axis=1, keepdims=True)
            # Normalize the signal for numerical stability
            x /= np.std(x, axis=(0,1), keepdims=True)
            # Compute the regularized covariance matrix 
            cov, shrinkage = ledoit_wolf(x.T, assume_centered=True)
            return cov

        def fit(self, X: np.ndarray, y: np.ndarray) -> object:
            """
            Fits the Riemannian means of covariance matrices for each class.
            This method computes the covariance over the channel dimension.

            Parameters:
                X (array-like of shape (n_samples, n_channels, n_timesteps), dtype float):
                    The input data, where each sample is a multi-channel time-series data matrix.
                y (array-like of shape (n_samples,), dtype int):
                    The class labels corresponding to each sample in X.

            Returns:
                self : object
                Returns the instance of the class with the computed Riemannian means stored.
            """
            self.classes = np.unique(y)
            self.means = {}
            for c in self.classes:
                Xc = X[y == c]
                Xc = [np.cov(x) for x in Xc]
                try:
                    self.means[c] = riemann_mean(Xc, reg=1e-6)
                except Exception as e:
                    print(f"Error computing mean for class {c}: {e}")
                    self.means[c] = np.eye(X.shape[1])
            return self
        
        def predict(self, X: list) -> tuple:
            """
            Predict the class of time-series signals.

            This method takes a list of time-series signal matrices, computes their covariance,
            and predicts the class for each signal based on the Riemannian distance 
            to the class means. A numerically stable softmax is applied to the distances 
            for a normalized similarity distribution.

            Parameters:
                X (array-like of shape (batch, n_channels, n_timesteps)): 
                    A list of EEG signal matrices, where each matrix represents a single trial.

            Returns:
                tuple (predicted_labels, distance_probs):
                    - ndarray: An array of predicted class labels. shape (batch,)
                    - ndarray: An array of distance-based probabilities for each class. shape (batch, n_classes)
            """
            y_pred = np.empty(len(X), dtype=int)
            for i, x in enumerate(X):
                x = np.cov(x)
                # so we add a small regularization term to the covariance matrix
                dists = [riemann_distance(x, self.means[c], reg=1e-9) for c in self.classes]
                dists = np.array(dists)

                y_pred[i] = self.classes[np.argmin(dists)]

                # convert distances to "probabilities" numerically stable softmax
                prob = np.exp(-dists - np.max(-dists))
                prob = prob / np.sum(prob)

            return np.array(y_pred), prob


