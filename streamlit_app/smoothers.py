"""
Smoothing Functions for EMD Local Median Approach

Contains three smoothing methods:
1. EPA Local Median - Epanechnikov kernel-weighted median (robust)
2. Median - Unweighted median within bandwidth window
3. Average - Nadaraya-Watson kernel mean
"""

import numpy as np
from typing import Tuple, List


def epanechnikov_kernel(u: np.ndarray) -> np.ndarray:
    """
    Epanechnikov kernel: K(u) = 0.75 * (1 - u^2) for |u| <= 1

    Optimal kernel in terms of mean integrated squared error (MISE).
    Compact support makes computation efficient.
    """
    weights = np.zeros_like(u, dtype=float)
    mask = np.abs(u) <= 1
    weights[mask] = 0.75 * (1 - u[mask]**2)
    return weights


def weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    """
    Compute weighted median using cumulative weight method.

    The weighted median minimizes: argmin_m SUM |y_i - m| * w_i

    Algorithm:
    1. Sort values and corresponding weights
    2. Compute cumulative normalized weights
    3. Find first index where cumulative weight >= 0.5
    4. Handle edge case at exactly 0.5 with interpolation
    """
    if len(values) == 0:
        return np.nan
    if np.sum(weights) == 0:
        # Fallback to unweighted median when weights sum to zero
        return float(np.median(values))

    # Sort by values
    sorted_indices = np.argsort(values)
    sorted_values = values[sorted_indices]
    sorted_weights = weights[sorted_indices]

    # Normalize weights
    total_weight = np.sum(sorted_weights)
    cumulative_weights = np.cumsum(sorted_weights) / total_weight

    # Find median position
    idx = np.searchsorted(cumulative_weights, 0.5)

    if idx >= len(sorted_values):
        return sorted_values[-1]
    elif idx == 0:
        return sorted_values[0]
    elif np.isclose(cumulative_weights[idx - 1], 0.5):
        # Exact 0.5 - interpolate between adjacent values
        return 0.5 * (sorted_values[idx - 1] + sorted_values[idx])
    else:
        return sorted_values[idx]


def unweighted_median_window(values: np.ndarray, in_window: np.ndarray) -> float:
    """
    Compute unweighted median of values within the bandwidth window.

    Unlike weighted median, all points in window have equal influence.
    """
    window_values = values[in_window]
    if len(window_values) == 0:
        return np.nan
    return np.median(window_values)


def epa_local_median(t: np.ndarray, y: np.ndarray, bandwidth: float) -> np.ndarray:
    """
    EPA Kernel-Weighted Local Median Smoother.

    For each point t_i, compute:
        y_smooth[i] = argmin_m SUM |y_j - m| * K((t_j - t_i) / h)

    where K is the Epanechnikov kernel.

    This is ROBUST to outliers because the median ignores outlier magnitudes.

    Parameters
    ----------
    t : array
        Time points (x-axis)
    y : array
        Signal values
    bandwidth : float
        Smoothing bandwidth h (kernel window width)

    Returns
    -------
    y_smooth : array
        Smoothed signal using weighted median
    """
    n = len(t)
    y_smooth = np.zeros(n)

    for i in range(n):
        u = (t - t[i]) / bandwidth
        weights = epanechnikov_kernel(u)
        y_smooth[i] = weighted_median(y, weights)

    return y_smooth


def median_smoother(t: np.ndarray, y: np.ndarray, bandwidth: float) -> np.ndarray:
    """
    Unweighted Median Smoother (within bandwidth window).

    For each point t_i, compute the median of all y_j where |t_j - t_i| <= h.
    All points in window contribute equally.

    This is also robust but doesn't use kernel weighting.

    Parameters
    ----------
    t : array
        Time points (x-axis)
    y : array
        Signal values
    bandwidth : float
        Window half-width

    Returns
    -------
    y_smooth : array
        Smoothed signal using unweighted median
    """
    n = len(t)
    y_smooth = np.zeros(n)

    for i in range(n):
        # Find points within bandwidth
        in_window = np.abs(t - t[i]) <= bandwidth
        y_smooth[i] = unweighted_median_window(y, in_window)

    return y_smooth


def average_smoother(t: np.ndarray, y: np.ndarray, bandwidth: float) -> np.ndarray:
    """
    Nadaraya-Watson Kernel Average Smoother (Local Mean).

    For each point t_i, compute:
        y_smooth[i] = SUM y_j * K((t_j - t_i) / h) / SUM K((t_j - t_i) / h)

    This is NOT robust - outliers heavily influence the weighted mean.

    Parameters
    ----------
    t : array
        Time points (x-axis)
    y : array
        Signal values
    bandwidth : float
        Smoothing bandwidth h

    Returns
    -------
    y_smooth : array
        Smoothed signal using weighted mean
    """
    n = len(t)
    y_smooth = np.zeros(n)

    for i in range(n):
        u = (t - t[i]) / bandwidth
        weights = epanechnikov_kernel(u)
        total = np.sum(weights)
        if total > 0:
            y_smooth[i] = np.sum(weights * y) / total
        else:
            y_smooth[i] = np.nan

    return y_smooth


def emd_iteration(t: np.ndarray, y: np.ndarray, h0: float, decay: float,
                  k_max: int, method: str = 'local_median') -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Perform EMD-style iterative decomposition.

    At each iteration k:
        X^(k+1) = X^(k) - X_tilde^(k)

    where X_tilde^(k) is the smooth component extracted using the chosen method.

    The bandwidth decreases with each iteration:
        h_k = h_0 / (decay^k)

    Parameters
    ----------
    t : array
        Time points
    y : array
        Input signal
    h0 : float
        Initial bandwidth
    decay : float
        Bandwidth decay factor (typically sqrt(2) = 1.414)
    k_max : int
        Maximum number of iterations
    method : str
        'local_median', 'median', or 'average'

    Returns
    -------
    X_list : list of arrays
        X^(k) for k = 0, 1, ..., k_max (input at each iteration)
    X_tilde_list : list of arrays
        X_tilde^(k) for k = 1, ..., k_max (smooth component at each iteration)
    """
    # Select smoothing function
    if method == 'local_median':
        smoother = epa_local_median
    elif method == 'median':
        smoother = median_smoother
    elif method == 'average':
        smoother = average_smoother
    else:
        raise ValueError(f"Unknown method: {method}")

    X_list = [y.copy()]  # X^(0) = original signal
    X_tilde_list = []

    X_current = y.copy()

    for k in range(1, k_max + 1):
        # Compute bandwidth for this iteration
        h_k = h0 / (decay ** (k - 1))

        # Compute smooth component
        X_tilde = smoother(t, X_current, h_k)
        X_tilde_list.append(X_tilde)

        # Compute residual for next iteration
        X_next = X_current - X_tilde
        X_list.append(X_next)

        X_current = X_next

    return X_list, X_tilde_list


def compute_metrics(y_estimated: np.ndarray, y_true: np.ndarray) -> Tuple[float, float]:
    """
    Compute RMSE and MAE metrics.

    Parameters
    ----------
    y_estimated : array
        Estimated/smoothed signal
    y_true : array
        True signal (ground truth)

    Returns
    -------
    rmse : float
        Root Mean Squared Error
    mae : float
        Mean Absolute Error
    """
    rmse = float(np.sqrt(np.mean((y_estimated - y_true) ** 2)))
    mae = float(np.mean(np.abs(y_estimated - y_true)))
    return rmse, mae
