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
    Compute weighted median with linear interpolation for smooth output.

    The weighted median minimizes: argmin_m SUM |y_i - m| * w_i

    Instead of returning a discrete sample value, this interpolates between
    adjacent values at the 0.5 cumulative weight crossing. This eliminates
    step-like artifacts when smoothing clean/noiseless data.

    Algorithm:
    1. Sort values and corresponding weights
    2. Compute cumulative normalized weights
    3. Find crossing point at cumulative weight = 0.5
    4. Linearly interpolate between adjacent values at crossing
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
    if idx == 0:
        return sorted_values[0]

    # Linear interpolation between y[idx-1] and y[idx]
    c_prev = cumulative_weights[idx - 1]
    c_curr = cumulative_weights[idx]

    if c_curr > c_prev:
        alpha = (0.5 - c_prev) / (c_curr - c_prev)
    else:
        alpha = 0.5

    alpha = np.clip(alpha, 0.0, 1.0)
    return sorted_values[idx - 1] + alpha * (sorted_values[idx] - sorted_values[idx - 1])


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


# =============================================================================
# Running Median (ionASE-style) Implementation for Comparison
# =============================================================================

def running_median_heap(arr: np.ndarray, k: int) -> np.ndarray:
    """
    Simple running median with fixed window size k.

    ionASE-style: fixed window, unweighted median, asymmetric boundaries.
    For comparison with EPA kernel-weighted approach.

    Parameters
    ----------
    arr : array
        Input signal values
    k : int
        Window size (will be forced to odd)

    Returns
    -------
    medians : array
        Running median smoothed signal
    """
    if k % 2 == 0:
        k += 1
    n = len(arr)
    half_k = k // 2
    medians = np.zeros(n)

    for i in range(n):
        left = max(0, i - half_k)
        right = min(n - 1, i + half_k)
        medians[i] = np.median(arr[left:right + 1])

    return medians


def bandwidth_to_window_size(h: float, t: np.ndarray) -> int:
    """
    Convert continuous bandwidth h to integer window size k.

    Parameters
    ----------
    h : float
        Continuous bandwidth (kernel half-width)
    t : array
        Time array (used to compute point spacing)

    Returns
    -------
    k : int
        Integer window size (always odd, minimum 3)
    """
    dt = t[1] - t[0] if len(t) > 1 else 1.0
    k = int(2 * h / dt) + 1
    if k % 2 == 0:
        k += 1
    return max(3, k)


def emd_iteration_heap(y: np.ndarray, window_size: int, k_max: int,
                       decay_bandwidth: bool = False,
                       decay: float = 1.414) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    EMD iteration using running median (ionASE-style).

    Key difference from EPA approach: NO bandwidth decay by default.
    The ionASE implementation uses constant window size throughout iterations.

    Parameters
    ----------
    y : array
        Input signal
    window_size : int
        Initial window size (must be odd)
    k_max : int
        Maximum number of iterations
    decay_bandwidth : bool
        If True, decay window size each iteration (not standard for ionASE)
    decay : float
        Decay factor if decay_bandwidth=True

    Returns
    -------
    X_list : list of arrays
        X^(k) for k = 0, 1, ..., k_max
    X_tilde_list : list of arrays
        X_tilde^(k) for k = 1, ..., k_max
    """
    X_list = [y.copy()]
    X_tilde_list = []
    X_current = y.copy()
    current_window = window_size

    # Ensure odd window size
    if current_window % 2 == 0:
        current_window += 1

    for k in range(1, k_max + 1):
        X_tilde = running_median_heap(X_current, current_window)
        X_tilde_list.append(X_tilde)
        X_next = X_current - X_tilde
        X_list.append(X_next)
        X_current = X_next

        if decay_bandwidth:
            current_window = max(3, int(current_window / decay))
            if current_window % 2 == 0:
                current_window += 1

    return X_list, X_tilde_list


# =============================================================================
# ionASE-style Local Median Implementation for Comparison
# Based on: https://github.com/ionASE-coder/A_New_EMD_Approach
# =============================================================================

def ionase_kernel_weights(t_idx: int, n: int, h: float) -> np.ndarray:
    """
    ionASE-style kernel weights: K_h(s - t) = K((s-t)/(n*h)) / h

    Key difference from our EPA: ionASE normalizes by bandwidth h
    and uses index-based distances scaled by n.

    Parameters
    ----------
    t_idx : int
        Index of evaluation point
    n : int
        Total number of points
    h : float
        Bandwidth parameter (fraction of total signal)

    Returns
    -------
    weights : array
        Kernel weights for all points
    """
    s = np.arange(n)
    # ionASE uses index-based distance scaled by n
    u = (s - t_idx) / n
    # Epanechnikov kernel with bandwidth normalization
    mask = np.abs(u / h) <= 1
    weights = np.zeros(n, dtype=float)
    weights[mask] = (0.75 * (1 - (u[mask] / h) ** 2)) / h
    return weights


def ionase_weighted_median(x: np.ndarray, weights: np.ndarray) -> float:
    """
    ionASE-style weighted median: NO interpolation, exact index.

    Key difference from our EPA: ionASE returns x[k] at exact index
    where cumsum first crosses 0.5, without linear interpolation.
    This can cause micro-jumps in the output.

    Parameters
    ----------
    x : array
        Signal values
    weights : array
        Kernel weights

    Returns
    -------
    median : float
        Weighted median value
    """
    mask = weights > 0
    if not np.any(mask):
        return np.nan

    x_filtered = x[mask]
    w_filtered = weights[mask]

    sorted_idx = np.argsort(x_filtered)
    x_sorted = x_filtered[sorted_idx]
    w_sorted = w_filtered[sorted_idx]

    cumsum = np.cumsum(w_sorted)
    median_idx = np.searchsorted(cumsum, cumsum[-1] / 2)

    # ionASE: return exact value at index, NO interpolation
    return x_sorted[min(median_idx, len(x_sorted) - 1)]


def ionase_local_median_fit(x: np.ndarray, h: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    ionASE-style local median fit with boundary trimming.

    Key difference from our EPA: Returns NaN outside valid range [h*n, (1-h)*n].
    This avoids boundary effects but reduces usable signal length.

    Parameters
    ----------
    x : array
        Input signal
    h : float
        Bandwidth parameter (fraction of total signal)

    Returns
    -------
    x_tilde : array
        Smoothed signal (NaN outside valid range)
    valid_mask : array
        Boolean mask indicating valid (non-NaN) points
    """
    n = len(x)
    x_tilde = np.full(n, np.nan)

    # Boundary trimming: only compute in valid range
    t_min = int(np.ceil(h * n))
    t_max = int(np.floor((1 - h) * n))

    # Ensure at least some valid range
    t_min = max(0, t_min)
    t_max = min(n - 1, t_max)

    for t in range(t_min, t_max + 1):
        weights = ionase_kernel_weights(t, n, h)
        x_tilde[t] = ionase_weighted_median(x, weights)

    valid_mask = ~np.isnan(x_tilde)
    return x_tilde, valid_mask


def ionase_emd_decomposition(x: np.ndarray, h_init: float, decay: float,
                              k_max: int) -> Tuple[List[np.ndarray], np.ndarray, np.ndarray]:
    """
    ionASE-style EMD with NaN propagation through iterations.

    Key difference from our EPA: NaN regions accumulate through iterations,
    reducing the valid signal region at each step.

    Parameters
    ----------
    x : array
        Input signal
    h_init : float
        Initial bandwidth
    decay : float
        Bandwidth decay factor
    k_max : int
        Maximum iterations

    Returns
    -------
    imfs : list of arrays
        Extracted smooth components (IMFs) at each iteration
    residual : array
        Final residual
    combined_mask : array
        Boolean mask of points valid across all iterations
    """
    n = len(x)
    residual = x.copy()
    imfs = []
    combined_mask = np.ones(n, dtype=bool)
    h_k = h_init

    for k in range(k_max):
        x_tilde, valid_mask = ionase_local_median_fit(residual, h_k)
        combined_mask &= valid_mask
        imfs.append(x_tilde.copy())
        # Update residual only where valid
        residual = np.where(valid_mask, residual - x_tilde, residual)
        # Bandwidth decay
        h_k = h_k / decay

    return imfs, residual, combined_mask


def epa_detailed_computation(t: np.ndarray, y: np.ndarray, t_i: float, h: float) -> dict:
    """
    Return detailed computation steps for EPA local median at point t_i.

    Useful for step-by-step algorithm trace visualization.

    Parameters
    ----------
    t : array
        Time points
    y : array
        Signal values
    t_i : float
        Evaluation point
    h : float
        Bandwidth

    Returns
    -------
    details : dict
        All intermediate computation values
    """
    u = (t - t_i) / h
    weights = epanechnikov_kernel(u)

    # Sort by y values
    sorted_indices = np.argsort(y)
    y_sorted = y[sorted_indices]
    weights_sorted = weights[sorted_indices]
    u_sorted = u[sorted_indices]
    t_sorted = t[sorted_indices]

    # Cumulative weights
    total_weight = np.sum(weights_sorted)
    if total_weight > 0:
        cumsum_norm = np.cumsum(weights_sorted) / total_weight
    else:
        cumsum_norm = np.zeros_like(weights_sorted)

    # Find median via 0.5 crossing WITH interpolation
    idx = np.searchsorted(cumsum_norm, 0.5)
    if idx >= len(y_sorted):
        median_result = y_sorted[-1]
        alpha = 1.0
    elif idx == 0:
        median_result = y_sorted[0]
        alpha = 0.0
    else:
        c_prev = cumsum_norm[idx - 1]
        c_curr = cumsum_norm[idx]
        if c_curr > c_prev:
            alpha = (0.5 - c_prev) / (c_curr - c_prev)
        else:
            alpha = 0.5
        alpha = np.clip(alpha, 0.0, 1.0)
        median_result = y_sorted[idx - 1] + alpha * (y_sorted[idx] - y_sorted[idx - 1])

    return {
        't': t,
        'y': y,
        't_i': t_i,
        'h': h,
        'u': u,
        'weights': weights,
        'y_sorted': y_sorted,
        'weights_sorted': weights_sorted,
        'u_sorted': u_sorted,
        't_sorted': t_sorted,
        'cumsum_norm': cumsum_norm,
        'total_weight': total_weight,
        'median_idx': idx,
        'alpha': alpha,
        'median_result': median_result
    }


def ionase_detailed_computation(x: np.ndarray, t_idx: int, h: float) -> dict:
    """
    Return detailed computation steps for ionASE local median at index t_idx.

    Useful for step-by-step algorithm trace visualization.

    Parameters
    ----------
    x : array
        Signal values
    t_idx : int
        Evaluation index
    h : float
        Bandwidth

    Returns
    -------
    details : dict
        All intermediate computation values
    """
    n = len(x)
    weights = ionase_kernel_weights(t_idx, n, h)

    # Filter to non-zero weights
    mask = weights > 0
    x_filtered = x[mask]
    w_filtered = weights[mask]
    indices_filtered = np.where(mask)[0]

    # Sort by x values
    sorted_idx = np.argsort(x_filtered)
    x_sorted = x_filtered[sorted_idx]
    w_sorted = w_filtered[sorted_idx]
    original_indices_sorted = indices_filtered[sorted_idx]

    # Cumulative weights
    cumsum = np.cumsum(w_sorted)
    total_weight = cumsum[-1] if len(cumsum) > 0 else 0
    cumsum_norm = cumsum / total_weight if total_weight > 0 else cumsum

    # Find median - NO interpolation (ionASE style)
    median_idx = np.searchsorted(cumsum_norm, 0.5)
    median_idx = min(median_idx, len(x_sorted) - 1) if len(x_sorted) > 0 else 0

    median_result = x_sorted[median_idx] if len(x_sorted) > 0 else np.nan

    return {
        'x': x,
        't_idx': t_idx,
        'h': h,
        'n': n,
        'weights': weights,
        'mask': mask,
        'x_filtered': x_filtered,
        'w_filtered': w_filtered,
        'indices_filtered': indices_filtered,
        'x_sorted': x_sorted,
        'w_sorted': w_sorted,
        'original_indices_sorted': original_indices_sorted,
        'cumsum': cumsum,
        'cumsum_norm': cumsum_norm,
        'total_weight': total_weight,
        'median_idx': median_idx,
        'median_result': median_result
    }
