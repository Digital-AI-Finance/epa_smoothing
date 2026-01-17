"""
Pre-computation Script for EMD Local Median Streamlit App

Generates pre-computed results for the parameter grid to enable
instant response in the interactive app.

Grid Parameters:
- h0: [0.01, 0.02, 0.03, 0.05, 0.07, 0.10, 0.15, 0.20, 0.25, 0.30]
- sigma: [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]
- decay: [1.1, 1.2, 1.3, 1.414, 1.5, 1.7, 2.0]
- k: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

For each combination, we store:
- X_list: signal at each iteration
- X_tilde_list: smooth component at each iteration
- Metrics: RMSE, MAE for each method

Usage:
    python precompute.py
"""

import numpy as np
import pickle
from pathlib import Path
from typing import Dict, Any
import time

from smoothers import emd_iteration, compute_metrics
from signals import generate_amfm_signal, get_default_signal_params


def precompute_grid() -> Dict[str, Any]:
    """
    Pre-compute EMD iterations for all parameter combinations.

    Returns
    -------
    data : dict
        Dictionary containing:
        - 't': time array
        - 'y_clean': clean signal (fixed AM-FM params)
        - 'grid': dictionary keyed by parameter string
    """
    print("Starting pre-computation...")
    start_time = time.time()

    # Fixed signal parameters (use defaults)
    signal_params = get_default_signal_params()
    n_points = signal_params['n_points']
    t_start = signal_params['t_start']
    t_end = signal_params['t_end']
    am_depth = signal_params['am_depth']
    base_freq = signal_params['base_freq']
    chirp_rate = signal_params['chirp_rate']
    seed = signal_params['seed']

    # Parameter grids (reduced for faster computation)
    h0_values = [0.03, 0.05, 0.10, 0.15, 0.20, 0.30]
    sigma_values = [0.10, 0.20, 0.30, 0.50]
    decay_values = [1.2, 1.414, 1.7]
    k_max = 10
    methods = ['local_median', 'median', 'average']

    # Generate time array (fixed)
    t = np.linspace(t_start, t_end, n_points)

    # Data structure
    data = {
        't': t.tolist(),
        'n_points': n_points,
        'signal_params': signal_params,
        'h0_values': h0_values,
        'sigma_values': sigma_values,
        'decay_values': decay_values,
        'k_max': k_max,
        'methods': methods,
        'results': {}
    }

    # Total combinations
    total = len(h0_values) * len(sigma_values) * len(decay_values) * len(methods)
    count = 0

    for sigma in sigma_values:
        # Generate noisy signal for this noise level
        _, y_clean, y_noisy = generate_amfm_signal(
            n_points=n_points,
            t_start=t_start,
            t_end=t_end,
            am_depth=am_depth,
            base_freq=base_freq,
            chirp_rate=chirp_rate,
            noise_sigma=sigma,
            seed=seed
        )

        # Store clean/noisy for this sigma
        sigma_key = f"sigma_{sigma}"
        data['results'][sigma_key] = {
            'y_clean': y_clean.tolist(),
            'y_noisy': y_noisy.tolist(),
            'iterations': {}
        }

        for h0 in h0_values:
            for decay in decay_values:
                for method in methods:
                    count += 1
                    if count % 100 == 0:
                        elapsed = time.time() - start_time
                        print(f"  Progress: {count}/{total} ({100*count/total:.1f}%) - {elapsed:.1f}s")

                    # Run EMD iteration
                    X_list, X_tilde_list = emd_iteration(
                        t, y_noisy, h0, decay, k_max, method
                    )

                    # Also compute for clean signal
                    X_list_clean, X_tilde_list_clean = emd_iteration(
                        t, y_clean, h0, decay, k_max, method
                    )

                    # Compute metrics for each iteration
                    metrics_noisy = []
                    metrics_clean = []

                    for k in range(k_max):
                        # Metrics comparing smooth estimate to clean signal
                        if k < len(X_tilde_list):
                            rmse_n, mae_n = compute_metrics(X_tilde_list[k], y_clean)
                            rmse_c, mae_c = compute_metrics(X_tilde_list_clean[k], y_clean)
                        else:
                            rmse_n, mae_n = np.nan, np.nan
                            rmse_c, mae_c = np.nan, np.nan

                        metrics_noisy.append({'rmse': rmse_n, 'mae': mae_n})
                        metrics_clean.append({'rmse': rmse_c, 'mae': mae_c})

                    # Store results
                    key = f"h0_{h0}_decay_{decay}_{method}"
                    data['results'][sigma_key]['iterations'][key] = {
                        'X_list': [x.tolist() for x in X_list],
                        'X_tilde_list': [x.tolist() for x in X_tilde_list],
                        'X_list_clean': [x.tolist() for x in X_list_clean],
                        'X_tilde_list_clean': [x.tolist() for x in X_tilde_list_clean],
                        'metrics_noisy': metrics_noisy,
                        'metrics_clean': metrics_clean
                    }

    elapsed = time.time() - start_time
    print(f"Completed {count} combinations in {elapsed:.1f} seconds")

    return data


def save_precomputed_data(data: Dict[str, Any], output_path: Path) -> None:
    """Save pre-computed data to pickle file."""
    with open(output_path, 'wb') as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"Saved: {output_path} ({size_mb:.2f} MB)")


def main():
    """Main entry point for pre-computation."""
    output_path = Path(__file__).parent / 'precomputed_data.pkl'

    print("EMD Local Median Pre-computation")
    print("=" * 50)

    data = precompute_grid()
    save_precomputed_data(data, output_path)

    print("\nPre-computation complete!")
    print(f"Output: {output_path}")


if __name__ == '__main__':
    main()
