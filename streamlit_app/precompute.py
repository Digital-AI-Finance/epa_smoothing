"""
Pre-computation Script for EMD Local Median Streamlit App

Generates pre-computed results for the parameter grid to enable
instant response in the interactive app.

Grid Parameters:
- h0: [0.03, 0.05, 0.10, 0.15, 0.20, 0.30]
- sigma: [0.10, 0.20, 0.30, 0.50]
- decay: [1.2, 1.414, 1.7]
- k: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

For each combination, we store:
- X_list: signal at each iteration
- X_tilde_list: smooth component at each iteration
- Metrics: RMSE, MAE for each method

Output: Split gzipped files by sigma, organized by n_points:
    precomputed_data/
        n_2000/
            metadata.pkl.gz
            sigma_0.10.pkl.gz, sigma_0.20.pkl.gz, ...
        n_4000/
            metadata.pkl.gz
            sigma_0.10.pkl.gz, sigma_0.20.pkl.gz, ...

Each sigma file is ~29 MB (n=2000) or ~59 MB (n=4000), under GitHub's 100 MB limit.

Usage:
    python precompute.py           # Compute for n=2000 (default)
    python precompute.py --n 4000  # Compute for n=4000
    python precompute.py --all     # Compute for all supported n_points
"""

import argparse
import numpy as np
import pickle
import gzip
from pathlib import Path
from typing import Dict, Any, List
import time

from smoothers import emd_iteration, compute_metrics
from signals import generate_amfm_signal, get_default_signal_params


def precompute_grid(n_points: int = 2000) -> Dict[str, Any]:
    """
    Pre-compute EMD iterations for all parameter combinations.

    Parameters
    ----------
    n_points : int
        Number of signal points to generate. Default 2000.

    Returns
    -------
    data : dict
        Dictionary containing:
        - 't': time array
        - 'n_points': number of points
        - 'y_clean': clean signal (fixed AM-FM params)
        - 'grid': dictionary keyed by parameter string
    """
    print(f"Starting pre-computation for n_points={n_points}...")
    start_time = time.time()

    # Fixed signal parameters (use defaults, override n_points)
    signal_params = get_default_signal_params()
    signal_params['n_points'] = n_points  # Override with requested n_points
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
    """Save pre-computed data to pickle file (legacy single-file format)."""
    with open(output_path, 'wb') as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"Saved: {output_path} ({size_mb:.2f} MB)")


def save_split_precomputed_data(data: Dict[str, Any], output_dir: Path) -> None:
    """Save precomputed data as split gzipped files by sigma in n_XXXX subdirectory.

    Creates structure:
        output_dir/n_XXXX/
            metadata.pkl.gz: shared params (t, signal_params, grid values)
            sigma_X.XX.pkl.gz: per-sigma data (y_clean, y_noisy, iterations)

    Each sigma file is ~29 MB (n=2000) or ~59 MB (n=4000), under GitHub's 100 MB limit.
    """
    # Create n_XXXX subdirectory
    n_points = data['n_points']
    n_dir = output_dir / f'n_{n_points}'
    n_dir.mkdir(parents=True, exist_ok=True)
    total_size = 0

    # Save metadata (shared across all sigma)
    metadata = {
        't': data['t'],
        'n_points': data['n_points'],
        'signal_params': data['signal_params'],
        'h0_values': data['h0_values'],
        'sigma_values': data['sigma_values'],
        'decay_values': data['decay_values'],
        'k_max': data['k_max'],
        'methods': data['methods']
    }

    metadata_path = n_dir / 'metadata.pkl.gz'
    with gzip.open(metadata_path, 'wb', compresslevel=9) as f:
        pickle.dump(metadata, f, protocol=pickle.HIGHEST_PROTOCOL)

    meta_size = metadata_path.stat().st_size / (1024 * 1024)
    total_size += meta_size
    print(f"  metadata.pkl.gz: {meta_size:.2f} MB")

    # Save per-sigma files
    for sigma in data['sigma_values']:
        sigma_key = f'sigma_{sigma}'
        sigma_data = data['results'][sigma_key]
        filename = f'sigma_{sigma:.2f}.pkl.gz'
        filepath = n_dir / filename

        with gzip.open(filepath, 'wb', compresslevel=9) as f:
            pickle.dump(sigma_data, f, protocol=pickle.HIGHEST_PROTOCOL)

        size_mb = filepath.stat().st_size / (1024 * 1024)
        total_size += size_mb
        print(f"  {filename}: {size_mb:.2f} MB")

    print(f"\nTotal: {total_size:.2f} MB across {len(data['sigma_values']) + 1} files")
    print(f"Output directory: {n_dir}")


SUPPORTED_N_POINTS = [2000, 4000, 10000]


def main():
    """Main entry point for pre-computation."""
    parser = argparse.ArgumentParser(
        description='Pre-compute EMD Local Median results for Streamlit app'
    )
    parser.add_argument(
        '--n', type=int, default=None,
        help=f'Number of signal points to compute. Supported: {SUPPORTED_N_POINTS}'
    )
    parser.add_argument(
        '--all', action='store_true',
        help=f'Compute for all supported n_points: {SUPPORTED_N_POINTS}'
    )
    args = parser.parse_args()

    base_dir = Path(__file__).parent
    output_dir = base_dir / 'precomputed_data'

    # Determine which n_points to compute
    if args.all:
        n_points_list = SUPPORTED_N_POINTS
    elif args.n is not None:
        if args.n not in SUPPORTED_N_POINTS:
            print(f"Warning: n={args.n} is not in standard list {SUPPORTED_N_POINTS}")
            print("Proceeding anyway...")
        n_points_list = [args.n]
    else:
        # Default: compute for n=2000
        n_points_list = [2000]

    print("EMD Local Median Pre-computation")
    print("=" * 50)
    print(f"Will compute for n_points: {n_points_list}")
    print()

    for n_points in n_points_list:
        print(f"\n{'='*50}")
        print(f"Computing for n_points = {n_points}")
        print(f"{'='*50}")

        data = precompute_grid(n_points=n_points)

        print("\nSaving split gzipped files...")
        save_split_precomputed_data(data, output_dir)

    print("\n" + "=" * 50)
    print("Pre-computation complete!")
    print(f"Output directory: {output_dir}")

    # List all generated subdirectories
    for n_subdir in sorted(output_dir.iterdir()):
        if n_subdir.is_dir() and n_subdir.name.startswith('n_'):
            files = list(n_subdir.glob('*.pkl.gz'))
            total_mb = sum(f.stat().st_size for f in files) / (1024 * 1024)
            print(f"  {n_subdir.name}/: {len(files)} files, {total_mb:.1f} MB total")


if __name__ == '__main__':
    main()
