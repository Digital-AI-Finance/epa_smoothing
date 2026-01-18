"""
AM-FM Signal Generation for EMD Local Median Demonstration

AM-FM signals (Amplitude Modulation - Frequency Modulation) are common in:
- Audio processing
- Radar/sonar
- Biomedical signals (ECG, EEG)
- Seismic analysis

The signal has the form:
    y(t) = a(t) * cos(phi(t)) + noise

where:
    a(t) = amplitude envelope (AM component)
    phi(t) = instantaneous phase (FM component via frequency chirp)
"""

import numpy as np
from typing import Tuple


def generate_amfm_signal(
    n_points: int = 2000,
    t_start: float = 0.0,
    t_end: float = 1.0,
    am_depth: float = 0.5,
    base_freq: float = 6.0,
    chirp_rate: float = 12.0,
    noise_sigma: float = 0.2,
    seed: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate AM-FM signal with additive Gaussian noise.

    Signal model:
        y_clean(t) = [1 + d * sin(4*pi*t)] * sin(2*pi*(f0*t + c*t^2))
        y_noisy(t) = y_clean(t) + sigma * epsilon(t)

    where epsilon ~ N(0, 1)

    Parameters
    ----------
    n_points : int
        Number of sample points
    t_start : float
        Start time
    t_end : float
        End time
    am_depth : float
        Amplitude modulation depth (d), range [0, 1]
        - d=0: no amplitude modulation
        - d=1: amplitude varies from 0 to 2
    base_freq : float
        Base frequency in Hz (f0)
    chirp_rate : float
        Frequency chirp coefficient (c)
        - Positive: frequency increases with time
        - Negative: frequency decreases with time
    noise_sigma : float
        Standard deviation of Gaussian noise
    seed : int
        Random seed for reproducibility

    Returns
    -------
    t : array
        Time points
    y_clean : array
        Clean AM-FM signal (ground truth)
    y_noisy : array
        Noisy AM-FM signal
    """
    np.random.seed(seed)

    # Time vector
    t = np.linspace(t_start, t_end, n_points)

    # Amplitude envelope: a(t) = 1 + d * sin(4*pi*t)
    # This creates slow amplitude modulation
    amplitude = 1 + am_depth * np.sin(4 * np.pi * t)

    # Instantaneous phase: phi(t) = 2*pi*(f0*t + c*t^2)
    # The chirp term c*t^2 causes frequency to increase linearly with time
    phase = 2 * np.pi * (base_freq * t + chirp_rate * t**2)

    # Clean signal
    y_clean = amplitude * np.sin(phase)

    # Add noise
    noise = noise_sigma * np.random.randn(n_points)
    y_noisy = y_clean + noise

    return t, y_clean, y_noisy


def generate_amfm_components(
    n_points: int = 2000,
    t_start: float = 0.0,
    t_end: float = 1.0,
    am_depth: float = 0.5,
    base_freq: float = 6.0,
    chirp_rate: float = 12.0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate AM-FM signal components for visualization.

    Returns the amplitude envelope and instantaneous frequency
    separately for pedagogical purposes.

    Parameters
    ----------
    n_points : int
        Number of sample points
    t_start : float
        Start time
    t_end : float
        End time
    am_depth : float
        Amplitude modulation depth
    base_freq : float
        Base frequency in Hz
    chirp_rate : float
        Frequency chirp coefficient

    Returns
    -------
    t : array
        Time points
    amplitude : array
        Amplitude envelope a(t)
    inst_freq : array
        Instantaneous frequency f(t) = d(phi)/dt / (2*pi)
    """
    t = np.linspace(t_start, t_end, n_points)

    # Amplitude envelope
    amplitude = 1 + am_depth * np.sin(4 * np.pi * t)

    # Instantaneous frequency: f(t) = f0 + 2*c*t
    # (derivative of phase divided by 2*pi)
    inst_freq = base_freq + 2 * chirp_rate * t

    return t, amplitude, inst_freq


def get_default_signal_params() -> dict:
    """
    Get default AM-FM signal parameters.

    These defaults produce a visually interesting signal
    that clearly demonstrates the EMD decomposition.
    """
    return {
        'n_points': 2000,
        't_start': 0.0,
        't_end': 1.0,
        'am_depth': 0.5,
        'base_freq': 6.0,
        'chirp_rate': 12.0,
        'noise_sigma': 0.20,
        'seed': 42
    }


def get_smoothing_defaults() -> dict:
    """
    Get default smoothing parameters.

    These defaults work well for the default AM-FM signal.
    """
    return {
        'h0': 0.10,           # Initial bandwidth
        'sigma': 0.20,        # Noise level
        'decay': 1.414,       # sqrt(2) bandwidth decay
        'k_max': 10           # Maximum iterations
    }
