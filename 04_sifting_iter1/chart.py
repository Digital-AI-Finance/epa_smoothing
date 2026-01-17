"""EMD Sifting Iteration 1 - Bandwidth h=0.1

Shows the first sifting iteration with initial signal and extracted IMF.
AM-FM Signal: A(x) = 1 + 0.5*sin(4*pi*x), phi(x) = 2*pi*(6x + 12x^2)
"""
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

plt.rcParams.update({
    'font.size': 14, 'axes.labelsize': 14, 'axes.titlesize': 16,
    'xtick.labelsize': 13, 'ytick.labelsize': 13, 'legend.fontsize': 12,
    'figure.figsize': (10, 6), 'figure.dpi': 150
})

# Colors
MLBLUE = '#0066CC'
MLGREEN = '#2CA02C'
MLRED = '#D62728'
MLGRAY = '#888888'
MLORANGE = '#FF7F0E'

# =============================================================================
# Smoothing functions
# =============================================================================

def epanechnikov_kernel(u):
    """EPA kernel: K(u) = 0.75(1 - u^2) for |u| <= 1"""
    weights = np.zeros_like(u, dtype=float)
    mask = np.abs(u) <= 1
    weights[mask] = 0.75 * (1 - u[mask]**2)
    return weights


def weighted_median(values, weights):
    """Compute weighted median."""
    if len(values) == 0 or np.sum(weights) == 0:
        return np.nan
    sorted_indices = np.argsort(values)
    sorted_values = values[sorted_indices]
    sorted_weights = weights[sorted_indices]
    cumulative_weights = np.cumsum(sorted_weights) / np.sum(sorted_weights)
    idx = np.searchsorted(cumulative_weights, 0.5)
    if idx >= len(sorted_values):
        return sorted_values[-1]
    elif idx == 0:
        return sorted_values[0]
    elif np.isclose(cumulative_weights[idx-1], 0.5):
        return 0.5 * (sorted_values[idx-1] + sorted_values[idx])
    return sorted_values[idx]


def epa_local_median(x, y, bandwidth):
    """Kernel-weighted local median smoother."""
    n = len(x)
    y_smooth = np.zeros(n)
    for i in range(n):
        u = (x - x[i]) / bandwidth
        weights = epanechnikov_kernel(u)
        y_smooth[i] = weighted_median(y, weights)
    return y_smooth


# =============================================================================
# Signal generation
# =============================================================================

np.random.seed(42)
n = 500
x = np.linspace(0, 1, n)

# AM-FM signal parameters from docx
A_x = 1 + 0.5 * np.sin(4 * np.pi * x)  # Amplitude modulation
phi_x = 2 * np.pi * (6 * x + 12 * x**2)  # Phase
y_clean = A_x * np.sin(phi_x)  # Clean AM-FM signal

# Add noise
sigma = 0.2
noise = sigma * np.random.randn(n)
y_noisy = y_clean + noise

# Iteration 1: h = 0.1
h1 = 0.1

# Initial residual is the noisy signal
r1_noisy = y_noisy
r1_clean = y_clean

# Extract IMF using local median smoothing
m1_noisy = epa_local_median(x, r1_noisy, h1)
m1_clean = epa_local_median(x, r1_clean, h1)

# =============================================================================
# Plot
# =============================================================================

fig, ax = plt.subplots(figsize=(10, 6))

# Plot noisy residual (light)
ax.plot(x, r1_noisy, color=MLGRAY, alpha=0.4, linewidth=0.8, label='Noisy signal')

# Plot clean residual
ax.plot(x, r1_clean, color=MLBLUE, linewidth=1.5, alpha=0.7, label='Clean AM-FM signal')

# Plot extracted IMF (median smoothed)
ax.plot(x, m1_noisy, color=MLRED, linewidth=2.5, label=f'IMF estimate (noisy, h={h1})')
ax.plot(x, m1_clean, color=MLGREEN, linewidth=2, linestyle='--', label=f'IMF estimate (clean, h={h1})')

ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_title(f'Sifting Iteration 1: h = {h1}')
ax.legend(loc='upper right', framealpha=0.9)
ax.grid(True, alpha=0.3)
ax.set_xlim(0, 1)

plt.tight_layout()
plt.savefig(Path(__file__).parent / 'chart.pdf', dpi=300, bbox_inches='tight')
plt.close()

print(f"Saved: {Path(__file__).parent / 'chart.pdf'}")
