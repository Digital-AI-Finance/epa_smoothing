"""EMD Sifting Iteration 2 - Bandwidth h=0.0707

Shows second sifting iteration after subtracting first IMF.
Bandwidth: h2 = h1/sqrt(2) = 0.1/sqrt(2) = 0.0707
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
    weights = np.zeros_like(u, dtype=float)
    mask = np.abs(u) <= 1
    weights[mask] = 0.75 * (1 - u[mask]**2)
    return weights


def weighted_median(values, weights):
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
    n = len(x)
    y_smooth = np.zeros(n)
    for i in range(n):
        u = (x - x[i]) / bandwidth
        weights = epanechnikov_kernel(u)
        y_smooth[i] = weighted_median(y, weights)
    return y_smooth


# =============================================================================
# Signal generation and sifting
# =============================================================================

np.random.seed(42)
n = 500
x = np.linspace(0, 1, n)

# AM-FM signal
A_x = 1 + 0.5 * np.sin(4 * np.pi * x)
phi_x = 2 * np.pi * (6 * x + 12 * x**2)
y_clean = A_x * np.sin(phi_x)
sigma = 0.2
noise = sigma * np.random.randn(n)
y_noisy = y_clean + noise

# Bandwidth sequence
h1 = 0.1
h2 = h1 / np.sqrt(2)  # 0.0707

# Iteration 1
m1_noisy = epa_local_median(x, y_noisy, h1)
m1_clean = epa_local_median(x, y_clean, h1)

# Residual after iteration 1
r2_noisy = y_noisy - m1_noisy
r2_clean = y_clean - m1_clean

# Iteration 2: smooth the residual
m2_noisy = epa_local_median(x, r2_noisy, h2)
m2_clean = epa_local_median(x, r2_clean, h2)

# =============================================================================
# Plot
# =============================================================================

fig, ax = plt.subplots(figsize=(10, 6))

# Plot residual from iteration 1
ax.plot(x, r2_noisy, color=MLGRAY, alpha=0.4, linewidth=0.8, label='Residual (noisy)')
ax.plot(x, r2_clean, color=MLBLUE, linewidth=1.5, alpha=0.7, label='Residual (clean)')

# Plot extracted IMF
ax.plot(x, m2_noisy, color=MLRED, linewidth=2.5, label=f'IMF estimate (noisy, h={h2:.4f})')
ax.plot(x, m2_clean, color=MLGREEN, linewidth=2, linestyle='--', label=f'IMF estimate (clean, h={h2:.4f})')

ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_title(f'Sifting Iteration 2: h = {h2:.4f} (h1/sqrt(2))')
ax.legend(loc='upper right', framealpha=0.9)
ax.grid(True, alpha=0.3)
ax.set_xlim(0, 1)

plt.tight_layout()
plt.savefig(Path(__file__).parent / 'chart.pdf', dpi=300, bbox_inches='tight')
plt.close()

print(f"Saved: {Path(__file__).parent / 'chart.pdf'}")
