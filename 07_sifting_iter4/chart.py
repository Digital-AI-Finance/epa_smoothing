"""EMD Sifting Iteration 4 - Bandwidth h=0.0354

Shows fourth sifting iteration.
Bandwidth: h4 = h1/(2*sqrt(2)) = 0.0354
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

# =============================================================================
# Smoothing functions
# =============================================================================

def epanechnikov_kernel(u):
    weights = np.zeros_like(u, dtype=float)
    mask = np.abs(u) <= 1
    weights[mask] = 0.75 * (1 - u[mask]**2)
    return weights


def weighted_median(values, weights):
    """Compute weighted median with linear interpolation for smooth output."""
    if len(values) == 0 or np.sum(weights) == 0:
        return np.nan
    sorted_indices = np.argsort(values)
    sorted_values = values[sorted_indices]
    sorted_weights = weights[sorted_indices]
    cumulative_weights = np.cumsum(sorted_weights) / np.sum(sorted_weights)
    idx = np.searchsorted(cumulative_weights, 0.5)
    if idx >= len(sorted_values):
        return sorted_values[-1]
    if idx == 0:
        return sorted_values[0]
    c_prev = cumulative_weights[idx - 1]
    c_curr = cumulative_weights[idx]
    if c_curr > c_prev:
        alpha = (0.5 - c_prev) / (c_curr - c_prev)
    else:
        alpha = 0.5
    alpha = np.clip(alpha, 0.0, 1.0)
    return sorted_values[idx - 1] + alpha * (sorted_values[idx] - sorted_values[idx - 1])


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
n = 2000
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
h2 = h1 / np.sqrt(2)
h3 = h1 / 2
h4 = h1 / (2 * np.sqrt(2))  # 0.0354

# Iterations 1-3
m1_noisy = epa_local_median(x, y_noisy, h1)
m1_clean = epa_local_median(x, y_clean, h1)
r2_noisy = y_noisy - m1_noisy
r2_clean = y_clean - m1_clean

m2_noisy = epa_local_median(x, r2_noisy, h2)
m2_clean = epa_local_median(x, r2_clean, h2)
r3_noisy = r2_noisy - m2_noisy
r3_clean = r2_clean - m2_clean

m3_noisy = epa_local_median(x, r3_noisy, h3)
m3_clean = epa_local_median(x, r3_clean, h3)
r4_noisy = r3_noisy - m3_noisy
r4_clean = r3_clean - m3_clean

# Iteration 4
m4_noisy = epa_local_median(x, r4_noisy, h4)
m4_clean = epa_local_median(x, r4_clean, h4)

# =============================================================================
# Plot
# =============================================================================

fig, ax = plt.subplots(figsize=(10, 6))

ax.plot(x, r4_noisy, color=MLGRAY, alpha=0.4, linewidth=0.8, label='Residual (noisy)')
ax.plot(x, r4_clean, color=MLBLUE, linewidth=1.5, alpha=0.7, label='Residual (clean)')
ax.plot(x, m4_noisy, color=MLRED, linewidth=2.5, label=f'IMF estimate (noisy, h={h4:.4f})')
ax.plot(x, m4_clean, color=MLGREEN, linewidth=2, linestyle='--', label=f'IMF estimate (clean, h={h4:.4f})')

ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_title(f'Sifting Iteration 4: h = {h4:.4f} (h1/(2*sqrt(2)))')
ax.legend(loc='upper right', framealpha=0.9)
ax.grid(True, alpha=0.3)
ax.set_xlim(0, 1)

plt.tight_layout()
plt.savefig(Path(__file__).parent / 'chart.pdf', dpi=300, bbox_inches='tight')
plt.close()

print(f"Saved: {Path(__file__).parent / 'chart.pdf'}")
