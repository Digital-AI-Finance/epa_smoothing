"""
EMD Local Median Streamlit Interactive App

Demonstrates the Local Median EMD approach with:
- Interactive controls for smoothing parameters
- Mathematical explanations with LaTeX
- Method comparison (Local Median, Median, Average)
- Real-time metrics computation

Run: streamlit run app.py
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pickle
import gzip
from pathlib import Path
from typing import Optional, Dict, Any

from smoothers import emd_iteration, compute_metrics
from signals import generate_amfm_signal, get_default_signal_params, get_smoothing_defaults


# Page configuration
st.set_page_config(
    page_title="EMD Local Median Smoother",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)


@st.cache_resource
def load_precomputed_data() -> Optional[Dict[str, Any]]:
    """Load pre-computed data from split gzipped files.

    Supports both new split format (precomputed_data/ directory with .pkl.gz files)
    and legacy single file format (precomputed_data.pkl) for backwards compatibility.
    """
    base_dir = Path(__file__).parent
    data_dir = base_dir / 'precomputed_data'

    # Try new split format first
    if data_dir.exists():
        metadata_path = data_dir / 'metadata.pkl.gz'
        if metadata_path.exists():
            # Load metadata
            with gzip.open(metadata_path, 'rb') as f:
                data = pickle.load(f)

            # Load and merge all sigma files
            data['results'] = {}
            for sigma in data['sigma_values']:
                filename = f'sigma_{sigma:.2f}.pkl.gz'
                filepath = data_dir / filename
                if filepath.exists():
                    with gzip.open(filepath, 'rb') as f:
                        data['results'][f'sigma_{sigma}'] = pickle.load(f)

            if data['results']:
                return data

    # Fall back to legacy single file format
    legacy_path = base_dir / 'precomputed_data.pkl'
    if legacy_path.exists():
        with open(legacy_path, 'rb') as f:
            return pickle.load(f)

    return None


def find_nearest_grid_value(value: float, grid: list) -> float:
    """Find nearest value in grid."""
    return min(grid, key=lambda x: abs(x - value))


def get_data_from_cache_or_compute(
    precomputed: Optional[Dict],
    t: np.ndarray,
    y_noisy: np.ndarray,
    y_clean: np.ndarray,
    h0: float,
    sigma: float,
    decay: float,
    k: int,
    method: str
) -> Dict[str, Any]:
    """Get data from cache or compute on-demand."""

    if precomputed is not None:
        # Find nearest grid values
        h0_grid = find_nearest_grid_value(h0, precomputed['h0_values'])
        sigma_grid = find_nearest_grid_value(sigma, precomputed['sigma_values'])
        decay_grid = find_nearest_grid_value(decay, precomputed['decay_values'])

        sigma_key = f"sigma_{sigma_grid}"
        iter_key = f"h0_{h0_grid}_decay_{decay_grid}_{method}"

        if sigma_key in precomputed['results']:
            sigma_data = precomputed['results'][sigma_key]
            if iter_key in sigma_data['iterations']:
                cached = sigma_data['iterations'][iter_key]
                return {
                    'X_list': [np.array(x) for x in cached['X_list']],
                    'X_tilde_list': [np.array(x) for x in cached['X_tilde_list']],
                    'X_list_clean': [np.array(x) for x in cached['X_list_clean']],
                    'X_tilde_list_clean': [np.array(x) for x in cached['X_tilde_list_clean']],
                    'y_clean': np.array(sigma_data['y_clean']),
                    'y_noisy': np.array(sigma_data['y_noisy']),
                    'metrics_noisy': cached['metrics_noisy'],
                    'metrics_clean': cached['metrics_clean'],
                    'from_cache': True
                }

    # Compute on-demand
    X_list, X_tilde_list = emd_iteration(t, y_noisy, h0, decay, k, method)
    X_list_clean, X_tilde_list_clean = emd_iteration(t, y_clean, h0, decay, k, method)

    return {
        'X_list': X_list,
        'X_tilde_list': X_tilde_list,
        'X_list_clean': X_list_clean,
        'X_tilde_list_clean': X_tilde_list_clean,
        'y_clean': y_clean,
        'y_noisy': y_noisy,
        'metrics_noisy': None,
        'metrics_clean': None,
        'from_cache': False
    }


def create_iteration_plot(
    t: np.ndarray,
    data: Dict[str, Any],
    k: int,
    method_name: str
) -> go.Figure:
    """Create 8-panel plot showing iteration k with previous iteration overlay."""

    X_list = data['X_list']
    X_tilde_list = data['X_tilde_list']
    X_list_clean = data['X_list_clean']
    X_tilde_list_clean = data['X_tilde_list_clean']
    y_clean = data['y_clean']
    y_noisy = data['y_noisy']

    # Row 2 title depends on k
    if k > 1:
        row2_title_noisy = f'X<sub>t</sub><sup>({k-1})</sup> (noisy) - Prev Residual'
        row2_title_clean = f'X<sub>t</sub><sup>({k-1})</sup> (clean) - Prev Residual'
    else:
        row2_title_noisy = 'No previous residual (k=1)'
        row2_title_clean = 'No previous residual (k=1)'

    # Create 8-panel subplot (4 rows x 2 cols)
    fig = make_subplots(
        rows=4, cols=2,
        subplot_titles=[
            'X<sub>t</sub> (noisy) - Input',
            'X<sub>t</sub> (clean) - Ground Truth',
            row2_title_noisy,
            row2_title_clean,
            f'X&#771;<sub>t</sub><sup>({k})</sup> (noisy) - Smooth',
            f'X&#771;<sub>t</sub><sup>({k})</sup> (clean) - Smooth',
            f'X<sub>t</sub><sup>({k})</sup> (noisy) - Residual',
            f'X<sub>t</sub><sup>({k})</sup> (clean) - Residual'
        ],
        vertical_spacing=0.08,
        horizontal_spacing=0.08
    )

    # Colors
    color_noisy = '#7f7f7f'  # gray
    color_clean = '#1f77b4'  # blue
    color_smooth_noisy = '#ff7f0e'  # orange
    color_smooth_clean = '#2ca02c'  # green
    color_residual_noisy = '#d62728'  # red
    color_residual_clean = '#9467bd'  # purple

    # Get iteration data (k is 1-indexed, list is 0-indexed)
    k_idx = min(k - 1, len(X_tilde_list) - 1)

    # Row 1: Original signals
    fig.add_trace(
        go.Scatter(x=t, y=y_noisy, mode='lines', name='Noisy',
                   line=dict(color=color_noisy, width=1)),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=t, y=y_clean, mode='lines', name='Clean',
                   line=dict(color=color_clean, width=1.5)),
        row=1, col=2
    )

    # Row 2: Previous iteration residual X^(k-1) = input to iteration k, or flat zero if k=1
    if k > 1 and (k - 1) < len(X_list):
        # Show X^(k-1) - the residual from previous iteration (input to k)
        fig.add_trace(
            go.Scatter(x=t, y=X_list[k - 1], mode='lines', name=f'Residual (k={k-1})',
                       line=dict(color=color_residual_noisy, width=1.5)),
            row=2, col=1
        )
        fig.add_trace(
            go.Scatter(x=t, y=X_list_clean[k - 1], mode='lines', name=f'Residual (k={k-1})',
                       line=dict(color=color_residual_clean, width=1.5)),
            row=2, col=2
        )
    else:
        # k=1: X^(0) = original signal (same as Row 1), show flat zero line
        fig.add_trace(
            go.Scatter(x=t, y=np.zeros_like(t), mode='lines', name='No previous',
                       line=dict(color='#cccccc', width=1, dash='dash')),
            row=2, col=1
        )
        fig.add_trace(
            go.Scatter(x=t, y=np.zeros_like(t), mode='lines', name='No previous',
                       line=dict(color='#cccccc', width=1, dash='dash')),
            row=2, col=2
        )

    # Row 3: Current smooth components
    if k_idx < len(X_tilde_list):
        fig.add_trace(
            go.Scatter(x=t, y=X_tilde_list[k_idx], mode='lines', name='Smooth (noisy)',
                       line=dict(color=color_smooth_noisy, width=2)),
            row=3, col=1
        )
        fig.add_trace(
            go.Scatter(x=t, y=X_tilde_list_clean[k_idx], mode='lines', name='Smooth (clean)',
                       line=dict(color=color_smooth_clean, width=2)),
            row=3, col=2
        )

        # Row 4: Residuals
        fig.add_trace(
            go.Scatter(x=t, y=X_list[k_idx + 1], mode='lines', name='Residual (noisy)',
                       line=dict(color=color_residual_noisy, width=1.5)),
            row=4, col=1
        )
        fig.add_trace(
            go.Scatter(x=t, y=X_list_clean[k_idx + 1], mode='lines', name='Residual (clean)',
                       line=dict(color=color_residual_clean, width=1.5)),
            row=4, col=2
        )

    # Update layout - taller chart for 4 rows
    fig.update_layout(
        height=950,
        showlegend=False,
        title_text=f"{method_name} - Iteration k={k}",
        title_x=0.5,
        margin=dict(l=60, r=50, t=80, b=50)
    )

    # Update axes with Y-axis labels on left column
    for i in range(1, 5):
        for j in range(1, 3):
            fig.update_xaxes(title_text='t' if i == 4 else '', row=i, col=j)
            fig.update_yaxes(title_text='Amplitude' if j == 1 else '', row=i, col=j)

    return fig


def create_reconstruction_plots(
    t: np.ndarray,
    data: Dict[str, Any],
    k: int,
    method_name: str
):
    """Create two reconstruction charts: noisy and clean."""

    X_tilde_list = data['X_tilde_list']
    X_tilde_list_clean = data['X_tilde_list_clean']
    y_clean = data['y_clean']
    y_noisy = data['y_noisy']

    # Compute cumulative sum: S_k = X_tilde^(1) + X_tilde^(2) + ... + X_tilde^(k)
    k_actual = min(k, len(X_tilde_list))
    if k_actual > 0:
        cumsum_noisy = np.sum(X_tilde_list[:k_actual], axis=0)
        cumsum_clean = np.sum(X_tilde_list_clean[:k_actual], axis=0)
    else:
        cumsum_noisy = np.zeros_like(t)
        cumsum_clean = np.zeros_like(t)

    # Chart 1: Noisy Signal Reconstruction
    fig_noisy = go.Figure()
    fig_noisy.add_trace(go.Scatter(
        x=t, y=y_noisy,
        name='Y (noisy)',
        line=dict(color='gray', dash='dash', width=1)
    ))
    fig_noisy.add_trace(go.Scatter(
        x=t, y=cumsum_noisy,
        name=f'Sum k=1..{k}',
        line=dict(color='#ff7f0e', width=2)
    ))
    fig_noisy.update_layout(
        title=dict(text=f"Reconstruction (Noisy)", x=0.5),
        height=300,
        xaxis_title='t',
        yaxis_title='Amplitude',
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='center', x=0.5),
        margin=dict(l=50, r=30, t=60, b=40)
    )

    # Chart 2: Clean Signal Reconstruction
    fig_clean = go.Figure()
    fig_clean.add_trace(go.Scatter(
        x=t, y=y_clean,
        name='Y (clean)',
        line=dict(color='#1f77b4', width=2)
    ))
    fig_clean.add_trace(go.Scatter(
        x=t, y=cumsum_clean,
        name=f'Sum k=1..{k}',
        line=dict(color='#2ca02c', width=2)
    ))
    fig_clean.update_layout(
        title=dict(text=f"Reconstruction (Clean)", x=0.5),
        height=300,
        xaxis_title='t',
        yaxis_title='Amplitude',
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='center', x=0.5),
        margin=dict(l=50, r=30, t=60, b=40)
    )

    return fig_noisy, fig_clean, cumsum_noisy, cumsum_clean


def create_methods_comparison_plot(
    t: np.ndarray,
    all_results: Dict[str, Dict[str, Any]],
    y_clean: np.ndarray,
    k: int
) -> go.Figure:
    """Create overlay plot comparing all 3 methods' smooth components."""

    fig = go.Figure()

    # Add ground truth
    fig.add_trace(go.Scatter(
        x=t, y=y_clean,
        name='Ground Truth',
        line=dict(color='black', width=2, dash='dash'),
        opacity=0.7
    ))

    # Method colors and names
    method_config = {
        'local_median': {'color': '#ff7f0e', 'name': 'Local Median (EPA)'},
        'median': {'color': '#2ca02c', 'name': 'Median (Unweighted)'},
        'average': {'color': '#1f77b4', 'name': 'Average (NW Mean)'}
    }

    k_idx = k - 1

    for method, config in method_config.items():
        if method in all_results:
            data = all_results[method]
            if k_idx < len(data['X_tilde_list']):
                fig.add_trace(go.Scatter(
                    x=t, y=data['X_tilde_list'][k_idx],
                    name=config['name'],
                    line=dict(color=config['color'], width=2)
                ))

    fig.update_layout(
        title=dict(text=f"Methods Comparison - Smooth Component (k={k})", x=0.5),
        height=350,
        xaxis_title='t',
        yaxis_title='Amplitude',
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='center', x=0.5),
        margin=dict(l=50, r=30, t=80, b=40)
    )

    return fig


def create_residual_analysis_plot(
    all_results: Dict[str, Dict[str, Any]],
    k: int
) -> go.Figure:
    """Create histogram of residuals for each method."""

    from plotly.subplots import make_subplots

    methods = ['local_median', 'median', 'average']
    method_names = ['Local Median', 'Median', 'Average']
    colors = ['#ff7f0e', '#2ca02c', '#1f77b4']

    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=method_names,
        horizontal_spacing=0.08
    )

    k_idx = k - 1

    for i, (method, name, color) in enumerate(zip(methods, method_names, colors)):
        if method in all_results:
            data = all_results[method]
            if k_idx < len(data['X_list']) - 1:
                residuals = data['X_list'][k_idx + 1]

                fig.add_trace(
                    go.Histogram(
                        x=residuals,
                        nbinsx=50,
                        marker_color=color,
                        opacity=0.7,
                        showlegend=False
                    ),
                    row=1, col=i+1
                )

                # Add statistics annotation
                mean_r = np.mean(residuals)
                std_r = np.std(residuals)
                fig.add_annotation(
                    x=0.5, y=0.95,
                    xref=f'x{i+1} domain', yref=f'y{i+1} domain',
                    text=f'mean={mean_r:.3f}<br>std={std_r:.3f}',
                    showarrow=False,
                    font=dict(size=10),
                    bgcolor='rgba(255,255,255,0.8)'
                )

    fig.update_layout(
        title=dict(text=f"Residual Distribution (k={k})", x=0.5),
        height=300,
        margin=dict(l=50, r=30, t=80, b=40)
    )

    for i in range(1, 4):
        fig.update_xaxes(title_text='Residual Value', row=1, col=i)
        fig.update_yaxes(title_text='Count' if i == 1 else '', row=1, col=i)

    return fig


def create_bandwidth_decay_plot(
    h0: float,
    decay: float,
    current_k: int
) -> go.Figure:
    """Create visualization of bandwidth decay h_k = h0 / decay^(k-1)."""

    k_values = np.arange(1, 11)
    h_values = h0 / (decay ** (k_values - 1))

    fig = go.Figure()

    # Line plot
    fig.add_trace(go.Scatter(
        x=k_values,
        y=h_values,
        mode='lines+markers',
        name='h_k',
        line=dict(color='#1f77b4', width=2),
        marker=dict(size=8, color='#1f77b4')
    ))

    # Highlight current k
    current_h = h0 / (decay ** (current_k - 1))
    fig.add_trace(go.Scatter(
        x=[current_k],
        y=[current_h],
        mode='markers',
        name=f'Current (k={current_k})',
        marker=dict(size=15, color='#ff7f0e', symbol='star', line=dict(width=2, color='black'))
    ))

    # Add formula annotation
    fig.add_annotation(
        x=0.95, y=0.95,
        xref='paper', yref='paper',
        text=f'h_k = {h0:.2f} / {decay:.3f}^(k-1)',
        showarrow=False,
        font=dict(size=12),
        bgcolor='rgba(255,255,255,0.9)',
        bordercolor='gray',
        borderwidth=1
    )

    fig.update_layout(
        title=dict(text="Bandwidth Decay", x=0.5),
        height=300,
        xaxis_title='Iteration k',
        yaxis_title='Bandwidth h_k',
        xaxis=dict(tickmode='linear', tick0=1, dtick=1),
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='center', x=0.5),
        margin=dict(l=50, r=30, t=80, b=40)
    )

    return fig


def render_theory_tab():
    """Render the Theory & Math tab with full theoretical exposition."""

    st.header("Mathematical Foundation of EMD Local Median Smoothing")

    # Section 1: AM-FM Signal Model
    with st.expander("1. AM-FM Signal Model", expanded=True):
        st.markdown("""
        **Amplitude-Frequency Modulated (AM-FM) signals** are fundamental in signal processing,
        representing signals whose amplitude and instantaneous frequency vary over time.
        """)

        st.markdown("**General Form:**")
        st.latex(r"Y(t) = a(t) \cdot \cos(\phi(t)) + \varepsilon(t)")

        st.markdown("where:")
        st.markdown("""
        - $a(t)$ is the **instantaneous amplitude** (amplitude modulation)
        - $\\phi(t)$ is the **instantaneous phase** (frequency modulation)
        - $\\varepsilon(t)$ is additive noise
        """)

        st.markdown("**Our Implementation:**")
        st.latex(r"a(t) = 1 + d \cdot \sin(4\pi t)")
        st.latex(r"\phi(t) = 2\pi(f_0 t + c t^2)")

        st.markdown("""
        The chirp rate $c$ causes the instantaneous frequency to increase linearly with time:
        """)
        st.latex(r"f_{inst}(t) = \frac{1}{2\pi}\frac{d\phi}{dt} = f_0 + 2ct")

        st.markdown("**Noise Model:**")
        st.latex(r"y(t) = y_{clean}(t) + \sigma \cdot \varepsilon(t), \quad \varepsilon \sim \mathcal{N}(0,1)")

    # Section 2: Epanechnikov Kernel
    with st.expander("2. Epanechnikov Kernel", expanded=True):
        st.markdown("""
        The **Epanechnikov kernel** (also called the parabolic kernel) is optimal in the sense
        of minimizing the asymptotic mean integrated squared error (AMISE) among all
        second-order kernels.
        """)

        st.markdown("**Definition:**")
        st.latex(r"K(u) = \frac{3}{4}(1 - u^2) \cdot \mathbf{1}_{|u| \leq 1}")

        st.markdown("**Key Properties:**")
        st.markdown("""
        1. **Compact support**: $K(u) = 0$ for $|u| > 1$
        2. **Non-negative**: $K(u) \\geq 0$ for all $u$
        3. **Symmetric**: $K(-u) = K(u)$
        4. **Normalized**: $\\int_{-1}^{1} K(u) du = 1$
        5. **AMISE optimal**: Minimizes mean integrated squared error
        """)

        st.markdown("**Scaled Kernel:**")
        st.latex(r"K_h(u) = \frac{1}{h} K\left(\frac{u}{h}\right) = \frac{3}{4h}\left(1 - \frac{u^2}{h^2}\right) \cdot \mathbf{1}_{|u| \leq h}")

        st.markdown("""
        The bandwidth $h$ controls the **effective window size**:
        - Larger $h$ $\\rightarrow$ more smoothing, captures global trends
        - Smaller $h$ $\\rightarrow$ less smoothing, captures local details
        """)

        st.markdown("**AMISE Optimality Derivation:**")
        st.markdown("""
        For kernel density estimation, the AMISE is:
        """)
        st.latex(r"\text{AMISE}(h) = \frac{R(K)}{nh} + \frac{h^4 \mu_2(K)^2 R(f'')}{4}")
        st.markdown("""
        where $R(K) = \\int K(u)^2 du$ and $\\mu_2(K) = \\int u^2 K(u) du$.

        The Epanechnikov kernel minimizes $R(K)$ subject to $\\mu_2(K) = 1$,
        making it the **optimal choice** for minimizing estimation error.
        """)

    # Section 3: Weighted Median Estimator
    with st.expander("3. Weighted Median Estimator", expanded=True):
        st.markdown("""
        The **weighted median** extends the classical median to incorporate importance weights,
        providing a robust measure of central tendency.
        """)

        st.markdown("**Definition (Variational Form):**")
        st.latex(r"\tilde{X}^{(k)}_t = \arg\min_m \sum_s |X^{(k)}_s - m| \cdot K_h(s - t)")

        st.markdown("""
        This minimizes the **weighted sum of absolute deviations**, analogous to how the
        ordinary median minimizes $\\sum |x_i - m|$.
        """)

        st.markdown("**Comparison with Weighted Mean (Nadaraya-Watson):**")

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Weighted Median:**")
            st.latex(r"\tilde{m}(t) = \arg\min_m \sum_s |Y_s - m| \cdot K_h(s-t)")
            st.markdown("Robust to outliers")
        with col2:
            st.markdown("**Weighted Mean (NW):**")
            st.latex(r"\hat{m}(t) = \frac{\sum_s Y_s \cdot K_h(s-t)}{\sum_s K_h(s-t)}")
            st.markdown("Optimal for Gaussian noise")

        st.markdown("**Robustness Properties:**")
        st.markdown("""
        - **Breakdown Point**: The weighted median has a breakdown point of 50%,
          meaning up to 50% of the data can be corrupted before the estimate becomes unbounded.
        - **Influence Function**: The influence function of the median is bounded,
          unlike the mean whose influence function is unbounded.
        """)

        st.latex(r"\text{IF}(x; \text{median}) = \text{sign}(x - \text{median})")
        st.latex(r"\text{IF}(x; \text{mean}) = x - \text{mean} \quad \text{(unbounded)}")

    # Section 4: EMD Iteration Procedure
    with st.expander("4. EMD Iteration Procedure", expanded=True):
        st.markdown("""
        **Empirical Mode Decomposition (EMD)** iteratively extracts oscillatory components
        called **Intrinsic Mode Functions (IMFs)** from a signal.
        """)

        st.markdown("**Algorithm:**")
        st.markdown("""
        1. **Initialize**: $X^{(0)}_t = Y_t$ (noisy signal)
        2. **For** $k = 1, 2, 3, \\ldots$:
           - Compute smooth component: $\\tilde{X}^{(k)}_t = \\text{LocalMedian}(X^{(k-1)}, h_k)$
           - Compute residual: $X^{(k)}_t = X^{(k-1)}_t - \\tilde{X}^{(k)}_t$
           - Update bandwidth: $h_{k+1} = h_k / a$
        """)

        st.markdown("**Bandwidth Decay:**")
        st.latex(r"h_k = \frac{h_0}{a^{k-1}}")

        st.markdown("""
        The decay factor $a > 1$ (typically $a = \\sqrt{2} \\approx 1.414$) causes the bandwidth
        to decrease exponentially, allowing:
        - **Early iterations**: Extract coarse/smooth components (large $h$)
        - **Later iterations**: Extract fine/oscillatory components (small $h$)
        """)

        st.markdown("**Bandwidth Sequence Example** ($h_0 = 0.1$, $a = \\sqrt{2}$):")
        st.latex(r"h_1 = 0.100, \quad h_2 = 0.071, \quad h_3 = 0.050, \quad h_4 = 0.035, \ldots")

    # Section 5: Reconstruction Theorem
    with st.expander("5. Reconstruction Theorem", expanded=True):
        st.markdown("""
        A fundamental property of EMD is that the original signal can be **perfectly reconstructed**
        from the extracted components.
        """)

        st.markdown("**Theorem (Signal Reconstruction):**")
        st.latex(r"X^{(0)}_t = \sum_{k=1}^{K} \tilde{X}^{(k)}_t + X^{(K)}_t")

        st.markdown("**Proof by Telescoping Sum:**")
        st.markdown("""
        From the iteration formula $X^{(k)}_t = X^{(k-1)}_t - \\tilde{X}^{(k)}_t$, we have:
        """)
        st.latex(r"\tilde{X}^{(k)}_t = X^{(k-1)}_t - X^{(k)}_t")

        st.markdown("Summing from $k=1$ to $K$:")
        st.latex(r"""
        \sum_{k=1}^{K} \tilde{X}^{(k)}_t = \sum_{k=1}^{K} \left(X^{(k-1)}_t - X^{(k)}_t\right)
        """)
        st.latex(r"""
        = (X^{(0)}_t - X^{(1)}_t) + (X^{(1)}_t - X^{(2)}_t) + \cdots + (X^{(K-1)}_t - X^{(K)}_t)
        """)
        st.latex(r"""
        = X^{(0)}_t - X^{(K)}_t
        """)

        st.markdown("Therefore:")
        st.latex(r"X^{(0)}_t = \sum_{k=1}^{K} \tilde{X}^{(k)}_t + X^{(K)}_t \quad \blacksquare")

        st.markdown("""
        **Interpretation:**
        - Each $\\tilde{X}^{(k)}_t$ is an **IMF** (Intrinsic Mode Function)
        - $X^{(K)}_t$ is the **final residual** (trend or DC component)
        - As $K \\to \\infty$, the residual $X^{(K)}_t \\to 0$
        """)

    # Section 6: Convergence Analysis
    with st.expander("6. Convergence Analysis", expanded=True):
        st.markdown("""
        **Convergence** of the EMD procedure depends on the bandwidth decay and signal properties.
        """)

        st.markdown("**Stopping Criteria:**")
        st.markdown("""
        Common stopping criteria include:
        1. **Fixed iterations**: Stop after $K_{max}$ iterations
        2. **Residual energy**: Stop when $\\|X^{(K)}\\|_2 < \\epsilon$
        3. **Bandwidth threshold**: Stop when $h_K < h_{min}$
        """)

        st.markdown("**Bandwidth Selection Theory:**")
        st.markdown("""
        The optimal bandwidth balances **bias** and **variance**:
        - **Bias**: Large bandwidth over-smooths, missing local features
        - **Variance**: Small bandwidth under-smooths, fitting noise
        """)
        st.latex(r"\text{MSE}(h) = \text{Bias}^2(h) + \text{Variance}(h)")

        st.markdown("**Asymptotic Optimal Bandwidth:**")
        st.latex(r"h_{opt} \propto n^{-1/5}")
        st.markdown("""
        For $n$ data points, this gives the classical rate of convergence $O(n^{-4/5})$ for
        nonparametric regression.
        """)

        st.markdown("**Decay Factor Interpretation:**")
        st.markdown("""
        - $a = 1$: No decay, constant bandwidth (not EMD)
        - $a = \\sqrt{2} \\approx 1.414$: **Standard choice**, halves bandwidth every 2 iterations
        - $a = 2$: Aggressive decay, halves bandwidth every iteration
        - $a < 1$: Bandwidth grows (unusual, for exploration only)
        """)

        st.markdown("**Convergence Rate:**")
        st.latex(r"h_k = h_0 \cdot a^{-(k-1)} \to 0 \quad \text{as } k \to \infty \text{ (for } a > 1\text{)}")

    # Section 7: Method Comparison Summary
    with st.expander("7. Method Comparison Summary", expanded=False):
        st.markdown("**Three Smoothing Methods:**")

        comparison_data = {
            'Property': ['Estimator Type', 'Robustness', 'Optimality', 'Breakdown Point', 'Computational Cost'],
            'Local Median (EPA)': ['Weighted L1', 'High', 'Robust regression', '50%', 'O(n log n)'],
            'Median (Unweighted)': ['Unweighted L1', 'High', 'None', '50%', 'O(n log n)'],
            'Average (NW Mean)': ['Weighted L2', 'Low', 'Gaussian noise', '0%', 'O(n)']
        }
        import pandas as pd
        df = pd.DataFrame(comparison_data)
        st.dataframe(df, hide_index=True)

        st.markdown("""
        **When to use each method:**
        - **Local Median**: Default choice for real-world data with potential outliers
        - **Median**: When you want simplicity without kernel weighting
        - **Average (NW)**: When you're confident the noise is Gaussian with no outliers
        """)


def main():
    """Main Streamlit app."""

    # Load pre-computed data
    precomputed = load_precomputed_data()

    # Sidebar - Parameters
    st.sidebar.title("EMD Local Median")
    st.sidebar.markdown("---")

    # Smoothing parameters section
    st.sidebar.subheader("Smoothing Parameters")

    defaults = get_smoothing_defaults()

    # Use select_slider to match exact precomputed grid values for cache hits
    h0_options = [0.03, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50]
    h0 = st.sidebar.select_slider(
        "h₀ (initial bandwidth)",
        options=h0_options,
        value=0.10 if 0.10 in h0_options else h0_options[2],
        format_func=lambda x: f"{x:.2f}",
        help="Initial smoothing bandwidth. Smaller = less smoothing"
    )

    sigma_options = [0.01, 0.04, 0.10, 0.20, 0.30, 0.50]
    sigma = st.sidebar.select_slider(
        "σ (noise level)",
        options=sigma_options,
        value=0.20 if 0.20 in sigma_options else sigma_options[1],
        format_func=lambda x: f"{x:.2f}",
        help="Standard deviation of Gaussian noise"
    )

    decay_options = [0.8, 1.0, 1.2, 1.414, 1.7, 2.0]
    decay = st.sidebar.select_slider(
        "a (decay factor)",
        options=decay_options,
        value=1.414 if 1.414 in decay_options else decay_options[1],
        format_func=lambda x: f"{x:.3f}" if x == 1.414 else f"{x:.1f}",
        help="Bandwidth decay factor. Default √2 ≈ 1.414"
    )

    k = st.sidebar.radio(
        "k (iteration)",
        options=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        index=0,
        horizontal=True,
        help="EMD iteration number"
    )

    st.sidebar.markdown("---")

    # AM-FM Signal parameters
    st.sidebar.subheader("AM-FM Signal")

    signal_defaults = get_default_signal_params()

    am_depth = st.sidebar.slider(
        "AM depth (d)",
        min_value=0.1,
        max_value=1.0,
        value=signal_defaults['am_depth'],
        step=0.1,
        help="Amplitude modulation depth"
    )

    base_freq = st.sidebar.slider(
        "Base freq (f₀)",
        min_value=1,
        max_value=20,
        value=int(signal_defaults['base_freq']),
        step=1,
        help="Base frequency in Hz"
    )

    chirp_rate = st.sidebar.slider(
        "Chirp rate (c)",
        min_value=0,
        max_value=30,
        value=int(signal_defaults['chirp_rate']),
        step=1,
        help="Frequency chirp coefficient"
    )

    st.sidebar.markdown("---")

    # Mathematical formulas section
    st.sidebar.subheader("Mathematical Formulas")

    with st.sidebar.expander("AM-FM Signal", expanded=False):
        st.latex(r"y_{clean}(t) = [1 + d \sin(4\pi t)] \cdot \sin(2\pi(f_0 t + c t^2))")
        st.latex(r"y(t) = y_{clean}(t) + \sigma \cdot \varepsilon(t)")
        st.markdown("where $\\varepsilon \\sim \\mathcal{N}(0,1)$")

    with st.sidebar.expander("Epanechnikov Kernel", expanded=False):
        st.latex(r"K(u) = \frac{3}{4}(1 - u^2) \cdot \mathbf{1}_{|u| \leq 1}")
        st.latex(r"K_h(u) = \frac{1}{h} K\left(\frac{u}{h}\right)")

    with st.sidebar.expander("Weighted Median", expanded=False):
        st.latex(r"\tilde{X}^{(k)}_t = \arg\min_m \sum_s |X^{(k)}_s - m| \cdot K_h(s - t)")

    with st.sidebar.expander("Iteration Procedure", expanded=False):
        st.latex(r"X^{(k+1)}_t = X^{(k)}_t - \tilde{X}^{(k)}_t")

    with st.sidebar.expander("Bandwidth Decay", expanded=False):
        st.latex(r"h_{k+1} = \frac{h_k}{a}, \quad a = \sqrt{2} \approx 1.414")
        st.latex(r"h_k = \frac{h_0}{a^{k-1}}")

    # Current bandwidth display
    h_k = h0 / (decay ** (k - 1))
    st.sidebar.markdown("---")
    st.sidebar.metric("Current h_k", f"{h_k:.4f}")

    # Generate signal
    t, y_clean, y_noisy = generate_amfm_signal(
        n_points=2000,
        am_depth=am_depth,
        base_freq=float(base_freq),
        chirp_rate=float(chirp_rate),
        noise_sigma=sigma,
        seed=42
    )

    # Main content area
    st.title("EMD Local Median Smoother")
    st.markdown("**Empirical Mode Decomposition with Kernel-Weighted Local Median**")

    # Method tabs (including Theory & Math)
    tabs = st.tabs(["Local Median", "Median", "Average", "Theory & Math"])

    methods = ['local_median', 'median', 'average']
    method_names = ['Local Median (EPA Weighted)', 'Median (Unweighted)', 'Average (NW Mean)']

    # Store results for metrics comparison
    all_results = {}

    # Process method tabs (first 3 tabs)
    for i, (method, method_name) in enumerate(zip(methods, method_names)):
        with tabs[i]:
            # Get or compute data
            data = get_data_from_cache_or_compute(
                precomputed, t, y_noisy, y_clean,
                h0, sigma, decay, k, method
            )

            all_results[method] = data

            # Create and display 8-panel iteration plot
            fig = create_iteration_plot(t, data, k, method_name)
            st.plotly_chart(fig, use_container_width=True)

            # Cache status indicator
            if data['from_cache']:
                st.caption("Using pre-computed data (instant)")
            else:
                st.info(f"Computing {method_name.split('(')[0].strip()} on-demand (not in cache)")

            # Reconstruction Sum Charts (side by side)
            st.markdown("---")
            st.markdown("### Reconstruction Sum")
            fig_noisy, fig_clean, cumsum_noisy, cumsum_clean = create_reconstruction_plots(t, data, k, method_name)

            col1, col2 = st.columns(2)
            with col1:
                st.plotly_chart(fig_noisy, use_container_width=True)
            with col2:
                st.plotly_chart(fig_clean, use_container_width=True)

            # Reconstruction metrics
            rmse_recon_noisy = np.sqrt(np.mean((cumsum_noisy - y_clean) ** 2))
            rmse_recon_clean = np.sqrt(np.mean((cumsum_clean - y_clean) ** 2))
            residual_norm = np.linalg.norm(y_noisy - cumsum_noisy)

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("RMSE (noisy recon)", f"{rmse_recon_noisy:.4f}")
            with col2:
                st.metric("RMSE (clean recon)", f"{rmse_recon_clean:.4f}")
            with col3:
                st.metric("Residual ||Y - S_k||", f"{residual_norm:.4f}")

    # Theory & Math Tab (4th tab)
    with tabs[3]:
        render_theory_tab()

    # Metrics comparison table
    st.markdown("---")
    st.subheader("Metrics Comparison")

    # Compute metrics for current iteration
    metrics_data = []
    # Dynamic index bound based on actual data length
    k_idx = min(k - 1, len(all_results[methods[0]]['X_tilde_list']) - 1)

    for method, method_name in zip(methods, method_names):
        data = all_results[method]

        if k_idx < len(data['X_tilde_list']):
            rmse_noisy, mae_noisy = compute_metrics(data['X_tilde_list'][k_idx], y_clean)
            rmse_clean, mae_clean = compute_metrics(data['X_tilde_list_clean'][k_idx], y_clean)
        else:
            rmse_noisy, mae_noisy = np.nan, np.nan
            rmse_clean, mae_clean = np.nan, np.nan

        metrics_data.append({
            'Method': method_name.split('(')[0].strip(),
            'RMSE (noisy)': rmse_noisy,
            'RMSE (clean)': rmse_clean,
            'MAE (noisy)': mae_noisy,
            'MAE (clean)': mae_clean
        })

    # Display as styled dataframe with best values highlighted
    df = pd.DataFrame(metrics_data)
    numeric_cols = ['RMSE (noisy)', 'RMSE (clean)', 'MAE (noisy)', 'MAE (clean)']

    # Style: highlight minimum (best) values in each column, format to 4 decimals
    styled_df = df.style.highlight_min(
        axis=0,
        subset=numeric_cols,
        props='background-color: #90EE90; font-weight: bold'
    ).format({col: '{:.4f}' for col in numeric_cols})

    col1, col2, col3 = st.columns([1, 4, 1])
    with col2:
        st.dataframe(styled_df, use_container_width=True, hide_index=True)

    # Additional Analysis Section
    st.markdown("---")
    st.subheader("Additional Analysis")

    # Methods Comparison Plot (full width)
    fig_comparison = create_methods_comparison_plot(t, all_results, y_clean, k)
    st.plotly_chart(fig_comparison, use_container_width=True)

    # Residual Analysis and Bandwidth Decay (side by side)
    col1, col2 = st.columns(2)

    with col1:
        fig_residuals = create_residual_analysis_plot(all_results, k)
        st.plotly_chart(fig_residuals, use_container_width=True)

    with col2:
        fig_bandwidth = create_bandwidth_decay_plot(h0, decay, k)
        st.plotly_chart(fig_bandwidth, use_container_width=True)

    # Additional info
    st.markdown("---")
    st.markdown("""
    **Key Insights:**
    - **Local Median**: Uses Epanechnikov kernel weights - robust to outliers
    - **Median**: Unweighted median within window - also robust but no smooth weighting
    - **Average**: Nadaraya-Watson kernel mean - optimal for Gaussian noise but sensitive to outliers

    **Iteration Interpretation:**
    - $X^{(k)}$: Input signal at iteration $k$
    - $\\tilde{X}^{(k)}$: Smooth component extracted at iteration $k$
    - $X^{(k+1)} = X^{(k)} - \\tilde{X}^{(k)}$: Residual (input for next iteration)
    """)

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
    <small>
    EMD Local Median Approach |
    <a href='https://digital-ai-finance.github.io/emd_local_median/' target='_blank'>Theory</a> |
    <a href='https://digital-ai-finance.github.io/epa_smoothing/' target='_blank'>Dashboard</a>
    </small>
    </div>
    """, unsafe_allow_html=True)


if __name__ == '__main__':
    main()
