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
                # Plotly xref: 'x domain' for first, 'x2 domain' for second, etc.
                xref = 'x domain' if i == 0 else f'x{i+1} domain'
                yref = 'y domain' if i == 0 else f'y{i+1} domain'
                fig.add_annotation(
                    x=0.5, y=0.95,
                    xref=xref, yref=yref,
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


def create_kernel_window_figure(t_example, y_example, t_i, h, idx_i):
    """Create Plotly figure showing kernel window at point t_i."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    # Find points within kernel window
    u = (t_example - t_i) / h
    weights = 0.75 * (1 - u**2) * (np.abs(u) <= 1)
    in_window = np.abs(u) <= 1

    fig = make_subplots(
        rows=2, cols=1,
        row_heights=[0.6, 0.4],
        subplot_titles=['Signal with Kernel Window', 'Kernel Weights'],
        vertical_spacing=0.15
    )

    # Top: Signal with window highlight
    fig.add_trace(go.Scatter(
        x=t_example, y=y_example,
        mode='lines+markers',
        marker=dict(size=8, color='gray'),
        line=dict(color='gray', width=1),
        name='Signal'
    ), row=1, col=1)

    # Highlight points in window
    fig.add_trace(go.Scatter(
        x=t_example[in_window], y=y_example[in_window],
        mode='markers',
        marker=dict(size=12, color='blue', symbol='circle'),
        name='In Window'
    ), row=1, col=1)

    # Current evaluation point
    fig.add_trace(go.Scatter(
        x=[t_i], y=[y_example[idx_i]],
        mode='markers',
        marker=dict(size=16, color='red', symbol='star'),
        name=f't_i = {t_i:.1f}'
    ), row=1, col=1)

    # Window boundaries
    fig.add_vline(x=t_i - h, line=dict(color='orange', dash='dash'), row=1, col=1)
    fig.add_vline(x=t_i + h, line=dict(color='orange', dash='dash'), row=1, col=1)

    # Bottom: Kernel weights
    colors = ['blue' if w > 0 else 'lightgray' for w in weights]
    fig.add_trace(go.Bar(
        x=t_example, y=weights,
        marker_color=colors,
        name='Kernel Weights',
        showlegend=False
    ), row=2, col=1)

    fig.update_layout(
        height=450,
        showlegend=True,
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='center', x=0.5),
        margin=dict(l=50, r=30, t=60, b=40)
    )

    fig.update_xaxes(title_text='t', row=2, col=1)
    fig.update_yaxes(title_text='y', row=1, col=1)
    fig.update_yaxes(title_text='K(u)', row=2, col=1)

    return fig


def create_weighted_median_figure(y_sorted, weights_sorted, cumsum_norm):
    """Create Plotly figure showing weighted median computation."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=['Sorted Values with Weights', 'Cumulative Weight'],
        horizontal_spacing=0.12
    )

    n = len(y_sorted)
    x_positions = list(range(n))

    # Left: Bar chart of weights for each sorted y value
    fig.add_trace(go.Bar(
        x=x_positions, y=weights_sorted,
        marker_color='steelblue',
        text=[f'{w:.3f}' for w in weights_sorted],
        textposition='outside',
        name='Weights',
        showlegend=False
    ), row=1, col=1)

    # Right: Cumulative weight line
    fig.add_trace(go.Scatter(
        x=x_positions, y=cumsum_norm,
        mode='lines+markers',
        line=dict(color='green', width=2),
        marker=dict(size=8, color='green'),
        name='Cumulative',
        showlegend=False
    ), row=1, col=2)

    # 0.5 threshold line
    fig.add_hline(y=0.5, line=dict(color='red', dash='dash', width=2), row=1, col=2)

    # Find crossing point
    idx = np.searchsorted(cumsum_norm, 0.5)
    if 0 < idx < len(cumsum_norm):
        # Interpolation point
        c_prev = cumsum_norm[idx - 1]
        c_curr = cumsum_norm[idx]
        if c_curr > c_prev:
            alpha = (0.5 - c_prev) / (c_curr - c_prev)
            x_cross = idx - 1 + alpha
            y_cross = 0.5
            fig.add_trace(go.Scatter(
                x=[x_cross], y=[y_cross],
                mode='markers',
                marker=dict(size=14, color='red', symbol='x'),
                name='Median Point',
                showlegend=False
            ), row=1, col=2)

    fig.update_layout(
        height=300,
        margin=dict(l=50, r=30, t=60, b=40)
    )

    # X-axis labels as y values
    fig.update_xaxes(
        tickmode='array',
        tickvals=x_positions,
        ticktext=[f'{y:.2f}' for y in y_sorted],
        title_text='y (sorted)',
        row=1, col=1
    )
    fig.update_xaxes(
        tickmode='array',
        tickvals=x_positions,
        ticktext=[f'{y:.2f}' for y in y_sorted],
        title_text='y (sorted)',
        row=1, col=2
    )
    fig.update_yaxes(title_text='Weight', row=1, col=1)
    fig.update_yaxes(title_text='Cumulative Weight', row=1, col=2)

    return fig


def compute_example_walkthrough(t, y, t_i, h, idx_i):
    """Compute and return all intermediate values for display."""
    # Step 1: Compute scaled distances
    u = (t - t_i) / h

    # Step 2: Compute kernel weights
    weights = 0.75 * (1 - u**2) * (np.abs(u) <= 1)

    # Step 3: Sort by y values
    sorted_indices = np.argsort(y)
    y_sorted = y[sorted_indices]
    weights_sorted = weights[sorted_indices]
    u_sorted = u[sorted_indices]
    t_sorted = t[sorted_indices]

    # Step 4: Cumulative normalized weights
    total_weight = np.sum(weights_sorted)
    if total_weight > 0:
        cumsum_norm = np.cumsum(weights_sorted) / total_weight
    else:
        cumsum_norm = np.zeros_like(weights_sorted)

    # Step 5: Find median via 0.5 crossing
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
        'sorted_indices': sorted_indices,
        'y_sorted': y_sorted,
        'weights_sorted': weights_sorted,
        'u_sorted': u_sorted,
        't_sorted': t_sorted,
        'cumsum_norm': cumsum_norm,
        'median_idx': idx,
        'alpha': alpha,
        'median_result': median_result,
        'total_weight': total_weight
    }


def render_computation_tab():
    """Render the 'How It Works' tab with detailed computation explanation."""

    st.header("How the EPA Local Median Smoother Works")

    st.markdown("""
    This tab provides an **ultra-detailed** walkthrough of how the EPA kernel-weighted
    local median smoother computes the smoothed signal. We cover:
    1. The algorithm step-by-step
    2. A numerical worked example
    3. Visual concept diagrams
    4. Key insights and takeaways
    """)

    # ========== SECTION 1: Algorithm Walkthrough ==========
    with st.expander("1. Algorithm Walkthrough", expanded=True):
        st.markdown("### 1.1 High-Level Overview")

        st.code("""
Input:  t[1..n]     - time array (equally or unequally spaced)
        y[1..n]     - signal values
        h           - bandwidth (kernel window half-width)
Output: y_smooth[1..n] - smoothed signal

For each point i from 1 to n:
    1. Compute scaled distances: u[j] = (t[j] - t[i]) / h  for all j
    2. Apply Epanechnikov kernel: w[j] = K(u[j])
    3. Compute weighted median: y_smooth[i] = WeightedMedian(y, w)
Return y_smooth
        """, language="text")

        st.markdown("### 1.2 Detailed Pseudocode with Comments")

        st.code("""
function epa_local_median(t, y, h):
    n = length(t)
    y_smooth = zeros(n)

    for i = 1 to n:
        # ---- Step 1: Compute scaled distances ----
        # For each data point j, compute how far it is from t[i]
        # scaled by the bandwidth h
        for j = 1 to n:
            u[j] = (t[j] - t[i]) / h

        # ---- Step 2: Apply Epanechnikov kernel ----
        # K(u) = 0.75 * (1 - u^2)  if |u| <= 1
        # K(u) = 0                 otherwise
        for j = 1 to n:
            if |u[j]| <= 1:
                w[j] = 0.75 * (1 - u[j]^2)
            else:
                w[j] = 0    # Outside kernel support

        # ---- Step 3: Compute weighted median ----
        y_smooth[i] = weighted_median(y, w)

    return y_smooth
        """, language="text")

        st.markdown("### 1.3 Weighted Median Algorithm")

        st.markdown("""
        The weighted median is the value $m$ that minimizes:
        """)
        st.latex(r"\sum_{j=1}^{n} w_j \cdot |y_j - m|")

        st.markdown("""
        **Computation via cumulative weights:**
        """)

        st.code("""
function weighted_median(y, w):
    # ---- Step 1: Sort by y values ----
    indices = argsort(y)           # Get indices that would sort y
    y_sorted = y[indices]          # Sorted y values
    w_sorted = w[indices]          # Weights in same order

    # ---- Step 2: Normalize and cumulate weights ----
    total = sum(w_sorted)
    if total == 0:
        return median(y)           # Fallback to unweighted

    cumsum = cumulative_sum(w_sorted) / total
    # cumsum[k] = (w_1 + w_2 + ... + w_k) / total

    # ---- Step 3: Find 0.5 crossing point ----
    # Weighted median is where cumulative weight first reaches 0.5
    k = first index where cumsum[k] >= 0.5

    # ---- Step 4: Linear interpolation for smooth output ----
    if k == 0:
        return y_sorted[0]
    else:
        C_prev = cumsum[k-1]
        C_curr = cumsum[k]
        alpha = (0.5 - C_prev) / (C_curr - C_prev)
        return y_sorted[k-1] + alpha * (y_sorted[k] - y_sorted[k-1])
        """, language="text")

        st.markdown("### 1.4 Complexity Analysis")

        st.markdown("""
        | Operation | Cost per point | Total |
        |-----------|----------------|-------|
        | Distance computation | O(n) | O(n^2) |
        | Kernel evaluation | O(n) | O(n^2) |
        | Sorting for median | O(n log n) | O(n^2 log n) |
        | Cumulative sum | O(n) | O(n^2) |
        | **Overall** | **O(n log n)** | **O(n^2 log n)** |

        For n=2000 points, this is approximately 2000 * 2000 * 11 = 44 million operations.
        In practice, we can optimize by only considering points where |u| <= 1 (about 2h/range * n points).
        """)

    # ========== SECTION 2: Numerical Worked Example ==========
    with st.expander("2. Numerical Worked Example", expanded=True):
        st.markdown("""
        ### Setup
        We use a small example with **n=10 points** for clarity.
        """)

        # Create example data
        np.random.seed(42)
        t_example = np.linspace(0, 0.9, 10)
        y_clean_example = np.sin(2 * np.pi * t_example)
        y_example = y_clean_example + 0.1 * np.random.randn(10)
        y_example[5] = 3.0  # Inject outlier at t=0.5
        h_example = 0.25

        # Display setup
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Parameters:**")
            st.markdown(f"- n = 10 points")
            st.markdown(f"- t = [0.0, 0.1, 0.2, ..., 0.9]")
            st.markdown(f"- h = {h_example} (bandwidth)")
            st.markdown(f"- **Outlier injected at t=0.5** (y=3.0)")

        with col2:
            st.markdown("**Data:**")
            data_df = pd.DataFrame({
                'j': range(10),
                't_j': [f'{t:.1f}' for t in t_example],
                'y_j': [f'{y:.3f}' for y in y_example],
                'Note': ['', '', '', '', '', 'OUTLIER', '', '', '', '']
            })
            st.dataframe(data_df, hide_index=True, height=200)

        st.markdown("---")

        # Point A: t_i = 0.0 (boundary)
        st.markdown("### Point A: Evaluation at t_i = 0.0 (Left Boundary)")

        result_A = compute_example_walkthrough(t_example, y_example, 0.0, h_example, 0)

        st.markdown(f"""
        **Step 1: Identify neighbors in window** [t_i - h, t_i + h] = [-0.25, 0.25]

        Points in window: t = 0.0, 0.1, 0.2 (indices 0, 1, 2)
        """)

        # Show computation table for Point A
        in_window_A = np.abs(result_A['u']) <= 1
        table_A = pd.DataFrame({
            'j': range(10),
            't_j': [f'{t:.1f}' for t in t_example],
            'y_j': [f'{y:.3f}' for y in y_example],
            'u_j': [f'{u:.2f}' for u in result_A['u']],
            'In Window': ['Yes' if iw else 'No' for iw in in_window_A],
            'w_j = K(u_j)': [f'{w:.4f}' for w in result_A['weights']]
        })
        st.dataframe(table_A, hide_index=True)

        st.markdown("**Step 2: Sort by y values and compute cumulative weights**")

        # Only show points with non-zero weights for clarity
        active_idx = result_A['weights'] > 0
        n_active = np.sum(active_idx)

        sort_table_A = pd.DataFrame({
            'Rank': range(1, n_active + 1),
            'y_(k)': [f'{y:.3f}' for y in result_A['y_sorted'][result_A['weights_sorted'] > 0]],
            'w_(k)': [f'{w:.4f}' for w in result_A['weights_sorted'][result_A['weights_sorted'] > 0]],
            'C_k (cumsum)': [f'{c:.4f}' for c in result_A['cumsum_norm'][result_A['weights_sorted'] > 0]]
        })
        st.dataframe(sort_table_A, hide_index=True)

        st.markdown(f"""
        **Step 3: Find 0.5 crossing**

        - Looking for first C_k >= 0.5
        - Interpolate between adjacent values
        - **Result: y_smooth[0] = {result_A['median_result']:.4f}**
        """)

        # Visual for Point A
        fig_A = create_kernel_window_figure(t_example, y_example, 0.0, h_example, 0)
        st.plotly_chart(fig_A, use_container_width=True)

        st.markdown("---")

        # Point B: t_i = 0.5 (with outlier)
        st.markdown("### Point B: Evaluation at t_i = 0.5 (Interior with Outlier)")

        result_B = compute_example_walkthrough(t_example, y_example, 0.5, h_example, 5)

        st.markdown(f"""
        **Step 1: Identify neighbors in window** [t_i - h, t_i + h] = [0.25, 0.75]

        Points in window: t = 0.3, 0.4, 0.5, 0.6, 0.7 (indices 3, 4, 5, 6, 7)

        **Note:** The outlier at t=0.5 (y=3.0) is in the window, but watch how it gets handled!
        """)

        in_window_B = np.abs(result_B['u']) <= 1
        table_B = pd.DataFrame({
            'j': range(10),
            't_j': [f'{t:.1f}' for t in t_example],
            'y_j': [f'{y:.3f}' for y in y_example],
            'u_j': [f'{u:.2f}' for u in result_B['u']],
            'In Window': ['Yes' if iw else 'No' for iw in in_window_B],
            'w_j': [f'{w:.4f}' for w in result_B['weights']],
            'Note': ['', '', '', '', '', 'OUTLIER', '', '', '', '']
        })
        st.dataframe(table_B, hide_index=True)

        st.markdown("**Step 2: Sort by y values**")

        active_B = result_B['weights_sorted'] > 0
        sort_table_B = pd.DataFrame({
            'Rank': range(1, np.sum(active_B) + 1),
            'y_(k)': [f'{y:.3f}' for y in result_B['y_sorted'][active_B]],
            'w_(k)': [f'{w:.4f}' for w in result_B['weights_sorted'][active_B]],
            'C_k': [f'{c:.4f}' for c in result_B['cumsum_norm'][active_B]]
        })
        st.dataframe(sort_table_B, hide_index=True)

        st.markdown(f"""
        **Step 3: Observe robustness**

        - The outlier y=3.0 is at the **top** after sorting
        - Its weight is added last to the cumulative sum
        - The 0.5 crossing happens **before** reaching the outlier!
        - **Result: y_smooth[5] = {result_B['median_result']:.4f}**

        The outlier has **minimal influence** on the weighted median!
        """)

        # Visual for Point B
        fig_B = create_kernel_window_figure(t_example, y_example, 0.5, h_example, 5)
        st.plotly_chart(fig_B, use_container_width=True)

        # Weighted median visualization
        fig_wm_B = create_weighted_median_figure(
            result_B['y_sorted'][active_B],
            result_B['weights_sorted'][active_B],
            result_B['cumsum_norm'][active_B]
        )
        st.plotly_chart(fig_wm_B, use_container_width=True)

        st.markdown("---")

        # Point C: t_i = 0.9 (right boundary)
        st.markdown("### Point C: Evaluation at t_i = 0.9 (Right Boundary)")

        result_C = compute_example_walkthrough(t_example, y_example, 0.9, h_example, 9)

        st.markdown(f"""
        **Step 1: Identify neighbors in window** [t_i - h, t_i + h] = [0.65, 1.15]

        Points in window: t = 0.7, 0.8, 0.9 (indices 7, 8, 9)

        **Note:** Asymmetric window - fewer neighbors on the right side (boundary effect)
        """)

        in_window_C = np.abs(result_C['u']) <= 1
        table_C = pd.DataFrame({
            'j': range(10),
            't_j': [f'{t:.1f}' for t in t_example],
            'y_j': [f'{y:.3f}' for y in y_example],
            'u_j': [f'{u:.2f}' for u in result_C['u']],
            'In Window': ['Yes' if iw else 'No' for iw in in_window_C],
            'w_j': [f'{w:.4f}' for w in result_C['weights']]
        })
        st.dataframe(table_C, hide_index=True)

        active_C = result_C['weights_sorted'] > 0
        st.markdown(f"""
        **Result: y_smooth[9] = {result_C['median_result']:.4f}**

        Boundary points have fewer neighbors but the algorithm handles this naturally.
        """)

        fig_C = create_kernel_window_figure(t_example, y_example, 0.9, h_example, 9)
        st.plotly_chart(fig_C, use_container_width=True)

    # ========== SECTION 3: Visual Concept Diagrams ==========
    with st.expander("3. Visual Concept Diagrams", expanded=True):
        st.markdown("### Diagram 1: The Sliding Window Concept")

        st.markdown("""
        As we compute the smoothed value at each point t_i, we "slide" a kernel window
        across the signal. The window extends from t_i - h to t_i + h.
        """)

        # Create sliding window animation as static frames
        st.markdown("**Window positions at different evaluation points:**")

        frame_positions = [0.1, 0.3, 0.5, 0.7, 0.9]
        frame_indices = [1, 3, 5, 7, 9]

        cols = st.columns(len(frame_positions))
        for i, (t_pos, idx) in enumerate(zip(frame_positions, frame_indices)):
            with cols[i]:
                st.markdown(f"**t_i = {t_pos}**")

                # Mini visualization
                import plotly.graph_objects as go
                fig_mini = go.Figure()

                # Signal
                fig_mini.add_trace(go.Scatter(
                    x=t_example, y=y_example,
                    mode='lines+markers',
                    marker=dict(size=6, color='gray'),
                    line=dict(color='gray', width=1),
                    showlegend=False
                ))

                # Window region (shaded)
                u = (t_example - t_pos) / h_example
                in_win = np.abs(u) <= 1
                fig_mini.add_vrect(
                    x0=max(0, t_pos - h_example),
                    x1=min(0.9, t_pos + h_example),
                    fillcolor='rgba(0,100,200,0.2)',
                    line_width=0
                )

                # Current point
                fig_mini.add_trace(go.Scatter(
                    x=[t_pos], y=[y_example[idx]],
                    mode='markers',
                    marker=dict(size=12, color='red', symbol='star'),
                    showlegend=False
                ))

                fig_mini.update_layout(
                    height=200,
                    margin=dict(l=20, r=20, t=20, b=20),
                    xaxis=dict(range=[-0.05, 0.95]),
                    yaxis=dict(range=[-1.5, 3.5])
                )

                st.plotly_chart(fig_mini, use_container_width=True)

        st.markdown("---")
        st.markdown("### Diagram 2: Kernel Weight Distribution")

        st.markdown("""
        The Epanechnikov kernel assigns weights based on distance from t_i:
        - **Center (u=0):** Maximum weight K(0) = 0.75
        - **Edge (|u|=1):** Zero weight K(+/-1) = 0
        - **Parabolic decay:** Smooth transition between center and edge
        """)

        # Plot kernel shape
        u_kernel = np.linspace(-1.5, 1.5, 200)
        k_values = 0.75 * (1 - u_kernel**2) * (np.abs(u_kernel) <= 1)

        fig_kernel = go.Figure()
        fig_kernel.add_trace(go.Scatter(
            x=u_kernel, y=k_values,
            mode='lines',
            line=dict(color='steelblue', width=3),
            fill='tozeroy',
            fillcolor='rgba(70,130,180,0.3)',
            name='K(u)'
        ))

        fig_kernel.add_vline(x=-1, line=dict(color='red', dash='dash'))
        fig_kernel.add_vline(x=1, line=dict(color='red', dash='dash'))

        fig_kernel.update_layout(
            height=300,
            title=dict(text='Epanechnikov Kernel K(u) = 0.75(1-u^2)', x=0.5),
            xaxis_title='Scaled distance u = (t - t_i) / h',
            yaxis_title='Weight K(u)',
            margin=dict(l=50, r=30, t=60, b=40)
        )

        st.plotly_chart(fig_kernel, use_container_width=True)

        st.markdown("---")
        st.markdown("### Diagram 3: Complete Smoothing Pass")

        st.markdown("""
        The final smoothed signal is computed by applying the weighted median
        at every point along the signal.
        """)

        # Compute full smoothed signal
        from smoothers import epa_local_median
        y_smooth_example = epa_local_median(t_example, y_example, h_example)

        fig_full = go.Figure()

        # Original with outlier
        fig_full.add_trace(go.Scatter(
            x=t_example, y=y_example,
            mode='lines+markers',
            marker=dict(size=10, color='gray'),
            line=dict(color='gray', width=1, dash='dash'),
            name='Original (with outlier)'
        ))

        # Highlight outlier
        fig_full.add_trace(go.Scatter(
            x=[0.5], y=[3.0],
            mode='markers',
            marker=dict(size=14, color='red', symbol='x'),
            name='Outlier'
        ))

        # Smoothed result
        fig_full.add_trace(go.Scatter(
            x=t_example, y=y_smooth_example,
            mode='lines+markers',
            marker=dict(size=10, color='green'),
            line=dict(color='green', width=2),
            name='Smoothed (EPA Local Median)'
        ))

        # Clean signal for reference
        fig_full.add_trace(go.Scatter(
            x=t_example, y=y_clean_example,
            mode='lines',
            line=dict(color='blue', width=2, dash='dot'),
            name='Clean Signal (ground truth)'
        ))

        fig_full.update_layout(
            height=400,
            title=dict(text='Complete Smoothing Result', x=0.5),
            xaxis_title='t',
            yaxis_title='y',
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='center', x=0.5),
            margin=dict(l=50, r=30, t=80, b=40)
        )

        st.plotly_chart(fig_full, use_container_width=True)

        st.success(f"""
        **Observation:** The smoothed curve closely follows the clean signal
        and is NOT pulled toward the outlier at t=0.5. This demonstrates the
        robustness of the weighted median approach!
        """)

    # ========== SECTION 4: Key Insights ==========
    with st.expander("4. Key Insights", expanded=True):
        st.markdown("### Understanding the Method")

        col1, col2 = st.columns(2)

        with col1:
            st.info("""
            **Why Epanechnikov Kernel?**

            - AMISE-optimal: minimizes mean integrated squared error
            - Compact support: K(u) = 0 for |u| > 1
            - Computational efficiency: only process nearby points
            - Smooth parabolic shape: no discontinuities in derivatives
            """)

            st.warning("""
            **Bandwidth Trade-off**

            - **Large h:** More smoothing, captures global trends,
              may miss local features
            - **Small h:** Less smoothing, captures details,
              may be noisy
            - **Rule of thumb:** h scales as n^(-1/5) for optimal MSE
            """)

        with col2:
            st.success("""
            **Why Weighted Median?**

            - Breakdown point = 50%: up to half the data can be
              corrupted without affecting the estimate unboundedly
            - Robust to outliers: outlier magnitude doesn't matter
            - Influence function bounded: IF(x) = sign(x - median)
            """)

            st.info("""
            **Why Linear Interpolation?**

            - Eliminates step artifacts in smooth signals
            - Produces continuous output even with discrete inputs
            - Exact at the 0.5 cumulative weight crossing
            - Makes EMD iterations more stable
            """)

        st.markdown("---")
        st.markdown("### Formula Summary")

        st.latex(r"""
        \text{Scaled distance: } u_j = \frac{t_j - t_i}{h}
        """)

        st.latex(r"""
        \text{Epanechnikov kernel: } K(u) = \frac{3}{4}(1 - u^2) \cdot \mathbf{1}_{|u| \leq 1}
        """)

        st.latex(r"""
        \text{Weighted median: } \tilde{m}(t_i) = \arg\min_m \sum_{j=1}^{n} |y_j - m| \cdot K\left(\frac{t_j - t_i}{h}\right)
        """)

        st.latex(r"""
        \text{Cumulative weight: } C_k = \frac{\sum_{j=1}^{k} w_{(j)}}{\sum_{j=1}^{n} w_j}
        """)

        st.latex(r"""
        \text{Linear interpolation: } \tilde{m} = y_{(k-1)} + \frac{0.5 - C_{k-1}}{C_k - C_{k-1}} (y_{(k)} - y_{(k-1)})
        """)

    # ========== SECTION 5: Understanding Micro-Jumps ==========
    with st.expander("5. Understanding Micro-Jumps in Smooth Output", expanded=False):
        st.markdown("""
        ### Why Micro-Jumps Occur

        Even with n=4000 points, you may observe small discontinuities ("jumps") in the
        smoothed output. This is **mathematically inherent** to the weighted median algorithm.
        """)

        st.markdown("**Root Cause:**")
        st.markdown("""
        1. `searchsorted()` finds the first index where cumulative weight >= 0.5
        2. As we slide across the signal, small weight changes cause this index to flip
        3. When the index changes, we interpolate between *different* pairs of sorted values
        4. This creates micro-discontinuities even with linear interpolation
        """)

        st.markdown("**Example:**")
        st.code("""
Point i:   cumsum = [0.48, 0.52, ...] -> idx=1 -> interpolate y[0]..y[1]
Point i+1: cumsum = [0.49, 0.51, ...] -> idx=1 -> same pair (smooth)
Point i+2: cumsum = [0.51, 0.53, ...] -> idx=0 -> JUMP! different pair
        """, language="text")

        st.markdown("""
        **Key Insight:** The interpolation is smooth *within* a pair of values, but
        discontinuous *when the pair changes*.
        """)

        st.markdown("---")
        st.markdown("### Mitigation Strategies")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.info("""
            **1. Increase Bandwidth (h0)**

            Larger windows average more points, reducing jump frequency.
            The effective number of points in each window is approximately:

            $n_{window} \\approx 2h \\cdot n$

            More points = more stable median estimates.
            """)

        with col2:
            st.success("""
            **2. Accept as Feature**

            The jumps indicate where the local median "switches" between dominant
            values in the sorted order. This is actually informative:

            - Shows transitions between local modes
            - Reveals signal structure
            - Preserves edge information
            """)

        with col3:
            st.warning("""
            **3. Post-Processing**

            If continuous output is required, apply light Gaussian smoothing externally:

            ```python
            from scipy.ndimage import gaussian_filter1d
            y_final = gaussian_filter1d(y_smooth, sigma=2)
            ```

            Trade-off: reduces robustness slightly.
            """)

        st.markdown("---")
        st.markdown("### Mathematical Explanation")

        st.markdown("""
        The weighted median is computed via cumulative weights:
        """)
        st.latex(r"C_k = \frac{\sum_{j=1}^{k} w_{(j)}}{\sum_{j=1}^{n} w_j}")

        st.markdown("""
        The median is the value $y_{(k)}$ where $C_k$ first crosses 0.5. As we move
        from evaluation point $t_i$ to $t_{i+1}$:

        - The kernel weights $w_j = K((t_j - t_i)/h)$ shift slightly
        - The sorted order of $(y, w)$ pairs may change
        - The cumulative sum $C_k$ changes
        - The index $k^*$ where $C_k \\geq 0.5$ may jump

        When $k^*$ changes, we suddenly interpolate between a *different pair* of
        sorted $y$ values, creating a discontinuity. This is unavoidable without
        fundamentally changing the median definition.
        """)

        st.markdown("---")
        st.markdown("### Comparison with Nadaraya-Watson Mean")

        st.markdown("""
        The NW mean (weighted average) does **not** have this issue because it uses
        all points continuously:
        """)
        st.latex(r"\hat{m}(t) = \frac{\sum_j y_j \cdot K((t_j - t)/h)}{\sum_j K((t_j - t)/h)}")

        st.markdown("""
        As weights shift smoothly, the weighted average shifts smoothly. However,
        the NW mean is **not robust** to outliers - a single extreme value can
        dominate the estimate.

        **The micro-jumps in the weighted median are the price we pay for robustness.**
        """)


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

    n_points_options = [500, 1000, 1500, 2000, 3000, 4000]
    n_points = st.sidebar.select_slider(
        "n (signal points)",
        options=n_points_options,
        value=2000,
        help="Number of time points. Higher = more detail but slower"
    )

    if n_points != 2000:
        st.sidebar.warning("n ≠ 2000: Computing on-demand (slower)")

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
        n_points=n_points,
        am_depth=am_depth,
        base_freq=float(base_freq),
        chirp_rate=float(chirp_rate),
        noise_sigma=sigma,
        seed=42
    )

    # Main content area
    st.title("EMD Local Median Smoother")
    st.markdown("**Empirical Mode Decomposition with Kernel-Weighted Local Median**")

    # Method tabs (including Theory & Math and How It Works)
    tabs = st.tabs(["Local Median", "Median", "Average", "Theory & Math", "How It Works"])

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

    # How It Works Tab (5th tab)
    with tabs[4]:
        render_computation_tab()

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
