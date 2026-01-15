"""
EPA Kernel-Weighted Local Median Smoothing - Comprehensive Interactive Dashboard

Generates a single self-contained HTML file with:
- Interactive sliders for bandwidth, outlier fraction, noise
- Real-time plot updates
- Animation of the smoothing process
- MathJax-rendered formulas
- Metrics comparison panel
"""

import numpy as np
import json
from pathlib import Path


def epanechnikov_kernel(u):
    """EPA kernel: K(u) = 0.75(1 - u^2) for |u| <= 1"""
    weights = np.zeros_like(u, dtype=float)
    mask = np.abs(u) <= 1
    weights[mask] = 0.75 * (1 - u[mask]**2)
    return weights


def weighted_median(values, weights):
    """Compute weighted median with proper edge case handling."""
    if len(values) == 0 or np.sum(weights) == 0:
        return np.nan

    # Sort by values
    sorted_indices = np.argsort(values)
    sorted_values = values[sorted_indices]
    sorted_weights = weights[sorted_indices]

    # Normalize weights
    total_weight = np.sum(sorted_weights)
    cumulative_weights = np.cumsum(sorted_weights) / total_weight

    # Find median with linear interpolation for edge case
    idx = np.searchsorted(cumulative_weights, 0.5)

    if idx >= len(sorted_values):
        return sorted_values[-1]
    elif idx == 0:
        return sorted_values[0]
    elif np.isclose(cumulative_weights[idx-1], 0.5):
        # Exact 0.5 - interpolate
        return 0.5 * (sorted_values[idx-1] + sorted_values[idx])
    else:
        return sorted_values[idx]


def epa_local_median(x, y, bandwidth):
    """EPA kernel-weighted local median smoother."""
    n = len(x)
    y_smooth = np.zeros(n)

    for i in range(n):
        u = (x - x[i]) / bandwidth
        weights = epanechnikov_kernel(u)
        y_smooth[i] = weighted_median(y, weights)

    return y_smooth


def epa_local_mean(x, y, bandwidth):
    """EPA kernel-weighted local mean (Nadaraya-Watson)."""
    n = len(x)
    y_smooth = np.zeros(n)

    for i in range(n):
        u = (x - x[i]) / bandwidth
        weights = epanechnikov_kernel(u)
        total = np.sum(weights)
        if total > 0:
            y_smooth[i] = np.sum(weights * y) / total
        else:
            y_smooth[i] = np.nan

    return y_smooth


def generate_data(n, noise_std, outlier_frac, seed=42):
    """Generate noisy curve with outliers."""
    np.random.seed(seed)
    x = np.linspace(0, 2 * np.pi, n)
    y_true = np.sin(x) + 0.5 * np.cos(2 * x)
    y_noisy = y_true + np.random.normal(0, noise_std, n)

    n_outliers = int(n * outlier_frac)
    if n_outliers > 0:
        outlier_idx = np.random.choice(n, n_outliers, replace=False)
        outlier_signs = np.random.choice([-1, 1], n_outliers)
        y_noisy[outlier_idx] += outlier_signs * 4.0

    return x, y_true, y_noisy


def precompute_all_data():
    """Pre-compute smoothed curves for all parameter combinations."""
    n = 150
    bandwidths = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0, 1.2, 1.5]
    outlier_fracs = [0.0, 0.02, 0.05, 0.08, 0.10, 0.15, 0.20]
    noise_stds = [0.15, 0.25, 0.35, 0.5]

    # Base x values
    x = np.linspace(0, 2 * np.pi, n).tolist()
    y_true = (np.sin(np.array(x)) + 0.5 * np.cos(2 * np.array(x))).tolist()

    # Pre-compute for each combination
    data = {
        'x': x,
        'y_true': y_true,
        'n': n,
        'bandwidths': bandwidths,
        'outlier_fracs': outlier_fracs,
        'noise_stds': noise_stds,
        'results': {}
    }

    for noise in noise_stds:
        for outlier in outlier_fracs:
            key = f"{noise}_{outlier}"
            _, _, y_noisy = generate_data(n, noise, outlier, seed=42)

            result = {
                'y_noisy': y_noisy.tolist(),
                'smoothed': {}
            }

            for bw in bandwidths:
                y_median = epa_local_median(np.array(x), y_noisy, bw)
                y_mean = epa_local_mean(np.array(x), y_noisy, bw)

                # Compute metrics
                rmse_median = float(np.sqrt(np.mean((y_median - np.array(y_true))**2)))
                rmse_mean = float(np.sqrt(np.mean((y_mean - np.array(y_true))**2)))
                mae_median = float(np.mean(np.abs(y_median - np.array(y_true))))
                mae_mean = float(np.mean(np.abs(y_mean - np.array(y_true))))

                result['smoothed'][str(bw)] = {
                    'median': y_median.tolist(),
                    'mean': y_mean.tolist(),
                    'rmse_median': rmse_median,
                    'rmse_mean': rmse_mean,
                    'mae_median': mae_median,
                    'mae_mean': mae_mean
                }

            data['results'][key] = result

    return data


def generate_html():
    """Generate the complete interactive HTML dashboard."""

    print("Pre-computing all parameter combinations...")
    data = precompute_all_data()
    data_json = json.dumps(data)

    html = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>EPA Kernel-Weighted Local Median Smoothing</title>
    <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
    <script src="https://polyfill.io/v3/polyfill.min.js?features=es6"></script>
    <script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
    <style>
        * {{
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }}
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }}
        .container {{
            max-width: 1400px;
            margin: 0 auto;
        }}
        .header {{
            text-align: center;
            color: white;
            margin-bottom: 20px;
        }}
        .header h1 {{
            font-size: 2em;
            margin-bottom: 10px;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        }}
        .header p {{
            font-size: 1.1em;
            opacity: 0.9;
        }}
        .panel {{
            background: white;
            border-radius: 12px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.2);
            padding: 20px;
            margin-bottom: 20px;
        }}
        .controls {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
            gap: 20px;
            align-items: center;
        }}
        .control-group {{
            display: flex;
            flex-direction: column;
        }}
        .control-group label {{
            font-weight: 600;
            color: #333;
            margin-bottom: 8px;
        }}
        .control-group input[type="range"] {{
            width: 100%;
            height: 8px;
            border-radius: 4px;
            background: #e0e0e0;
            outline: none;
            -webkit-appearance: none;
        }}
        .control-group input[type="range"]::-webkit-slider-thumb {{
            -webkit-appearance: none;
            width: 20px;
            height: 20px;
            border-radius: 50%;
            background: #667eea;
            cursor: pointer;
            box-shadow: 0 2px 6px rgba(0,0,0,0.2);
        }}
        .value-display {{
            text-align: center;
            font-size: 1.1em;
            color: #667eea;
            font-weight: bold;
            margin-top: 5px;
        }}
        .main-content {{
            display: grid;
            grid-template-columns: 1fr 320px;
            gap: 20px;
        }}
        @media (max-width: 1000px) {{
            .main-content {{
                grid-template-columns: 1fr;
            }}
        }}
        .metrics-panel {{
            display: flex;
            flex-direction: column;
            gap: 15px;
        }}
        .metric-card {{
            background: linear-gradient(135deg, #f5f7fa 0%, #e4e8ec 100%);
            border-radius: 10px;
            padding: 15px;
            text-align: center;
        }}
        .metric-card h3 {{
            font-size: 0.9em;
            color: #666;
            margin-bottom: 8px;
        }}
        .metric-value {{
            font-size: 1.8em;
            font-weight: bold;
        }}
        .metric-value.green {{ color: #2ca02c; }}
        .metric-value.red {{ color: #d62728; }}
        .metric-value.blue {{ color: #1f77b4; }}
        .improvement {{
            font-size: 0.85em;
            margin-top: 5px;
        }}
        .improvement.positive {{ color: #2ca02c; }}
        .improvement.negative {{ color: #d62728; }}
        .math-panel {{
            background: #f8f9fa;
            border-left: 4px solid #667eea;
            padding: 20px;
            margin-top: 10px;
        }}
        .math-panel h3 {{
            color: #333;
            margin-bottom: 15px;
        }}
        .formula-row {{
            display: flex;
            justify-content: space-around;
            flex-wrap: wrap;
            gap: 20px;
        }}
        .formula-item {{
            text-align: center;
            padding: 10px;
        }}
        .formula-item .label {{
            font-size: 0.85em;
            color: #666;
            margin-bottom: 5px;
        }}
        .tabs {{
            display: flex;
            border-bottom: 2px solid #e0e0e0;
            margin-bottom: 15px;
        }}
        .tab {{
            padding: 10px 20px;
            cursor: pointer;
            border-bottom: 3px solid transparent;
            transition: all 0.3s;
            font-weight: 500;
        }}
        .tab:hover {{
            background: #f5f5f5;
        }}
        .tab.active {{
            border-bottom-color: #667eea;
            color: #667eea;
        }}
        .tab-content {{
            display: none;
        }}
        .tab-content.active {{
            display: block;
        }}
        .animation-controls {{
            display: flex;
            gap: 10px;
            justify-content: center;
            margin-top: 15px;
        }}
        .btn {{
            padding: 10px 25px;
            border: none;
            border-radius: 6px;
            cursor: pointer;
            font-weight: 600;
            transition: all 0.3s;
        }}
        .btn-primary {{
            background: #667eea;
            color: white;
        }}
        .btn-primary:hover {{
            background: #5a6fd6;
        }}
        .btn-secondary {{
            background: #e0e0e0;
            color: #333;
        }}
        .btn-secondary:hover {{
            background: #d0d0d0;
        }}
        .legend-item {{
            display: inline-flex;
            align-items: center;
            margin-right: 20px;
            font-size: 0.9em;
        }}
        .legend-color {{
            width: 30px;
            height: 4px;
            margin-right: 8px;
            border-radius: 2px;
        }}
        .insight-box {{
            background: linear-gradient(135deg, #e8f5e9 0%, #c8e6c9 100%);
            border-radius: 8px;
            padding: 15px;
            margin-top: 15px;
        }}
        .insight-box h4 {{
            color: #2e7d32;
            margin-bottom: 8px;
        }}
        .insight-box p {{
            color: #1b5e20;
            font-size: 0.95em;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>EPA Kernel-Weighted Local Median Smoothing</h1>
            <p>Robust nonparametric smoothing that resists outliers</p>
        </div>

        <div class="panel">
            <div class="controls">
                <div class="control-group">
                    <label>Bandwidth (h)</label>
                    <input type="range" id="bandwidth" min="0.2" max="1.5" step="0.1" value="0.5">
                    <div class="value-display" id="bw-value">h = 0.5</div>
                </div>
                <div class="control-group">
                    <label>Outlier Fraction</label>
                    <input type="range" id="outliers" min="0" max="0.2" step="0.02" value="0.08">
                    <div class="value-display" id="outlier-value">8%</div>
                </div>
                <div class="control-group">
                    <label>Noise Level</label>
                    <input type="range" id="noise" min="0.15" max="0.5" step="0.1" value="0.25">
                    <div class="value-display" id="noise-value">0.25</div>
                </div>
            </div>
        </div>

        <div class="main-content">
            <div class="panel">
                <div id="main-plot" style="width:100%;height:450px;"></div>
                <div style="text-align:center;margin-top:10px;">
                    <span class="legend-item"><span class="legend-color" style="background:#999;"></span>Noisy Data</span>
                    <span class="legend-item"><span class="legend-color" style="background:#000;border-style:dashed;"></span>True Function</span>
                    <span class="legend-item"><span class="legend-color" style="background:#2ca02c;"></span>EPA Median</span>
                    <span class="legend-item"><span class="legend-color" style="background:#d62728;"></span>EPA Mean</span>
                </div>
            </div>
            <div class="metrics-panel">
                <div class="metric-card">
                    <h3>RMSE - Median</h3>
                    <div class="metric-value green" id="rmse-median">0.000</div>
                </div>
                <div class="metric-card">
                    <h3>RMSE - Mean</h3>
                    <div class="metric-value red" id="rmse-mean">0.000</div>
                </div>
                <div class="metric-card">
                    <h3>Median Improvement</h3>
                    <div class="metric-value blue" id="improvement">0%</div>
                    <div class="improvement" id="improvement-text">vs Mean smoother</div>
                </div>
                <div class="metric-card">
                    <h3>MAE - Median</h3>
                    <div class="metric-value green" id="mae-median">0.000</div>
                </div>
                <div class="metric-card">
                    <h3>MAE - Mean</h3>
                    <div class="metric-value red" id="mae-mean">0.000</div>
                </div>
            </div>
        </div>

        <div class="panel">
            <div class="math-panel">
                <h3>Algorithm Formulas</h3>
                <div class="formula-row">
                    <div class="formula-item">
                        <div class="label">Epanechnikov Kernel</div>
                        <div>\\( K(u) = \\frac{{3}}{{4}}(1 - u^2) \\cdot \\mathbf{{1}}_{{|u| \\leq 1}} \\)</div>
                    </div>
                    <div class="formula-item">
                        <div class="label">Scaled Distance</div>
                        <div>\\( u_i = \\frac{{x_i - x_0}}{{h}} \\)</div>
                    </div>
                    <div class="formula-item">
                        <div class="label">Weighted Median</div>
                        <div>\\( \\hat{{y}}(x_0) = \\text{{wmedian}}\\{{y_i : w_i = K(u_i)\\}} \\)</div>
                    </div>
                </div>
            </div>

            <div class="insight-box" id="insight-box">
                <h4>Key Insight</h4>
                <p id="insight-text">The weighted median ignores the magnitude of outliers - only their position in the sorted order matters. This makes it robust to extreme values that would heavily influence the weighted mean.</p>
            </div>
        </div>

        <div class="panel">
            <div class="tabs">
                <div class="tab active" onclick="showTab('kernel')">Kernel Shape</div>
                <div class="tab" onclick="showTab('bandwidth')">Bandwidth Comparison</div>
                <div class="tab" onclick="showTab('residuals')">Residual Analysis</div>
                <div class="tab" onclick="showTab('animation')">Step-by-Step Animation</div>
            </div>

            <div id="kernel-tab" class="tab-content active">
                <div id="kernel-plot" style="width:100%;height:350px;"></div>
            </div>

            <div id="bandwidth-tab" class="tab-content">
                <div id="bandwidth-plot" style="width:100%;height:350px;"></div>
            </div>

            <div id="residuals-tab" class="tab-content">
                <div id="residuals-plot" style="width:100%;height:350px;"></div>
            </div>

            <div id="animation-tab" class="tab-content">
                <div id="animation-plot" style="width:100%;height:350px;"></div>
                <div class="animation-controls">
                    <button class="btn btn-primary" onclick="startAnimation()">Play Animation</button>
                    <button class="btn btn-secondary" onclick="stopAnimation()">Stop</button>
                    <button class="btn btn-secondary" onclick="resetAnimation()">Reset</button>
                </div>
                <div style="text-align:center;margin-top:10px;">
                    <span id="anim-status">Point: 0 / 150</span>
                </div>
            </div>
        </div>
    </div>

    <script>
        // Pre-computed data
        const DATA = {data_json};

        // Current state
        let currentBandwidth = 0.5;
        let currentOutliers = 0.08;
        let currentNoise = 0.25;
        let animationInterval = null;
        let animationIndex = 0;

        // Snap to nearest available value
        function snapToAvailable(value, available) {{
            return available.reduce((prev, curr) =>
                Math.abs(curr - value) < Math.abs(prev - value) ? curr : prev
            );
        }}

        // Get current data key
        function getDataKey() {{
            const noise = snapToAvailable(currentNoise, DATA.noise_stds);
            const outlier = snapToAvailable(currentOutliers, DATA.outlier_fracs);
            return `${{noise}}_${{outlier}}`;
        }}

        // Update main plot
        function updateMainPlot() {{
            const key = getDataKey();
            const bw = snapToAvailable(currentBandwidth, DATA.bandwidths);
            const result = DATA.results[key];
            const smoothed = result.smoothed[bw];

            const traces = [
                {{
                    x: DATA.x,
                    y: result.y_noisy,
                    mode: 'markers',
                    name: 'Noisy Data',
                    marker: {{ size: 5, color: '#999', opacity: 0.6 }}
                }},
                {{
                    x: DATA.x,
                    y: DATA.y_true,
                    mode: 'lines',
                    name: 'True Function',
                    line: {{ color: 'black', width: 2, dash: 'dash' }}
                }},
                {{
                    x: DATA.x,
                    y: smoothed.median,
                    mode: 'lines',
                    name: 'EPA Median',
                    line: {{ color: '#2ca02c', width: 3 }}
                }},
                {{
                    x: DATA.x,
                    y: smoothed.mean,
                    mode: 'lines',
                    name: 'EPA Mean',
                    line: {{ color: '#d62728', width: 3 }}
                }}
            ];

            const layout = {{
                title: 'EPA Local Median vs Mean Smoothing',
                xaxis: {{ title: 'x' }},
                yaxis: {{ title: 'y' }},
                showlegend: false,
                margin: {{ t: 40, b: 40, l: 50, r: 20 }}
            }};

            Plotly.react('main-plot', traces, layout);

            // Update metrics
            document.getElementById('rmse-median').textContent = smoothed.rmse_median.toFixed(3);
            document.getElementById('rmse-mean').textContent = smoothed.rmse_mean.toFixed(3);
            document.getElementById('mae-median').textContent = smoothed.mae_median.toFixed(3);
            document.getElementById('mae-mean').textContent = smoothed.mae_mean.toFixed(3);

            const improvement = ((smoothed.rmse_mean - smoothed.rmse_median) / smoothed.rmse_mean * 100);
            document.getElementById('improvement').textContent = improvement.toFixed(1) + '%';

            const improvementEl = document.getElementById('improvement');
            if (improvement > 0) {{
                improvementEl.classList.remove('red');
                improvementEl.classList.add('green');
                document.getElementById('improvement-text').textContent = 'better than Mean';
                document.getElementById('improvement-text').className = 'improvement positive';
            }} else {{
                improvementEl.classList.remove('green');
                improvementEl.classList.add('red');
                document.getElementById('improvement-text').textContent = 'worse than Mean';
                document.getElementById('improvement-text').className = 'improvement negative';
            }}

            // Update insight
            updateInsight(improvement, currentOutliers);
        }}

        function updateInsight(improvement, outlierFrac) {{
            const insightEl = document.getElementById('insight-text');
            if (outlierFrac < 0.02) {{
                insightEl.textContent = "With no outliers, both methods perform similarly. The median's robustness provides no advantage here.";
            }} else if (improvement > 30) {{
                insightEl.textContent = `With ${{(outlierFrac*100).toFixed(0)}}% outliers, the median smoother shows ${{improvement.toFixed(0)}}% lower error! Outlier magnitude doesn't affect the weighted median - only the count of points above/below matters.`;
            }} else if (improvement > 10) {{
                insightEl.textContent = `The median smoother is more robust to the ${{(outlierFrac*100).toFixed(0)}}% outliers, achieving ${{improvement.toFixed(0)}}% lower RMSE. Increase outlier fraction to see larger differences.`;
            }} else {{
                insightEl.textContent = "At this setting, both smoothers perform comparably. Try increasing the outlier fraction or decreasing bandwidth to see the median's robustness advantage.";
            }}
        }}

        // Kernel shape plot
        function plotKernel() {{
            const u = [];
            const k = [];
            for (let i = -1.5; i <= 1.5; i += 0.01) {{
                u.push(i);
                k.push(Math.abs(i) <= 1 ? 0.75 * (1 - i*i) : 0);
            }}

            const traces = [{{
                x: u,
                y: k,
                mode: 'lines',
                fill: 'tozeroy',
                fillcolor: 'rgba(102, 126, 234, 0.3)',
                line: {{ color: '#667eea', width: 3 }},
                name: 'EPA Kernel'
            }}];

            const layout = {{
                title: 'Epanechnikov Kernel K(u) = 0.75(1 - u<sup>2</sup>) for |u| <= 1',
                xaxis: {{ title: 'u = (x - x0) / h', zeroline: true }},
                yaxis: {{ title: 'K(u)', range: [0, 0.85] }},
                shapes: [
                    {{ type: 'line', x0: -1, y0: 0, x1: -1, y1: 0.8, line: {{ color: 'red', dash: 'dash', width: 2 }} }},
                    {{ type: 'line', x0: 1, y0: 0, x1: 1, y1: 0.8, line: {{ color: 'red', dash: 'dash', width: 2 }} }}
                ],
                annotations: [
                    {{ x: -1, y: 0.82, text: 'u = -1', showarrow: false, font: {{ color: 'red' }} }},
                    {{ x: 1, y: 0.82, text: 'u = +1', showarrow: false, font: {{ color: 'red' }} }}
                ],
                margin: {{ t: 50, b: 50, l: 50, r: 20 }}
            }};

            Plotly.react('kernel-plot', traces, layout);
        }}

        // Bandwidth comparison plot
        function plotBandwidthComparison() {{
            const key = getDataKey();
            const result = DATA.results[key];
            const bws = [0.2, 0.4, 0.7, 1.0];
            const colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#9467bd'];

            const traces = [
                {{
                    x: DATA.x,
                    y: result.y_noisy,
                    mode: 'markers',
                    name: 'Noisy Data',
                    marker: {{ size: 4, color: '#ccc' }}
                }},
                {{
                    x: DATA.x,
                    y: DATA.y_true,
                    mode: 'lines',
                    name: 'True',
                    line: {{ color: 'black', width: 2, dash: 'dash' }}
                }}
            ];

            bws.forEach((bw, i) => {{
                const smoothed = result.smoothed[bw] || result.smoothed[snapToAvailable(bw, DATA.bandwidths)];
                if (smoothed) {{
                    traces.push({{
                        x: DATA.x,
                        y: smoothed.median,
                        mode: 'lines',
                        name: `h=${{bw}} (RMSE=${{smoothed.rmse_median.toFixed(3)}})`,
                        line: {{ color: colors[i], width: 2.5 }}
                    }});
                }}
            }});

            const layout = {{
                title: 'Effect of Bandwidth on EPA Local Median',
                xaxis: {{ title: 'x' }},
                yaxis: {{ title: 'y' }},
                legend: {{ x: 0.02, y: 0.98 }},
                margin: {{ t: 50, b: 50, l: 50, r: 20 }}
            }};

            Plotly.react('bandwidth-plot', traces, layout);
        }}

        // Residuals plot
        function plotResiduals() {{
            const key = getDataKey();
            const bw = snapToAvailable(currentBandwidth, DATA.bandwidths);
            const result = DATA.results[key];
            const smoothed = result.smoothed[bw];

            const residuals_median = DATA.y_true.map((y, i) => smoothed.median[i] - y);
            const residuals_mean = DATA.y_true.map((y, i) => smoothed.mean[i] - y);

            const traces = [
                {{
                    x: DATA.x,
                    y: residuals_median,
                    mode: 'lines+markers',
                    name: 'Median Residuals',
                    line: {{ color: '#2ca02c' }},
                    marker: {{ size: 4 }}
                }},
                {{
                    x: DATA.x,
                    y: residuals_mean,
                    mode: 'lines+markers',
                    name: 'Mean Residuals',
                    line: {{ color: '#d62728' }},
                    marker: {{ size: 4 }}
                }},
                {{
                    x: DATA.x,
                    y: DATA.x.map(() => 0),
                    mode: 'lines',
                    name: 'Zero',
                    line: {{ color: 'black', dash: 'dash', width: 1 }}
                }}
            ];

            const layout = {{
                title: 'Residuals: Smoothed - True Function',
                xaxis: {{ title: 'x' }},
                yaxis: {{ title: 'Residual' }},
                legend: {{ x: 0.02, y: 0.98 }},
                margin: {{ t: 50, b: 50, l: 50, r: 20 }}
            }};

            Plotly.react('residuals-plot', traces, layout);
        }}

        // Animation
        function plotAnimation(idx) {{
            const key = getDataKey();
            const bw = snapToAvailable(currentBandwidth, DATA.bandwidths);
            const result = DATA.results[key];
            const smoothed = result.smoothed[bw];

            const x0 = DATA.x[idx];
            const h = bw;

            // Compute weights for current point
            const weights = DATA.x.map(xi => {{
                const u = (xi - x0) / h;
                return Math.abs(u) <= 1 ? 0.75 * (1 - u*u) : 0;
            }});

            // Size based on weights
            const sizes = weights.map(w => 5 + w * 15);
            const colors = weights.map(w => w > 0 ? `rgba(102, 126, 234, ${{0.3 + w * 0.7}})` : 'rgba(200, 200, 200, 0.3)');

            const traces = [
                {{
                    x: DATA.x,
                    y: result.y_noisy,
                    mode: 'markers',
                    name: 'Data',
                    marker: {{ size: sizes, color: colors }}
                }},
                {{
                    x: DATA.x.slice(0, idx + 1),
                    y: smoothed.median.slice(0, idx + 1),
                    mode: 'lines',
                    name: 'EPA Median (built)',
                    line: {{ color: '#2ca02c', width: 3 }}
                }},
                {{
                    x: [x0],
                    y: [smoothed.median[idx]],
                    mode: 'markers',
                    name: 'Current Point',
                    marker: {{ size: 15, color: '#d62728', symbol: 'star' }}
                }}
            ];

            const layout = {{
                title: `Building EPA Local Median - Point ${{idx + 1}} / ${{DATA.n}}`,
                xaxis: {{ title: 'x', range: [0, 2 * Math.PI] }},
                yaxis: {{ title: 'y', range: [-2.5, 3] }},
                shapes: [{{
                    type: 'rect',
                    x0: x0 - h,
                    x1: x0 + h,
                    y0: -2.5,
                    y1: 3,
                    fillcolor: 'rgba(255, 255, 0, 0.15)',
                    line: {{ color: 'orange', dash: 'dash' }}
                }}],
                showlegend: false,
                margin: {{ t: 50, b: 50, l: 50, r: 20 }}
            }};

            Plotly.react('animation-plot', traces, layout);
            document.getElementById('anim-status').textContent = `Point: ${{idx + 1}} / ${{DATA.n}}`;
        }}

        function startAnimation() {{
            if (animationInterval) return;
            animationInterval = setInterval(() => {{
                plotAnimation(animationIndex);
                animationIndex = (animationIndex + 3) % DATA.n;
            }}, 100);
        }}

        function stopAnimation() {{
            if (animationInterval) {{
                clearInterval(animationInterval);
                animationInterval = null;
            }}
        }}

        function resetAnimation() {{
            stopAnimation();
            animationIndex = 0;
            plotAnimation(0);
        }}

        // Tab switching
        function showTab(tabName) {{
            document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
            document.querySelectorAll('.tab-content').forEach(t => t.classList.remove('active'));

            event.target.classList.add('active');
            document.getElementById(tabName + '-tab').classList.add('active');

            if (tabName === 'kernel') plotKernel();
            else if (tabName === 'bandwidth') plotBandwidthComparison();
            else if (tabName === 'residuals') plotResiduals();
            else if (tabName === 'animation') plotAnimation(animationIndex);
        }}

        // Event listeners
        document.getElementById('bandwidth').addEventListener('input', function() {{
            currentBandwidth = parseFloat(this.value);
            document.getElementById('bw-value').textContent = 'h = ' + currentBandwidth.toFixed(1);
            updateMainPlot();
            plotResiduals();
        }});

        document.getElementById('outliers').addEventListener('input', function() {{
            currentOutliers = parseFloat(this.value);
            document.getElementById('outlier-value').textContent = (currentOutliers * 100).toFixed(0) + '%';
            updateMainPlot();
            plotBandwidthComparison();
            plotResiduals();
        }});

        document.getElementById('noise').addEventListener('input', function() {{
            currentNoise = parseFloat(this.value);
            document.getElementById('noise-value').textContent = currentNoise.toFixed(2);
            updateMainPlot();
            plotBandwidthComparison();
            plotResiduals();
        }});

        // Initialize
        updateMainPlot();
        plotKernel();
    </script>
</body>
</html>
'''

    return html


def main():
    output_path = Path(__file__).parent / "epa_smoothing_dashboard.html"

    print("Generating comprehensive EPA Local Median Smoothing dashboard...")
    html = generate_html()

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)

    print(f"Saved: {output_path}")
    print(f"File size: {output_path.stat().st_size / 1024:.1f} KB")


if __name__ == "__main__":
    main()
