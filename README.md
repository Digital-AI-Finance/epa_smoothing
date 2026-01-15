# EPA Kernel-Weighted Local Median Smoothing

Robust nonparametric curve smoothing using the Epanechnikov kernel and weighted median.

## Live Demo

**[Try the Interactive Dashboard](https://digital-ai-finance.github.io/epa_smoothing/dashboard.html)**

## Features

- **Robust to Outliers**: Weighted median ignores outlier magnitude, achieving up to 40% lower error than mean-based smoothers
- **Optimal Kernel**: Epanechnikov kernel minimizes AMISE among second-order kernels
- **Interactive Visualization**: Real-time parameter adjustment with animations

## Quick Start

```python
import numpy as np
from epa_smoothing import epa_local_median

# Noisy data with outliers
x = np.linspace(0, 2*np.pi, 100)
y = np.sin(x) + np.random.normal(0, 0.3, 100)

# Smooth
y_smooth = epa_local_median(x, y, bandwidth=0.5)
```

## Documentation

- [Theory](https://digital-ai-finance.github.io/epa_smoothing/theory.html) - Mathematical background
- [API](https://digital-ai-finance.github.io/epa_smoothing/api.html) - Function documentation
- [Examples](https://digital-ai-finance.github.io/epa_smoothing/examples.html) - Usage examples
- [Code](https://digital-ai-finance.github.io/epa_smoothing/code.html) - Full source code

## Key Results

| Metric | EPA Median | EPA Mean | Improvement |
|--------|------------|----------|-------------|
| RMSE (8% outliers) | 0.108 | 0.184 | 41% |
| Breakdown Point | 50% | 0% | Infinitely more robust |

## License

MIT
