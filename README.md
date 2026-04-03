# destripe

A PDHG (Primal-Dual Hybrid Gradient) based **universal image stripe noise remover**.

## Key Features
- **Multi-directional Decomposition**: Automatically detects and extracts vertical and 4 diagonal stripe noise.
- **Tiling Support**: Efficiently processes massive images by splitting them into tiles with smooth blending.
- **Color Preservation**: For color images, the noise is estimated from the luminance channel to preserve the original hue.
- **Acceleration**: Built on PyTorch with full support for CPU and CUDA acceleration.

## Installation
```bash
pip install .
```

## Quick Start
```python
import numpy as np
from destripe import destripe

# Load image (numpy array: [H, W] or [H, W, 3])
image = ... 

# Perform destriping
clean_image = destripe(
    image, 
    mu1=0.1,       # TV regularization (smoothness)
    mu2=0.001,      # Stripe penalty (sensitivity)
    tiles=2,        # Number of tiles per side (grid)
)
```

## Core Parameters
- `mu1`: Controls the smoothness of the clean image (default: `0.33`).
- `mu2`: Controls the intensity of extracted stripes (default: `0.003`).
- `tiles`: Splits the image into `n x n` tiles (default: `1`).
- `device`: Computation device (`cuda` or `cpu`).