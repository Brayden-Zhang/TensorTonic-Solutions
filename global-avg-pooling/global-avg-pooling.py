import numpy as np

def global_avg_pool(x):
    """
    Compute global average pooling over spatial dims.
    Supports (C,H,W) => (C,) and (N,C,H,W) => (N,C).
    """
    # Write code here
    x = np.array(x)
    if x.ndim in (3,4):
        return np.mean(x, axis=(-1,-2))

    raise ValueError()