import numpy as np

def batch_norm_forward(x, gamma, beta, eps=1e-5):
    """
    Forward-only BatchNorm for (N,D) or (N,C,H,W).
    """
    # Write code here
    ndim = x.ndim
    if ndim == 2:
        axis = 0
    elif ndim == 4:
        axis = (0, 2, 3)
        # Reshape gamma/beta to (1, C, 1, 1) for broadcasting
        gamma = gamma[None, :, None, None]
        beta = beta[None, :, None, None]
    mu = np.mean(x, axis=axis, keepdims=True)
    var = np.var(x, axis=axis, keepdims=True)
    x_hat = (x - mu) / np.sqrt(var + eps)
    out = gamma * x_hat + beta
    return out