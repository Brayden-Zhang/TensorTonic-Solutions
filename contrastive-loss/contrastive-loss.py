import numpy as np

def contrastive_loss(a, b, y, margin=1.0, reduction="mean") -> float:
    """
    a, b: arrays of shape (N, D) or (D,)  (will broadcast to (N,D))
    y:    array of shape (N,) with values in {0,1}; 1=similar, 0=dissimilar
    margin: float > 0
    reduction: "mean" (default) or "sum"
    Return: float
    """
    # Write code here
    

    a = np.array(a)
    b = np.array(b)
    y = np.array(y)

    if a.ndim == 2:
        axis = 1
    else:
        axis = 0
    d_i = np.linalg.norm(a-b, axis = axis)
    result = y * (d_i **2) + (1-y) * np.maximum(0, margin - d_i)**2
    if reduction == "mean":
        return np.mean(result)
    else:
        return np.sum(result)