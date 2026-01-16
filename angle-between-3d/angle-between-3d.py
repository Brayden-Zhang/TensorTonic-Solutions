import numpy as np

def angle_between_3d(v, w):
    """
    Compute the angle (in radians) between two 3D vectors.
    """
    v = np.array(v)
    w = np.array(w)
    num = np.dot(v, w)
    v_norm = np.linalg.norm(v)
    w_norm = np.linalg.norm(w)

    if v_norm < 1e-10 or w_norm < 1e-10:
        return np.nan

    norm_product = v_norm * w_norm
    theta = np.arccos(np.clip(num / norm_product, -1, 1))
    return theta