import numpy as np

def dropout(x: np.ndarray, p: float = 0.5, training: bool = True) -> np.ndarray:
    """
    Apply inverted dropout to input.
    p: probability of dropping a neuron (0.5 means half are dropped)
    """
    if not training:
        return x

   
    mask = np.random.binomial(1, 1 - p, size=x.shape)

  
    mask = mask / (1 - p)

    return x * mask