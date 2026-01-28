import numpy as np

def feed_forward(x: np.ndarray, W1: np.ndarray, b1: np.ndarray,
                 W2: np.ndarray, b2: np.ndarray) -> np.ndarray:
    """
    Apply position-wise feed-forward network.
    Structure: linear -> ReLU -> linear
    """
    # Layer 1: Linear transformation to hidden dimension
    # x shape: (batch, seq_len, d_model) -> z shape: (batch, seq_len, d_ff)
    z = np.dot(x, W1) + b1
    
    # Activation: Element-wise ReLU
    # Replaces your relu(z) function for compatibility with arrays
    a = np.maximum(0, z)
    
    # Layer 2: Linear transformation back to d_model
    # a shape: (batch, seq_len, d_ff) -> output shape: (batch, seq_len, d_model)
    output = np.dot(a, W2) + b2
    
    return output