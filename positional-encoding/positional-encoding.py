import numpy as np
def positional_encoding(seq_len, d_model, base=10000.0):
    pos = np.arange(seq_len)[:, np.newaxis]
    i = np.arange(0, d_model, 2)
    div_term = np.exp(i * -(np.log(base) / d_model))
    
    pe = np.zeros((seq_len, d_model))
    
    # Fill even columns (0, 2, 4...) with Sine
    pe[:, 0::2] = np.sin(pos * div_term)
    
    # Fill odd columns (1, 3, 5...) with Cosine
    # We use div_term[:pe[:, 1::2].shape[1]] to make sure the math 
    # matches the number of available columns perfectly.
    pe[:, 1::2] = np.cos(pos * div_term[:pe[:, 1::2].shape[1]])
    
    return pe
def add_positional_encoding(x, base=10000.0):
    """
    Add PE to input x of shape (B, T, d_model); return same shape.
    """
    # Write code here
    B, T, d_model = x.shape
    
    pe = positional_encoding(T, d_model, base)
    
    return x + pe