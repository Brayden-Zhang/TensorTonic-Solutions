import torch
import torch.nn.functional as F
import math

def scaled_dot_product_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    """
    Compute scaled dot-product attention.
    """
    # Your code here
    d_k = Q.size(-1)
    attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)

    attn_weights = F.softmax(attn_scores, dim=-1) 
    output = torch.matmul(attn_weights, V)
    return output
