'''
Date: 2025-04-11 11:58:03
LastEditors: BHM-Bob 2262029386@qq.com
LastEditTime: 2025-05-24 21:17:06
Description: 
'''
import torch


@torch.compile
def add_rope(x: torch.Tensor):
    """
    Add Rotary Position Embedding (RoPE) to the input tensor.
    Args:
        x: Input tensor of shape [N, L, D].
    Returns:
        Tensor with RoPE applied, shape [N, L, D].
    """
    batch_size, seq_len, dim = x.shape
    half_dim = dim // 2
    freq_seq = torch.arange(half_dim, device=x.device, dtype=torch.float32)
    inv_freq = 1.0 / (10000 ** (freq_seq / half_dim))
    pos_seq = torch.arange(seq_len, device=x.device, dtype=torch.float32)
    sinusoid_inp = torch.einsum("i,j->ij", pos_seq, inv_freq)
    sin, cos = sinusoid_inp.sin(), sinusoid_inp.cos()
    x1, x2 = x[..., :half_dim], x[..., half_dim:]
    x = torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
    return x