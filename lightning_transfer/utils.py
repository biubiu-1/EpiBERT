"""
PyTorch Lightning utilities converted from TensorFlow EpiBERT implementation.

This module contains utility functions for the Lightning version of EpiBERT,
converted from the original TensorFlow src/utils.py.
"""

import torch
import torch.nn as nn
import numpy as np
import math
from typing import List, Union


def sinusoidal_positional_encoding(seq_length: int, d_model: int, device: torch.device = None) -> torch.Tensor:
    """
    Generate sinusoidal positional encoding as in Vaswani et al. 2017.
    
    Converted from TensorFlow implementation in src/utils.py.
    
    Args:
        seq_length: Length of the sequence
        d_model: Model dimension (must be even)
        device: Device to create tensor on
        
    Returns:
        Positional encoding tensor of shape (seq_length, d_model)
    """
    if d_model % 2 != 0:
        raise ValueError(f"d_model must be even, got {d_model}")
    
    position = torch.arange(0, seq_length, dtype=torch.float32, device=device).unsqueeze(1)
    
    div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32, device=device) * 
                        -(math.log(10000.0) / d_model))
    
    pe = torch.zeros(seq_length, d_model, dtype=torch.float32, device=device)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    
    return pe


def gen_channels_list(num: int, end_channels: int) -> List[int]:
    """
    Generate evenly spaced list of channel numbers from end_channels // 2 to end_channels.
    
    Converted from TensorFlow implementation in src/utils.py.
    
    Args:
        num: Number of layers/channels to generate
        end_channels: Final number of channels
        
    Returns:
        List of channel numbers
    """
    out = [end_channels // (2**i) for i in range(num)]
    return out[::-1]


def exponential_linspace_int(start: int, end: int, num: int, divisible_by: int = 1) -> List[int]:
    """
    Exponentially increasing values of integers.
    
    Converted from TensorFlow implementation in src/utils.py (from Enformer).
    
    Args:
        start: Starting value
        end: Ending value
        num: Number of values to generate
        divisible_by: Values must be divisible by this number
        
    Returns:
        List of exponentially spaced integer values
    """
    def _round(x):
        return int(round(x / divisible_by) * divisible_by)
    
    base = np.exp(np.log(end / start) / (num - 1))
    return [_round(start * base**i) for i in range(num)]


class SoftmaxPooling1D(nn.Module):
    """
    1D Softmax pooling layer for sequence data.
    
    This is a PyTorch implementation that can be used as an equivalent
    to custom pooling operations in the original TensorFlow model.
    """
    
    def __init__(self, kernel_size: int = 2, stride: int = None):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply softmax pooling to 1D sequence data.
        
        Args:
            x: Input tensor of shape (batch, channels, length)
            
        Returns:
            Pooled tensor
        """
        batch_size, channels, length = x.shape
        
        # Reshape for pooling operation
        new_length = length // self.stride
        x_reshaped = x[:, :, :new_length * self.stride].view(
            batch_size, channels, new_length, self.stride
        )
        
        # Apply softmax across the kernel dimension
        weights = torch.softmax(x_reshaped, dim=-1)
        
        # Weighted average
        pooled = torch.sum(x_reshaped * weights, dim=-1)
        
        return pooled


class RotaryPositionalEmbedding(nn.Module):
    """
    Rotary Positional Embedding (RoPE) implementation.
    
    This provides an alternative to sinusoidal embeddings used in the transformer layers.
    """
    
    def __init__(self, dim: int, max_seq_len: int = 4096):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        
        # Precompute the rotary embedding frequencies
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply rotary positional embedding.
        
        Args:
            x: Input tensor of shape (..., seq_len, dim)
            
        Returns:
            Rotary embedding of shape (seq_len, dim)
        """
        seq_len = x.shape[-2]
        t = torch.arange(seq_len, device=x.device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum('i,j->ij', t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb[:seq_len]


def apply_rotary_pos_emb(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """
    Apply rotary positional embedding to input tensor.
    
    Args:
        x: Input tensor
        cos: Cosine component of rotary embedding
        sin: Sine component of rotary embedding
        
    Returns:
        Tensor with rotary positional embedding applied
    """
    x1, x2 = x[..., ::2], x[..., 1::2]
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)