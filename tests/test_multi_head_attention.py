import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import math

from src.multi_head_attention import MultiHeadAttention


def test_multi_head_attention():
    torch.manual_seed(0)

    batch_size = 2
    seq_length = 4
    d_model = 8
    num_heads = 2
    mha = MultiHeadAttention(d_model=d_model, num_heads=num_heads)

    # Random input tensors Q, K, V of shape (batch_size, seq_length, d_model)
    Q = torch.randn(batch_size, seq_length, d_model)
    K = torch.randn(batch_size, seq_length, d_model)
    V = torch.randn(batch_size, seq_length, d_model)

    # Optional: create a mask of shape (1, seq_length, seq_length)
    mask = torch.tensor([
        [[1, 1, 0, 0],
         [1, 1, 0, 0],
         [1, 1, 1, 0],
         [1, 1, 1, 1]]
    ], dtype=torch.bool)

    # Forward pass
    output = mha(Q, K, V, mask=mask)

    print("Output shape:", output.shape)
    assert output.shape == (batch_size, seq_length, d_model), "Output shape mismatch"
    print("MHA test passed!")

test_multi_head_attention()
