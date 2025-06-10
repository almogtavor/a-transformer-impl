import torch
from torch import nn

from casual_transformer.positional_encoding import PositionalEncoding


def test_positional_enccoding_shapes():
    batch_size = 2
    seq_len = 5
    d_model = 16

    x = torch.randn(batch_size, seq_len, d_model)
    pe = PositionalEncoding(d_model=d_model, max_seq_length=seq_len)
    output = pe(x)

    # Check output shape
    assert output.shape == (batch_size, seq_len, d_model), \
        f"Expected shape {(batch_size, seq_len, d_model)}, got {output.shape}"
    assert not torch.allclose(output, x), "Output should differ from input due to positional encoding"
