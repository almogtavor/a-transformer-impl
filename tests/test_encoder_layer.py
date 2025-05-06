import torch
from torch import nn

from src.encoder_layer import EncoderLayer  # Assuming this is the correct path to your EncoderLayer implementation

def test_encoder_layer_shapes():
    batch_size = 2
    seq_len = 5
    d_model = 16
    num_heads = 4
    d_ff = 64
    dropout_rate = 0.1

    x = torch.randn(batch_size, seq_len, d_model)
    encoder_layer = EncoderLayer(d_model=d_model, num_heads=num_heads, d_ff=d_ff, dropout_rate=dropout_rate)
    output = encoder_layer(x)

    # Check output shape
    assert output.shape == (batch_size, seq_len, d_model), \
        f"Expected shape {(batch_size, seq_len, d_model)}, got {output.shape}"

    # Check that the output is not the same as input (since it's going through transformations)
    assert not torch.allclose(output, x), "Output should differ from input after transformations"
